import pickle
import numpy as np
from scipy.stats import multivariate_normal

import spikeinterface.comparison as sc
import spikeinterface.toolkit as st

class SpikeSortersCombiner(object):
    """
    This is the model for combining results from different sorters. 

    Assume that we want to combine K different sorters, then the key parameters are given as follows:
    1. K mean vectors (1*N) and the corresponding K covariance matrix (N*N);
    2. categorical distribution on K different sorters.
    """
    def __init__(self, well_detected_threshold:float=0.7):
        self.metric_names = ['isolation_distance', 'snr']

        # 'amplitude_cutoff', 'silhouette_score', , 'l_ratio', 'd_prime', 'nn_hit_rate', 'presence_ratio'
        # firing_rate',
        
        self._well_detected_threshold = well_detected_threshold

        # the following 3 are parameters
        self._params = {
            'positive':{
                'means': None, # should be a [K, N] np array
                'covars': None, # should be a [K, N, N] np array
                'sorter_prior': None, # should be a [1, K] np array
            },
            'negative':{
                'means': None, # should be a [K, N] np array
                'covars': None, # should be a [K, N, N] np array
                'sorter_prior': None, # should be a [1, K] np array
            }
        }

        self._sorter_names = []
        self._sortername_to_id = {}

    def _generate_data_for_one_sorter(self, sorter_sortings, gt_sortings, recordings):
        """
        Generate positive and negative data matrices based on the sorter's results and ground truth results.
        """
        assert sorter_sortings.keys() == gt_sortings.keys(), """
            The dataset stored in ground truth results and sorters' results are different.
        """

        positive_metrics = []
        negative_metrics = []

        for dataset_name in sorter_sortings.keys():
            sorting_gt = gt_sortings[dataset_name]
            sorting_sorter = sorter_sortings[dataset_name]
            recording = recordings[dataset_name]

            gt_comparison = sc.compare_sorter_to_ground_truth(sorting_gt, sorting_sorter, exhaustive_gt=True)
            well_detected_units = gt_comparison.get_well_detected_units(self._well_detected_threshold)

            metric_matrix = st.validation.MetricCalculator(sorting_sorter, recording)
            metric_matrix.compute_metrics(metric_names=self.metric_names)
            metrics_df = metric_matrix.get_metrics_df()

            for unit_id in sorting_sorter.get_unit_ids():
                # 1:3 is because that we only have 2 kinds of features
                appending_value = metrics_df.loc[metrics_df['unit_ids'] == unit_id].values[0][1:3].astype('float')
                if unit_id in well_detected_units and not np.isnan(appending_value).any():
                    positive_metrics.append(appending_value)
                elif not np.isnan(appending_value).any():
                    negative_metrics.append(appending_value)

        assert len(positive_metrics) > 0
        assert len(negative_metrics) > 0

        positive_metrics = np.array(positive_metrics, dtype=float)
        negative_metrics = np.array(negative_metrics, dtype=float)
        
        return positive_metrics, negative_metrics

    def _fit_one_sorter(self, data):
        mean = np.mean(data, axis=0)
        cov = np.cov(data, rowvar=0)
        assert cov.shape[1] == data.shape[1], """
            Covariance matrix fitted on the data is not fully ranked.
        """
        return mean, cov

    @staticmethod
    def _set_prior(K, mode:str='uniform'):
        if mode == 'uniform':
            return np.ones(K, dtype=float) / float(K)
        else:
            raise NotImplementedError

    def fit(self, dataloader, prior_mode:str='uniform', verbose:bool=False) -> None:
        K = len(dataloader.sorter_names)

        _positive_means = []
        _positive_covars = []
        _positive_prior = self._set_prior(K, prior_mode)

        _negative_means = []
        _negative_covars = []
        _negative_prior = self._set_prior(K, prior_mode)

        for sorter_name in dataloader.sorter_names:
            positive_data, negative_data = self._generate_data_for_one_sorter(
                    dataloader.gt_sortings,
                    dataloader.sorter_sortings[sorter_name],
                    dataloader.recordings
                )

            pos_mean, pos_covar = self._fit_one_sorter(positive_data)
            _positive_means.append(pos_mean)
            _positive_covars.append(pos_covar)

            neg_mean, neg_covar = self._fit_one_sorter(negative_data)
            _negative_means.append(neg_mean)
            _negative_covars.append(neg_covar)

            self._sortername_to_id[sorter_name] = len(_positive_means) - 1

        self._params['positive']['means'] = np.asarray(_positive_means, dtype='float')
        self._params['positive']['covars'] = np.asarray(_positive_covars, dtype='float')
        self._params['positive']['sorter_prior'] = np.asarray(_positive_prior, dtype='float')

        self._params['negative']['means'] = np.asarray(_negative_means, dtype='float')
        self._params['negative']['covars'] = np.asarray(_negative_covars, dtype='float')
        self._params['negative']['sorter_prior'] = np.asarray(_negative_prior, dtype='float')

        if verbose:
            print('Fitted to the provided data.')

    @staticmethod
    def _get_nan_dims(x):
        ids = []
        # TODO: check whether size could be applied to 1D array.
        for i in range(x.size):
            if np.isnan(x[i]):
                ids.append(i)

        return ids

    @staticmethod
    def _exclude_dimensions(x, ids):
        if x.ndim == 1:
            return np.delete(x, ids)
        elif x.ndim == 2:
            x = np.delete(x, ids, 0)
            return np.delete(x, ids, 1)
        else:
            raise NotImplementedError
    
    def _calculate_posterior_prob(self, unit_metric):
        nan_dims = self._get_nan_dims(unit_metric)

        unit_metric = self._exclude_dimensions(unit_metric, nan_dims)

        positive_posts_metric = {}
        negative_posts_metric = {}

        for sorter in self._sorter_names:
            _params = self.get_params_for_sorter(sorter)

            positive_mean = self._exclude_dimensions(_params['positive_mean'], nan_dims)
            positive_covar = self._exclude_dimensions(_params['positive_covar'], nan_dims)

            negative_mean = self._exclude_dimensions(_params['negative_mean'], nan_dims)
            negative_covar = self._exclude_dimensions(_params['negative_covar'], nan_dims)

            positive_p = multivariate_normal(mean=positive_mean, cov=positive_covar).pdf(unit_metric)
            negative_p = multivariate_normal(mean=negative_mean, cov=negative_covar).pdf(unit_metric)

            positive_posts_metric[sorter] = positive_p
            negative_posts_metric[sorter] = negative_p

        return positive_posts_metric, negative_posts_metric

    @staticmethod
    def _compare_one_sorter_with_all_others(sorting, all_sortings):
        """
        Return:

        A 2D np array with shape [N_Units, K]
        """
        agreements = {}

        for other_sorter_name in sorted(all_sortings.keys()):
            compare = sc.compare_two_sorters(sorting1=all_sortings[other_sorter_name], sorting2=sorting, 
                                             sorting1_name='other', sorting2_name='original')
            _agreement = compare.agreement_scores.to_numpy().max(axis=0)
            agreements[other_sorter_name] = _agreement

        return agreements

    def predict(self, sortings, recording, sorter_name:str, threshold:float=0.7):
        """
        Parameters:

        sortings: a dictionary that contains all the sorting results
        recording: the original recording for generating all the sorting results
        sorter_name: the sorting result we want to evaluate
        threshold: the threshold for agreement that another sorter also detects a unit in 'sorter_name'
        """
        # TODO: metric calculator could be passed in, epoch_start/end as well.

        assert all(True for t in self._params.values() for v in t.values() if v is not None), """
            Please fit or load model before prediction.
        """

        assert set(sortings.keys()).issubset(set(self._sorter_names)), """
            Input sortings contains results from un-modelled sorters.
        """

        mc = st.validation.MetricCalculator(sortings[sorter_name], recording)
        mc.compute_metrics(metric_names=self.metric_names)
        metric_matrix = mc.get_metrics_df()

        agreements = self._compare_one_sorter_with_all_others(sortings[sorter_name], sortings)

        unit_ids = mc.get_unit_ids()
        units_to_be_excluded = []

        for idx, unit_id in enumerate(unit_ids):
            unit_metric = metric_matrix.loc[metric_matrix['unit_ids'] == unit_id].values[0][1:3].astype('float')
            p_post_metric, neg_post_metric = self._calculate_posterior_prob(unit_metric)

            positive_ps = []
            negative_ps = []

            for _sorter_name in sortings.keys():
                if agreements[_sorter_name][idx] >= threshold:
                    positive_ps.append(agreements[_sorter_name][idx] * p_post_metric[_sorter_name])
                else:
                    negative_ps.append((1. - agreements[_sorter_name][idx]) * neg_post_metric[_sorter_name])
            
            positive_p = 0. if len(positive_ps) == 0 else np.mean(positive_ps)
            negative_p = 0. if len(negative_ps) == 0 else np.mean(negative_ps)

            if positive_p < negative_p:
                units_to_be_excluded.append(unit_id)

        cse = st.curation.CurationSortingExtractor(sortings[sorter_name])
        cse.exclude_units(units_to_be_excluded)
        
        return cse

    def get_params_for_sorter(self, sortername:str=None):
        return {
            'positive_mean': self._params['positive']['means'][self._sortername_to_id[sortername]],
            'positive_covar': self._params['positive']['covars'][self._sortername_to_id[sortername]],
            'negative_mean': self._params['negative']['means'][self._sortername_to_id[sortername]],
            'negative_covar': self._params['negative']['covars'][self._sortername_to_id[sortername]],
        }

    def load(self, path:str):
        with open(path, 'rb') as in_file:
            loaded_obj = pickle.load(in_file)

        self._params = loaded_obj['params']
        self._sortername_to_id = loaded_obj['sortername2id']
        self._sorter_names = sorted(self._sortername_to_id.keys())

        del loaded_obj

    def save(self, path:str):
        with open(path, 'wb') as out_file:
            pickle.dump(
                {
                    'params': self._params,
                    'sortername2id': self._sortername_to_id
                },
                out_file
            )
