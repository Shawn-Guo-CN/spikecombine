import pickle
import numpy as np

class SpikeSortersCombiner(object):
    """
    This is the model for combining results from different sorters. 

    Assume that we want to combine K different sorters, then the key parameters are given as follows:
    1. K mean vectors (1*N) and the corresponding K covariance matrix (N*N);
    2. categorical distribution on K different sorters.
    """
    def __init__(self):
        self._sorter_names = None

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

        self._sortername_to_id = None

    def _generate_data_for_one_sorter(self):
        raise NotImplementedError

    def _fit_one_sorter(self, data):
        raise NotImplementedError

    @staticmethod
    def _set_prior(K, mode:str='uniform'):
        if mode == 'uniform':
            return np.ones(K, dtype=float) / float(K)
        else:
            rasie NotImplementedError

    def fit(self, dataloader, prior_mode:str='uniform') -> None:
        K = len(dataloader.sorter_names)

        _positive_means = []
        _positive_covars = []
        _positive_prior = self._set_prior(K, prior_mode)

        _negative_means = []
        _negative_covars = []
        _negative_prior = self._set_prior(K, prior_mode)

        for sorter_name in dataloader.sorter_names():
            positive_data, negative_data = self._generate_data_for_one_sorter(dataloader.sorter_sortings[sorter_name])

            pos_mean, pos_covar = self._fit_one_sorter(positive_data)
            _positive_means.append(pos_mean)
            _positive_covars.append(pos_covar)

            neg_mean, neg_covar = self._fit_one_sorter(negative_data)
            _negative_means.append(neg_mean)
            _negative_covars.append(neg_covar)

            self._sortername_to_id[sorter_name] = len(_positive_means) - 1

    def predict(self):
        assert all(), """
            Please fit or load model before prediction.
        """
        raise NotImplementedError

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
