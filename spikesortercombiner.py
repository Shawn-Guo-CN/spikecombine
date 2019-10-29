import pickle

class SpikeSortersCombiner(object):
    """
    This is the model for combining results from different sorters. 

    Assume that we want to combine K different sorters, then the key parameters are given as follows:
    1. K mean vectors and the corresponding K covariance matrix;
    2. categorical distribution on K different sorters.
    """
    def __init__(self):
        self._sorter_names = None

        # the following 3 are parameters
        self._means = None
        self._covars = None
        self._sorter_prior = None
        
        self._name_to_mean = None
        self._name_to_covar = None
        self._name_to_sorter_prior = None

    def _filter_out_wrong_data(self):
        raise NotImplementedError

    def _generate_data_matrix_for_one_sorter(self):
        raise NotImplementedError

    def fit(self, dataloader):
        gt_sortings = dataloader.gt_sortings
        sorter_sortings = dataloader.sorter_sortings

        self._sorter_names = sorter_sortings.keys()

        raise NotImplementedError

    def predict(self):
        assert self.means is not None and self.covars is not None and self.sorter_prior is not None, """
            Please fit or load model before prediction.
        """
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError
