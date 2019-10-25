import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join('e:\\Edin\\RA\\projects', 'spikeinterface')))
sys.path.insert(0, os.path.abspath(os.path.join('e:\\Edin\\RA\\projects', 'spikeextractors')))
sys.path.insert(0, os.path.abspath(os.path.join('e:\\Edin\\RA\\projects', 'spiketoolkit')))
sys.path.insert(0, os.path.abspath(os.path.join('e:\\Edin\\RA\\projects', 'spikesorters')))
sys.path.insert(0, os.path.abspath(os.path.join('e:\\Edin\\RA\\projects', 'spikecomparison')))
sys.path.insert(0, os.path.abspath(os.path.join('e:\\Edin\\RA\\projects', 'spikewidgets')))
sys.path.insert(0, os.path.abspath(os.path.join('e:\\Edin\\RA\\projects', 'spikemetrics')))
sys.path.insert(0, os.path.abspath(os.path.join('e:\\Edin\\RA\\projects', 'shawn')))
sys.path.append('D:\\Anaconda3\\envs\\spike_dev\\lib\\site-packages')
sys.path.append('D:\\Anaconda3\\envs\\spike_dev')
sys.path.append('D:\\Anaconda3\\envs\\spike_dev\\lib')
sys.path.append('D:\\Anaconda3\\envs\\spike_dev\\python37.zip')
sys.path.append('D:\\Anaconda3\\envs\\spike_dev\\DLLs')
print('working dir: ', os.getcwd())

#################################### beginning of script ####################################

import os
from pathlib import Path

import spikeinterface.sorters as sorters


class DataLoader(object):
    """
        Loading results from different sorters and corresponding ground truth.
        All are stored in MEArec .h5 formats.

        Assume that we have 2 different sorters, A and B, and 2 different datasets, a and b.
        Then, the input directory should have the following structure:

        --------------------------
        |   - <dir_name>         |
        |   | - ground_truth     |
        |   |   | - a.h5         |
        |   |   | - b.h5         |
        |   | - sorter_results   |
        |   |   | - A            |
        |   |   |   | - a.h5     |
        |   |   |   | - b.h5     |
        |   |   | - B            |
        |   |   |   | - a.h5     |
        |   |   |   | - b.h5     |
        --------------------------

    """
    def __init__(self, data_path:str, verbose:bool=False) -> bool:
        sub_dir_names = [_dir_name for _dir_name in Path(data_path).iterdir() if _dir_name.is_dir()]
        assert sub_dir_names == ['sorter_results', 'ground_truth'] or \
            sub_dir_names == ['ground_truth', 'sorter_results']
        self._sorter_results_path = Path(data_path) / 'sorter_results'
        self._ground_truth_path = Path(data_path) / 'ground_truth'

        self.dataset_names = [_file_name.split('.')[0] for _file_name in self._ground_truth_path.glob('*.h5')]
        self.sorter_names = [_dir_name for _dir_name in self._sorter_results_path.iterdir() if _dir_name.is_dir()]

        self.gt_sortings = self._build_ground_truth_sortings()
        self.sorter_sortings = self._build_sorters_sortings()
    
    def _loading_ground_truth(self):
        raise NotImplementedError

    def _loading_a_sorter_results(self):
        raise NotImplementedError

    def _loading_all_sorter_results(self):
        raise NotImplementedError

    def _build_ground_truth_sortings(self):
        _gt_sortings = {}
        for dataset_name in self.dataset_names:
            _gt_sortings[dataset_name] = \
                se.MEArecSortingExtractor(file_path=self._ground_truth_path/ (dataset_name + '.h5'))
        return _gt_sortings

    def get_ground_truth_sortings(self):
        return self.gt_sortings

    def _build_sorters_sortings(self):
        _sorter2dataset = {}

        for sorter_name in self.sorter_names:
            _dataset2result = {}
            for dataset_name in self.dataset_names:
                _dataset2result[dataset_name] = \
                    se.MEArecSortingExtractor(file_path=self._sorter_results_path/ sorter_name / (dataset_name + '.h5'))
            _sorter2dataset[sorter_name] = _dataset2result

        return _sorter2dataset

    def get_sorter_sortings(self, sorter=None):
        assert sorter is not None
        assert sorter in self.sorter_names, """
            The data path only contains results from 
        """ + ' '.join(self.sorter_names)

        return self.sorter_sortings[sorter]

    def get_all_sorter_sortings(self):
        return self.sorter_sortings
    

