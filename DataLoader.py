import os
from pathlib import Path

import spikeinterface.sorters as sorters
import spikeinterface.extractors as se


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
        sub_dir_names = sorted(sub_dir_names)

        assert sub_dir_names == ['ground_truth', 'sorter_results']

        self._ground_truth_path = Path(data_path) / 'ground_truth'
        self._sorter_results_path = Path(data_path) / 'sorter_results'

        self.dataset_names = [_file_name.split('.')[0] for _file_name in self._ground_truth_path.glob('*.h5')]
        self.sorter_names = [_dir_name for _dir_name in self._sorter_results_path.iterdir() if _dir_name.is_dir()]

        self.gt_sortings = self._build_ground_truth_sortings()
        self.sorter_sortings = self._build_sorters_sortings()
    
    @staticmethod
    def _loading_sorting(file_path):
        return se.MEArecSortingExtractor(file_path)

    def _build_ground_truth_sortings(self):
        _gt_sortings = {}
        for dataset_name in self.dataset_names:
            _gt_sortings[dataset_name] = self._loading_sorting(self._ground_truth_path / (dataset_name + '.h5'))
        return _gt_sortings

    def get_ground_truth_sortings(self):
        return self.gt_sortings

    def _build_sorters_sortings(self):
        _sorter2dataset = {}

        for sorter_name in self.sorter_names:
            _dataset2result = {}
            for dataset_name in self.dataset_names:
                _dataset2result[dataset_name] = \
                    self._loading_sorting(self._sorter_results_path/ sorter_name / (dataset_name + '.h5'))
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
    

