import os
import random
from pathlib import Path

import spikeinterface.sorters as ss
import spikeinterface.extractors as se


class DataGenerator(object):
    """
    Generate datasets based on specified config files, and then save the 
    """
    def __init__(self, config_path:str=None, sorters:list=[]) -> None:
        config_path = Path(config_path)
        self._config_files = [config_path / f for f in os.listdir(config_path) if os.path.isfile(config_path / f)]

        assert set(sorters).issubset(set(ss.available_sorters())), """
            Please check the input sorter names and the installed sorters.
        """
        self._sorter_names = sorters

    def _load_dataset_configs(self):
        # TODO: the followings are faked
        configs = []

        for idx, f in enumerate(self._config_files):
            new_item = {
                'name': chr(ord('a') + idx),
                'file_path': f,
            }
            configs.append(new_item)

        return configs

    def _parse_dataset_config(self):
        # TODO: implement this function
        raise NotImplementedError

    def generate_a_dataset(self, config:dict=None):
        """
        Generate a dataset (recording, sorting_true) with parameter from config file.

        TODO: currently, the config file is out-of-operation.
        """
        # return se.example_datasets.toy_example(num_channels=10, duration=50)
        recording = se.MEArecRecordingExtractor(config['file_path'])
        sorting_true = se.MEArecSortingExtractor(config['file_path'])
        return recording, sorting_true


    @staticmethod
    def _create_dir(dir:str, verbose:bool=False):
            try:
                os.mkdir(dir)
            except OSError:
                print("Creation of the directory %s failed" % dir)
            else:
                if verbose:
                    print("Successfully created the directory %s " % dir)

    def _create_dirs(self, out_dir:str, verbose:bool=False) -> None:
        out_dir = Path(out_dir)

        assert not os.path.exists(out_dir), """
            The specified dir has existed, please specify another one.
        """

        self._create_dir(out_dir)
        self._create_dir(out_dir / 'ground_truth')
        self._create_dir(out_dir / 'recordings')
        self._create_dir(out_dir / 'sorter_results')
        self._create_dir(out_dir / 'tmp')
        for sorter_name in self._sorter_names:
            self._create_dir(out_dir / 'sorter_results' / sorter_name)

    def run(self, out_dir:str, verbose:bool=False) -> None:
        self._create_dirs(out_dir, verbose=verbose)
        
        config_list = self._load_dataset_configs()
        
        for config in config_list:
            if verbose:
                print('processing ' + str(config['file_path'].absolute()))

            recording, sorting_true = self.generate_a_dataset(config)
            sampling_freq = recording.get_sampling_frequency()

            se.MEArecSortingExtractor.write_sorting(
                sorting=sorting_true,
                save_path=Path(out_dir) / 'ground_truth' / (config['name']+'.h5'),
                sampling_frequency=sampling_freq,
            )
            if verbose:
                print("Successfully saved the true sorting of generated date to " + \
                                    str((Path(out_dir) / 'ground_truth' / (config['name']+'.h5')).absolute()))

            se.MEArecRecordingExtractor.write_recording(
                recording=recording,
                save_path=Path(out_dir) / 'recordings' / (config['name']+'.h5'),
            )
            if verbose:
                print("Successfully saved the generated recording to " + \
                                    str((Path(out_dir) / 'recording' / (config['name']+'.h5')).absolute()))

            for sorter_name in self._sorter_names:
                if verbose:
                    print('processing by ' + sorter_name)
                    
                sorting = ss.sorterlist.run_sorter(
                    sorter_name, recording, 
                    output_folder= Path(out_dir) / 'tmp' / (sorter_name + '_' + config['name']),
                    delete_output_folder=True
                )
                
                se.MEArecSortingExtractor.write_sorting(
                    sorting=sorting,
                    save_path=Path(out_dir) / 'sorter_results' / sorter_name / (config['name']+'.h5'),
                    sampling_frequency=sampling_freq,
                )

                if verbose:
                    print("Successfully saved the sorting of generated date to " + \
                                    str((Path(out_dir) / 'sorter_results' / sorter_name / (config['name']+'.h5')).absolute()))


        if verbose:
            print("Successfully genereated dataset and corresponding sorter results.")


