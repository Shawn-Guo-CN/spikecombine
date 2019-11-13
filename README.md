# spikecombine

## Pipeline for running the combiner

### 1. Run DataGenerator

Currently, the `config` part is not active. Thus, all synthetic data need to be loaded from some folder. For example, data generator can run with a path `input_path` with the following files:
 - a.h5
 - b.h5
 - c.h5

The code would like:

 ```python
gen = DataGenerator(<useless_config_path>, ['klusta','tridesclous'])
gen.run(<gen_output_path>, verbose=True)
 ```

 ### 2. Run DataLoader

 Once the DataGenerator finished running, it would generate a folder with following structure:

```python
        --------------------------
        |   - <dir_name>         |
        |   | - ground_truth     |
        |   |   | - a.h5         |
        |   |   | - b.h5         |
        |   | - recordings       |
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
```

Then, we can run a DataLoader which takes the folder as the input, e.g.

```python
loader = DataLoader(<gen_output_path>)
```

### 3. Fit Model

With a DataLoader, we could fit a model by

```python
ssc = SpikeSortersCombiner(well_detected_threshold=0.9)
ssc.fit(loader)
```

Note, `well_detected_threshold` is a hyperparameter for the model.

Of course, the parameters of the model could be saved and loaded as follow:

```python
ssc.save(<param_path>)
ssc.load(<param_path>)
```

### 4. Curate with model

Once we have a fitted model, we could curate a sorting like

```python
recording = se.MEArecRecordingExtractor(<path_to_recording_h5_file>)
sorting_KL = se.MEArecSortingExtractor(<path_to_klusta_result_h5_file>)
sorting_TS = se.MEArecSortingExtractor(<path_to_tridesclous_result_h5_file>)

sortings = {
    'klusta': sorting_KL,
    'tridesclous': sorting_TS
}

curated = ssc.predict(sortings, recording, 'klusta')
```

Note that, we need sorting results from other sorters to curate the target sorting result.

## A Minimum Script

```python
import matplotlib.pyplot as plt

import spikeinterface.extractors as se
import spikeinterface.widgets as sw
import spikeinterface.comparison as sc

from spikecombine import DataGenerator
from spikecombine import DataLoader
from spikecombine import SpikeSortersCombiner

config_path = <useless_path>
sorter_list = [
    'klusta',
    'tridesclous',
]

gen = DataGenerator(config_path, sorter_list)
gen.run(<data_path>)

loader = DataLoader(<data_path>)

ssc = SpikeSortersCombiner(well_detected_threshold=0.9)
ssc.fit(loader)

ssc.save(<param_path>)

ssc = SpikeSortersCombiner()
ssc.load(<param_path>)

print(ssc._params['positive']['means'])
print(ssc._params['negative']['means'])

print(ssc._params['positive']['covars'])
print(ssc._params['negative']['covars'])


recording = se.MEArecRecordingExtractor(<a_recording_under_data_path>)
sorting_true = se.MEArecSortingExtractor(<corresponding_ground_truth_under_data_path>)

sorting_KL = ss.run_klusta(recording)
sorting_TS = ss.run_tridesclous(recording)

sortings = {
    'klusta': sorting_KL,
    'tridesclous': sorting_TS
}

print(sortings['klusta'].get_unit_ids())

curated = ssc.predict(sortings, recording, 'klusta')
print(curated.get_unit_ids())

cmp_gt_KL = sc.compare_sorter_to_ground_truth(sorting_true, sortings['klusta'], exhaustive_gt=True)
sw.plot_agreement_matrix(cmp_gt_KL, ordered=True)
plt.show()
```

### Todo Note

To generate simulated data based on more sensible configurations, the following functions need to be implemented:

 1. `DataGenerator._load_dataset_configs`, this is to load all config files (if a config file contains only one configuration for generating data).
 2. `DataGenerator._parse_dataset_config`, this may be needed to parse config files.
 3. `DataGenerator.generate_a_dataset`, this function **has to be over written** to generate different simulated data based on all sorts of configurations.