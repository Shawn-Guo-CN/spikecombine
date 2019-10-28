# spikecombine

### Note

Currently, the `config` part is not active. Thus, all simulated data are generated simply by `se.example_datasets.toy_example(num_channels=10, duration=50)`.

To generate simulated data based on more sensible configurations, the following functions need to be implemented:

 1. `DataGenerator._load_dataset_configs`, this is to load all config files (if a config file contains only one configuration for generating data).
 2. `DataGenerator._parse_dataset_config`, this may be needed to parse config files.
 3. `DataGenerator.generate_a_dataset`, this function **has to be over written** to generate different simulated data based on all sorts of configurations.

### Minimum Tutorial

```python
from spikecombine.datagenerator import DataGenerator
from spikecombine.dataloader import DataLoader

config_path = 'sample_config_dir'

sorter_list = [
    'klusta',
    'tridesclous',
]

gen = DataGenerator(config_path, sorter_list)
gen.run('sample_dir')

loader = DataLoader('sample_dir')
print(loader.gt_sortings)
print(loader.sorter_sortings)

```