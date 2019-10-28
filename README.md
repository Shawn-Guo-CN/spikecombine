# spikecombine

### Minimum Tutorial

```python
from spikecombine.datagenerator import DataGenerator
from spikecombine.dataloader import DataLoader

config_path = 'sample_dir'

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