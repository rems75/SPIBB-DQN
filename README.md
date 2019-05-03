# Prerequisites

SPIBB-DQN is implemented in Python 3 and requires PyTorch.

# Commands

To generate a dataset, run:

`python baseline.py baseline/dummy_env/ weights.pt --generate_dataset --dataset_size 1000`

where ``baseline/dummy_env/`` is the path to the baseline and ``weights.pt`` is the baseline filename. The dataset will be generated in ``baseline/dummy_env/dataset`` by default.

To train a policy on that dataset, define the training parameters in a yaml file e.g. `config_batch` (in particular, that file should contain the path to the baseline and dataset to use) and then run:

`ipython train.py -- -o batch True --config config_batch`

To specify different learning types or parameters, either change the `config_batch` file or pass options to the command line, e.g. `--options epsilon_soft 1.5`, or `--options learning_type pi_b`.

We have provided the baseline we used in our experiment in `baseline/dummy_env`. To train a new one, define the training parameters in a yaml file e.g. `config` and then run:

`ipython train.py -- --domain dummy --config config_dummy`
