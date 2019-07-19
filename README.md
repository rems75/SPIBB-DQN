## Prerequisites

This repo contains implementations of SPIBB-DQN and Soft-SPIBB-DQN, plus the DQN and RaMDP baselines.
The code is implemented in Python 3 and requires PyTorch.

## Commands

To generate a dataset for the helicopter environment, run:

`python baseline.py baseline/helicopter_env/ weights.pt --generate_dataset --dataset_size 1000`

where ``baseline/helicopter_env/`` is the path to the baseline and ``weights.pt`` is the baseline filename. The dataset will be generated in ``baseline/helicopter_env/dataset`` by default. We have provided the baseline used in our experiment in `baseline/helicopter_env`. To train a new one, define the training parameters in a yaml file e.g. `config` and run:

`ipython train.py -- --config_file config.yaml -o domain helicopter `

To train a policy on that dataset, define the training parameters in a yaml file e.g. `config_batch` (in particular, that file should contain the path to the baseline and dataset to use) and then run:

`ipython train.py -- -o batch True --config_file config_batch.yaml`

To specify different learning types or parameters, either change the `config_batch` file or pass options to the command line, e.g. `--options learning_type ramdp`, or `--options minimum_count 5` (see the file `config_batch.yaml` for the existing options). In particular, the learning_type parameter can be regular (ie DQN), pi_b (ie SPIBB-DQN), soft_sort (ie Soft-SPIBB-DQN) or ramdp.

Results for the run will be saved in the dataset folder, in a csv file named `{algo_runid_parameter}`.

We also provide a baseline for CartPole, use the following commands to train on that environment:

`python baseline.py baseline/cartpole_env/ weights.pt --generate_dataset --dataset_size 1000`

`ipython train.py -- -o batch True --config_file config_batch_cartpole.yaml`.


## Reference

Please use the following bibtex entry if you use this code for SPIBB:

```
@inproceedings{Laroche2019,
    title={Safe Policy Improvement with Baseline Bootstrapping},
    author={Laroche, Romain and Trichelair, Paul and Tachet des Combes, R\'emi},
    booktitle={Proceedings of the 36th International Conference on Machine Learning (ICML)},
    year={2019}
}
```

Please use the following bibtex entry if you use this code for Soft-SPIBB:

```
@inproceedings{Laroche2019,
    title={Safe Policy Improvement with Soft Baseline Bootstrapping},
    author={Nadjahi, Kimia and Laroche, Romain and Tachet des Combes, R\'emi},
    booktitle={Proceedings of the 2019 European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD)},
    year={2019}
}
```