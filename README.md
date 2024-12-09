# Floco

Official repository for Floco ([Federated Learning over Connected Modes](https://openreview.net/pdf?id=JL2eMCfDW8)), published at NeurIPS'24.  

We offer implementations for two federated learning frameworks:
- [FL-Bench](https://github.com/KarhouTam/FL-bench): [Pull request under review](https://github.com/KarhouTam/FL-bench/pull/138)
- [Flower](https://github.com/adap/flower): In progress

This repository is based on the FL-bench implementation and includes additional non-IID splits and utility functions used in the paper.


# Installation
```sh
python -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

# Datasets
We have benchmarked our method against multiple different non-IID splits. 
In order to generate the dataset splits as proposed in the paper, the following commands should be run beforehand.
### CIFAR-10
#### Dirichlet(0.3)
```py
python generate_data.py -d cifar10 -a 0.3 -cn 100 --least_samples 600 --val_ratio 0.2 --test_ratio 0.0 --seed <seed>
```
#### Five-Fold
```py
python generate_data.py -d cifar10 -f 5 -cn 100 --least_samples 600 --val_ratio 0.2 --test_ratio 0.0 --seed <seed>
```
### FEMNIST
Before running this script, the [LEAF](https://leaf.cmu.edu/) dataset has to be downloaded and preprocessed as described [here](https://github.com/KarhouTam/FL-bench/tree/master/data/femnist).
```py
python generate_data.py -d femnist
```
# Reproducing the results
```py
python main.py [--config-name <CONFIG_NAME>] [method=<METHOD_NAME>] [--seed=<SEED>]
```
- `--config-name`: Name of `.yaml` config file (w/o the `.yaml` extension) in the `config/` directory.
- `method`: The algorithm's name, e.g., `method=fedavg` which should be identical to the `.py` file name in `src/server`.. 
- `--seed`: Name of `.yaml` config file (w/o the `.yaml` extension) in the `config/` directory.
### Example:
```py
python main.py --config-name cifar10_dir method=floco
```
## Parallel Training via `Ray`
This feature can **vastly improve your training efficiency**. At the same time, this feature is user-friendly and easy to use!!!
### Activate (What You ONLY Need To Do)
```yaml
# your_config.yaml
mode: parallel
parallel:
  num_workers: 2 # any positive integer that larger than 1
  ...
```
## Monitor runs
This implementation supports `tensorboard`.
1. Run `tensorboard --logdir=<your_log_dir>` on terminal.
2. Go check `localhost:6006` on your browser.

# Bibtex

```bibtex
@inproceedings{grinwald2024floco,
  title={Federated Learning over Connected Modes},
  author={Grinwald, Dennis and Wiesner, Philipp and Nakajima, Shinichi},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS'24)},
  year={2024},
  url={https://openreview.net/forum?id=JL2eMCfDW8}
}
```