<p align='center'>
  <img width='100%' src='./assets/banner.png' />
</p>

--------------------------------------------------------------------------------
[![Documentation Status](https://readthedocs.org/projects/mitiq/badge/?version=stable)](https://woods.readthedocs.io/en/latest)

A set of Out-of-Distribution Generalization Benchmarks for Sequential Prediction Tasks

# Quick Start

## Download
Some datasets require downloads in order to be used. We provide the download script to directly download our own preprocessed versions of the datasets which are ready for use. If you want to look into the preprocessing yourself, check the `fetch_and_preprocess.py` script. You can provide a specific dataset to download, or provide nothing to download all of them. See the raw and preprocessed sizes of the dataset on the dataset page of the [documentation](https://woods.readthedocs.io/en/latest/downloading_datasets.html#datasets-info).

```sh
python3 -m woods.scripts.download {dataset} \
        --data_path /path/to/data/directory
```

## Train a model
Train a model using one objective on one dataset with one test environment. For data that requires a download, you need to provide the path to the data directory with `--data_path`.

```sh
python3 -m woods.scripts.main train \
        --dataset Spurious_Fourier \
        --objective ERM \
        --test_env 0 \
        --data_path /path/to/data/directory
```

## Make a sweep
To launch a hyper parameter sweep, use the following script. Define a collection of Objective and Dataset to investigate a combination. All test environment are gonna be investigated automatically. By default, the sweep is gonna do a random search over 20 hyper parameter seeds with 3 trial seed each.

```sh
python3 -m woods.scripts.hparams_sweep \
        --dataset Spurious_Fourier TCMNIST_seq\
        --objective ERM IRM \
        --save_path ./results \
        --launcher local
```

To change the number of seeds investigated, you can call the `--n_hparams` and `--n_trials` argument.

```sh
python3 -m woods.scripts.hparams_sweep \
        --dataset Spurious_Fourier TCMNIST_seq \
        --objective ERM \
        --save_path ./results \
        --launcher local \
        --n_hparams 10 \
        --n_trials 1
```

You can also specify which test environment you want to investigate using the `--unique_test_env` argument

```sh
python3 -m woods.scripts.hparams_sweep \
        --dataset Spurious_Fourier TCMNIST_seq \
        --objective ERM IRM \
        --save_path ./results \
        --launcher local \
        --unique_test_env 0
```

When the sweep is complete you can compile the results in neat tables, the '--latex' argument outputs a table that can be directly copy pasted into a .tex documents (with the **** package)

```sh
python3 -m woods.scripts.compile_results \
        --results_dir ./results \
        --latex
```

## Evaluate a model

Evaluate a single model using one objective on one dataset with one test environment

```sh
python3 -m woods.main eval \
        --model_path /path/to/model\
        --dataset Spurious_Fourier \
        --objective ERM \
        --test_env 0
```

# Contributors
