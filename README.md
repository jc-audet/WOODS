<p align='center'>
  <img width='100%' src='./assets/banner.png' />
</p>

--------------------------------------------------------------------------------

A set of Out-of-Distribution Generalization Benchmarks for Sequential Prediction Tasks

# Quick Start

## Download
Some datasets require downloads in order to be used. We provide the download script to directly download our own preprocessed versions of the datasets which are ready for use. If you want to look into the preprocessing yourself, check the `fetch_and_preprocess.py` script. You can provide a specific dataset to download, or provide nothing to download all of them.

```sh
python3 -m woods.scripts.download {dataset} \
        --data_path /path/to/data/directory
```

## Train a model

Train a single model using one objective on one dataset with one test environment

```sh
python3 -m woods.scripts.main train \
        --dataset CAP \
        --objective ERM \
        --test_env 0 \
        --data_path /hdd/data
```

From a training run results, you can also visualiza the results with the following script

```sh
python3 -m woods.scripts.visualize_results \
        --result_path ./results/something.json
```

### Make a sweep

To launch a sweep, use the following script. Define a collection of Objective and Dataset to investigate a combination. All test environment are gonna be investigated automatically. By default, the sweep is gonna do a random search over 20 hyper parameter seeds with 3 trial seed each.

```sh
python3 -m woods.scripts.hparams_sweep \
        --dataset Spurious_Fourier \
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

### Evaluate a model

Evaluate a single model using one objective on one dataset with one test environment

```sh
python3 -m woods.main eval \
        --model_path something\
        --dataset Spurious_Fourier \
        --objective ERM \
        --test_env 0
```

## Current benchmarks
### Spurious Fourier
#### The task
The task is to classify Fourier spectrum from observed signals.
![CFourier](assets/clean_task.png)

#### Spurious Version
In the spurious version of the dataset, we add a spike in the Fourier spectrum that is correlated with the label.
![SFourier](assets/env_task.png)

### Temporal ColoredMNIST
#### Greyscale
We create a sequence of 4 digits (one digit per frame) and the task is to predict if the sum of the last digit and the current digit is an odd number.

![TCMNIST_grey](assets/TCMNIST_grey.png)

#### OOD setup
We add 25% label noise to the dataset and add colors to the images that is correlated with different degrees to the labels. This degree of correlation is dictated by the environment from which the sample is taken from.

### Definition of environments
Two definition of environments are of interest in this dataset.

#### Sequences as environments
First definition of interest is the setup where a single sequence is taken from an enviroment and the level of correlation between the color and the label is constant across time-steps.

![TCMNIST-seq](assets/TCMNIST_seq.png)


#### Time steps as environments
The second definition of interest is the setup where every time step is an environment, so sequences of data have different level of correlation of color within the sequence

![TCMNIST-step](assets/TCMNIST_step.png)
