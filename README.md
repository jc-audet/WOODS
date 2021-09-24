# Temporal_OOD
Repository aiming to benchmark OOD performance of temporal data.

## Quick Start

### Download

Some datasets require downloads before running any training scripts notably the PhysioNet dataset which requires a download (~50G) and preprocessing (~1h30)

```sh
python3 -m scripts.download \
        --data_path ./data
```

### Train a model

Train a single model using one objective on one dataset with one test environment

```sh
python3 -m main train \
        --dataset Spurious_Fourier \
        --objective ERM \
        --test_env 0
```

### Make a sweep

To launch a sweep, use the following script. Define a collection of Objective and Dataset to investigate a combination. All test environment are gonna be investigated automatically. By default, the sweep is gonna do a random search over 20 hyper parameter seeds with 3 trial seed each.

```sh
python3 -m scripts.hparams_sweep \
        --dataset Spurious_Fourier \
        --objective ERM IRM \
        --save_path ./results \
        --launcher local
```

To change the number of seeds investigated, you can call the `--n_hparams` and `--n_trials` argument.

```sh
python3 -m scripts.hparams_sweep \
        --dataset Spurious_Fourier TCMNIST_seq \
        --objective ERM \
        --save_path ./results \
        --launcher local \
        --n_hparams 10 \
        --n_trials 1
```

You can also specify which test environment you want to investigate using the `--unique_test_env` argument

```sh
python3 -m scripts.hparams_sweep \
        --dataset Spurious_Fourier TCMNIST_seq \
        --objective ERM IRM \
        --save_path ./results \
        --launcher local \
        --unique_test_env 0
```

If you want to sweep a dataset with the 'step' environment setup, you must give the test_step you want the dataset to have

```sh
python3 -m scripts.hparams_sweep \
        --dataset TCMNIST_step \
        --test_step 2\
        --objective ERM IRM \
        --save_path ./results \
        --launcher local
```

When the sweep is complete you can compile the results in neat tables, the '--latex' argument outputs a table that can be directly copy pasted into a .tex documents (with the **** package)

```sh
python3 -m scripts.compile_results \
        --results_dir ./results \
        --latex
```

### Evaluate a model

Evaluate a single model using one objective on one dataset with one test environment

```sh
python3 -m main eval \
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
