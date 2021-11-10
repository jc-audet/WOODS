
# Running a  Sweep

In WOODS, we evaluate the performance of a domain generalization algorithm by running a sweep over the hyper parameters definition space and then performing model selection on the training runs conducted during the sweep.

## Downloading the data
Before running any training run, we need to make sure we have the data to train on. 
### Direct Preprocessed Download
The repository offers direct download to the preprocessed data which is the quickest and most efficient way to get started. To download the preprocessed data, run the download module of the woods.scripts package and specify the dataset you want to download:
```sh
python3 -m woods.scripts.download DATASET\
        --data_path ./path/to/data/directory
```
### Source Download and Preprocess
For the sake of transparency, WOODS also offers the preprocessing scripts we took for all datasets in the preprecessing module of the woods.scripts package. You can also use the same module to download the raw data from the original source and run preprocessing yourself on it. DISCLAIMER: Some of the datasets take a long time to preprocess, especially the EEG datasets.
```sh
python3 -m woods.scripts.preprocess DATASET\
        --data_path ./path/to/data/directory
```
### Datasets Info
The following table lists the available datasets and their corresponding raw and preprocessed sizes.

|      Datasets     | Modality  | Requires Download | Preprocessed Size | Raw Size |
|-------------------|-----------|--------------------|-------------------|-------------------|
| Basic_Fourier | 1D Signal | No | - | - | - |
| Spurious_Fourier | 1D Signal | No | - | - | - |
| TMNIST | Video | Yes, but done automatically | 0.11 GB | - |
| TCMNIST_seq | Video | Yes, but done automatically | 0.11 GB | - |
| TCMNSIT_step | Video | Yes, but done automatically | 0.11 GB | - |
| CAP_DB | EEG | Yes | 9.1 GB | 40.1 GB |
| SEDFx_DB | EEG | Yes | 10.7 GB | 8.1 GB |
| LSA64 | Video | Yes | 0.26 GB | 1.5 GB |
| HAR | Sensor | Yes | 0.16 GB | 3.1 GB |

## Running the sweep
Once we have the data, we can start running the sweep. The hparams_sweep module of the woods.scripts package provides the command line interface to create the list of jobs to run, which is then passed to the command launcher to launch all jobs. The list of jobs includes all of the necessary training runs to get the results from all trial seeds, and hyper parameter seeds for a given algorithm, dataset and test environment.

In other words, for every combination of (algorithm, dataset, test environment) we train 20 different hyper parameter configurations on which we investigate 3 different trial seeds. This means that for every combination of (algorithm, dataset, test environment) we run 20 * 3 = 60 training runs.
```sh
python3 -m woods.scripts.hparams_sweep \
        --dataset Spurious_Fourier TCMNIST_seq \
        --objective ERM IRM \
        --save_path ./results \
        --launcher local
```
Here we are using the local launcher to run the jobs locally, which is the simplest launcher. We also offer other lauchers in the command_launcher module, such as slurm_launcher which is a parallel job launcher for the SLURM workload manager.

## Compiling the results
Once the sweep is finished, we can compile the results. The compile_results module of the woods.scripts package provides the command line interface to compile the results. The --latex option is used to generate the latex table.
```sh
python3 -m woods.scripts.compile_results \
        --results_dir ./results \
        --latex
```

## Advanced usage
If 60 jobs is too many jobs for you available compute, or too few for you experiments you can change the number of seeds investigated, you can call the `--n_hparams` and `--n_trials` argument.
```sh
python3 -m woods.scripts.hparams_sweep \
        --dataset Spurious_Fourier TCMNIST_seq \
        --objective ERM IRM \
        --save_path ./results \
        --launcher local \
        --n_hparams 10 \
        --n_trials 1
```
If some of the test environment of a dataset is not of interest to you, you can specify which test environment you want to investigate using the `--unique_test_env` argument
```sh
python3 -m woods.scripts.hparams_sweep \
        --dataset Spurious_Fourier TCMNIST_seq \
        --objective ERM IRM \
        --save_path ./results \
        --launcher local \
        --unique_test_env 0
```