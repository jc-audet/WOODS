
# Running a  Sweep

In WOODS, we evaluate the performance of a domain generalization algorithm by running a sweep over the hyper parameters definition space and then performing model selection on the training runs conducted during the sweep.

## Running the sweep
Once we have the data, we can start running the sweep. The hparams_sweep module of the woods.scripts package provides the command line interface to create the list of jobs to run, which is then passed to the command launcher to launch all jobs. The list of jobs includes all of the necessary training runs to get the results from all trial seeds, and hyper parameter seeds for a given algorithm, dataset and test domain.

All datasets have the `SWEEP_ENVS` attributes that defines which test environments are included in the sweep. For example, the `SWEEP_ENVS` attribute for the `Spurious Fourier` dataset is only 1 test domain while for most real datasets `SWEEP_ENVS` consists of all domains.

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
        --results_dir path/to/results \
        --latex
```

It is also possible to compile the results from multiple directories containing complementary sweeps results. This will put all of those results in the same table.
```sh
python3 -m woods.scripts.compile_results \
        --results_dir path/to/results/1 path/to/results/2 path/to/results/3 \
        --latex
```
There are other mode of operation for the compile_results module, such as `--mode IID` which takes results from a sweep with no test environment and report the results for each test environment separately.  
```sh
python3 -m woods.scripts.compile_results \
        --results_dir path/to/results/1 path/to/results/2 path/to/results/3 \
        --mode IID
```
There is also `--mode summary` which reports the average results for every dataset of all objectives in the sweep.
```sh
python3 -m woods.scripts.compile_results \
        --results_dir path/to/results/1 path/to/results/2 path/to/results/3 \
        --mode summary
```
You can also use the `--mode hparams` which reports the hparams of the model chosen by model selection
```sh
python3 -m woods.scripts.compile_results \
        --results_dir path/to/results/1 path/to/results/2 path/to/results/3 \
        --mode hparams
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
You can run a sweep with no test environment by specifying the `--unique_test_env` argument as `None`.
```sh
python3 -m woods.scripts.hparams_sweep \
        --dataset Spurious_Fourier TCMNIST_seq \
        --objective ERM IRM \
        --save_path ./results \
        --launcher local \
        --unique_test_env None
```