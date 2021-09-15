import os
import time
import subprocess
from multiprocessing import Pool

def dummy_launcher(commands):
    """
    Doesn't run anything; instead, prints each command.
    Useful for testing.
    """
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')
        
def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)

        
def slurm_launcher(commands):
    """
    Parallel job launcher for computationnal cluster using SLURM workload manager.
    An example of SBATCH options:

        #!/bin/bash
        #SBATCH --job-name=<job_name>
        #SBATCH --output=<job_name>.out
        #SBATCH --error=<job_name>_error.out
        #SBATCH --ntasks=4
        #SBATCH --cpus-per-task=8
        #SBATCH --gres=gpu:4
        #SBATCH --time=1-00:00:00
        #SBATCH --mem=81Gb

    Note: --cpus-per-task should match the N_WORKERS defined in datasets.py (default 8)
    Note: there should be equal number of --ntasks and --gres
    """

    with Pool(processes=int(os.environ["SLURM_NTASKS"])) as pool:

        processes = []
        for command in commands:
            process = pool.apply_async(
                subprocess.run, 
                [f'srun --ntasks=1 --cpus-per-task={os.environ["SLURM_CPUS_PER_TASK"]} --mem=20G --gres=gpu:1 --exclusive {command}'], 
                {"shell": True}
                )
            processes.append(process)
            time.sleep(10)

        for i, process in enumerate(processes):
            process.wait()
            print("//////////////////////////////")
            print("//// Completed ", i , " / ", len(commands), "////")
            print("//////////////////////////////")


REGISTRY = {
    'dummy': dummy_launcher,
    'local': local_launcher,
    'slurm_launcher': slurm_launcher
}