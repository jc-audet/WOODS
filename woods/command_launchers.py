"""Set of functions used to launch lists of python scripts

TODO:
    Check Joblib library for parallel compute?"""
import os
import time
import subprocess
from multiprocessing import Pool

def dummy_launcher(commands):
    """Doesn't launch any scripts in commands, it only prints the commands. Useful for testing.

    Taken from : https://github.com/facebookresearch/DomainBed/

    Args:
        commands (List): List of list of string that consists of a python script call
    """
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')
        
def local_launcher(commands):
    """Launch all of the scripts in commands on the local machine serially. If GPU is available it is gonna use it.

    Taken from : https://github.com/facebookresearch/DomainBed/

    Args:
        commands (List): List of list of string that consists of a python script call
    """
    for cmd in commands:
        subprocess.call(cmd, shell=True)

        
def slurm_launcher(commands):
    """Parallel job launcher for computationnal cluster using the SLURM workload manager. 

    Launches all the jobs in commands in parallel according to the number of tasks in the slurm allocation.
    An example of SBATCH options::

            #!/bin/bash
            #SBATCH --job-name=<job_name>
            #SBATCH --output=<job_name>.out
            #SBATCH --error=<job_name>_error.out
            #SBATCH --ntasks=4
            #SBATCH --cpus-per-task=8
            #SBATCH --gres=gpu:4
            #SBATCH --time=1-00:00:00
            #SBATCH --mem=81Gb

    Note: 
        --cpus-per-task should match the N_WORKERS defined in datasets.py (default 4)
    Note: 
        there should be equal number of --ntasks and --gres

    Args:
        commands (List): List of list of string that consists of a python script call
    """
    mem_per_run = int(float(os.environ['SLURM_MEM_PER_NODE']) // int(os.environ["SLURM_NTASKS"]) // 1000)
    with Pool(processes=int(os.environ["SLURM_NTASKS"])) as pool:

        processes = []
        for command in commands:
            process = pool.apply_async(
                subprocess.run, 
                [f'srun --ntasks=1 --cpus-per-task={os.environ["SLURM_CPUS_PER_TASK"]} --mem={mem_per_run}G --gres=gpu:1 --exclusive {command}'], 
                {"shell": True}
                )
            processes.append(process)
            time.sleep(10)

        for i, process in enumerate(processes):
            process.wait()
            print("//////////////////////////////")
            print("//// Completed ", i+1 , " / ", len(commands), "////")
            print("//////////////////////////////")


REGISTRY = {
    'dummy': dummy_launcher,
    'local': local_launcher,
    'slurm_launcher': slurm_launcher
}
