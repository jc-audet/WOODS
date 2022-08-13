""" Set of functions used to launch lists of python scripts """
import os
import time
import math
import subprocess
import torch
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

def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.

    Taken from : https://github.com/facebookresearch/DomainBed/

    Args:
        commands (List): List of list of string that consists of a python script call
    """

    n_gpus = torch.cuda.device_count()
    gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
    procs_by_gpu = [None]*len(gpus)

    while len(commands) > 0:
        for gpu_idx, gpu in enumerate(gpus):
            proc = procs_by_gpu[gpu_idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu} {cmd}', shell=True)
                procs_by_gpu[gpu_idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()

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
    mem_per_run = 10*math.floor(mem_per_run/10)
    with Pool(processes=int(os.environ["SLURM_NTASKS"])) as pool:

        processes = []
        for command in commands:
            process = pool.apply_async(
                subprocess.run, 
                [f'srun --ntasks=1 --cpus-per-task={os.environ["SLURM_CPUS_PER_TASK"]} --mem={mem_per_run}G --gres=gpu:1 --exclusive {command}'], 
                {"shell": True}
                )
            processes.append(process)
            time.sleep(1)

        for i, process in enumerate(processes):
            process.wait()
            print("//////////////////////////////")
            print("//// Completed ", i+1 , " / ", len(commands), "////")
            print("//////////////////////////////")


def multi_node_slurm_launcher(commands):
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
    n_nodes = os.environ['SLURM_NNODES']
    nodes = os.environ['SLURM_NODELIST']
    print(nodes)
    with Pool(processes=int(os.environ["SLURM_NTASKS_PER_NODE"])*int(n_nodes)) as pool:

        processes = []
        for i, command in enumerate(commands):
            node = nodes[i%n_nodes]
            process = pool.apply_async(
                subprocess.run, 
                [f'srun --nodelist=nodes --ntasks=1 --cpus-per-task={os.environ["SLURM_CPUS_PER_TASK"]} --gpus-per-task=1 --exclusive {command}'], 
                {"shell": True}
                )
            processes.append(process)
            time.sleep(1)

        for i, process in enumerate(processes):
            process.wait()
            print("//////////////////////////////")
            print("//// Completed ", i+1 , " / ", len(commands), "////")
            print("//////////////////////////////")

REGISTRY = {
    'dummy': dummy_launcher,
    'local': local_launcher,
    'slurm_launcher': slurm_launcher,
    'multi_node_slurm_launcher': multi_node_slurm_launcher,
    'multi_gpu': multi_gpu_launcher,
}
