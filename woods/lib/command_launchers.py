import os
import time
import subprocess
from multiprocessing import Pool

def dummy_launcher(commands):
    """Doesn't launch any scripts in commands, it only prints the commands. Useful for testing.
    Taken from: https://github.com/facebookresearch/DomainBed/blob/9e864cc4057d1678765ab3ecb10ae37a4c75a840/domainbed/command_launchers.py#L18

    Args:
        commands (List): List of list of string that consists of a python script call
    """
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')
        
def local_launcher(commands):
    """Launch all of the scripts in commands on the local machine serially. If GPU is available it is gonna use it.
    Taken from: https://github.com/facebookresearch/DomainBed/blob/9e864cc4057d1678765ab3ecb10ae37a4c75a840/domainbed/command_launchers.py#L13

    Args:
        commands (List): List of list of string that consists of a python script call
    """
    for cmd in commands:
        subprocess.call(cmd, shell=True)

        
def slurm_launcher(commands):
    """Parallel job launcher for computationnal cluster using the SLURM workload manager. 
    Launches all the jobs in commands in parallel according to the number of tasks in the slurm allocation.

    Example:
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

       
def slurm_launcher_sbatch(commands):
    """Parallel job launcher for computational cluster using the SLURM workload manager. 
    Launches all the jobs in commands in parallel using job array.

    Args:
        commands (List): List of list of string that consists of a python script call
    """
    # Read in the file
    with open('job_array_template.sh', 'r') as file :
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('n_of_jobs', str(len(commands)))
    # filedata = filedata.replace('command_to_run', str(command))

    # Write the file out again
    with open('job_array.sh', 'w') as file:
        file.write(filedata)

    print(type(commands),len(commands))
    for cmd in commands:
        f=open('commands.txt','w')
        for ele in commands:
            f.write(ele+'\n')
        f.close()
        
    command = 'sbatch'+ ' ' + 'job_array.sh' 
    print(command)
    subprocess.call(command, shell=True)
           

REGISTRY = {
    'dummy': dummy_launcher,
    'local': local_launcher,
    'slurm_launcher': slurm_launcher,
    'slurm_launcher_sbatch': slurm_launcher_sbatch
}

if __name__ == '__main__':
    n_of_jobs = 100

    # Read in the file
    with open('job_array_template.sh', 'r') as file :
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('n_of_jobs', str(n_of_jobs))
    filedata = filedata.replace('command_to_run', str(command))

    # Write the file out again
    with open('job_array.sh', 'w') as file:
        file.write(filedata)