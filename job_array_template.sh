#!/bin/bash
#SBATCH --array=1-n_of_jobs
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32000M
#SBATCH --time=08:00:00
#SBATCH --job-name=sweep_
#SBATCH --wait

# Copy data
# cp -r ~/prepared_data/ $SLURM_TMPDIR/

# activate the enviroment
module load anaconda/3
conda activate tood

# replace dummy file location with $SLURM_TMPDIR
#sed -i "s/dummy/$SLURM_TMPDIR/g" commands.txt 

# #SLURM_ARRAY_TASK_ID=11
command=`cat commands.txt | sed "${SLURM_ARRAY_TASK_ID}!d"`
# dir=$SLURM_TMPDIR/prepared_data/
# command="${command/dummy/$dir}"

echo ${SLURM_ARRAY_TASK_ID}
echo $command
eval $command 
