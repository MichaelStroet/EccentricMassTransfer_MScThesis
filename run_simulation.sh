#!/bin/bash

#SBATCH --partition=astera       # Request nodes from this partition
#SBATCH --nodes 1                # How many nodes to ask for
#SBATCH --nodelist helios-cn042  # Specific node to use
#SBATCH --ntasks 45              # Number of tasks (MPI processes) (per astera node, 126 is maximum)
#SBATCH --cpus-per-task 1        # Number of logical CPUS (threads) per task
#SBATCH --time 10-0:00:00        # How long you need to run for in days-hours:minutes:seconds
#SBATCH --mem 32gb               # How much memory you need per node
#SBATCH -J EMT                   # The job name. If not set, slurm uses the name of your script
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err


function clean_up {
     echo "### Running Clean_up ###"

     echo "Whac-A-Mole'ing remaining gadget2 and huayno workers" # Otherwise it will get stuck copying the files and not remove the tmp dir
     # THIS ALSO KILLS ALL OTHER SIMULATIONS CURRENTLY RUNNING ON THE SAME NODE!!!
     # Need to somehow specify the workers that belong to this job
     # and/or not run this if amuse didnt crash or got aborted (and thus ran <code>.stop())
     killall -vw -u $USER gadget2_worker_normal
     killall -vw -u $USER huayno_worker_normal

     # Define workdir as last-updated directory (there should be only one anyway)
     work_dir=$(basename $(ls -td $OUTPUT_FOLDER/simulations/* | head -n 1))

     # copy the data off of /hddstore and onto filer0 (home directory)
     echo "Copying files..."
     rsync -av $OUTPUT_FOLDER/simulations	/home/$USER/

     # remove the folder from /hddstore
     echo "Removing files..."
     rm -rf $OUTPUT_FOLDER

     echo "Finished"
     date

     # Move the log files to the work dir and the "logs/finished" directory
     echo "Moving log files..."
     cp /home/$USER/logs/$SLURM_JOB_ID.out	/home/$USER/logs/finished/$SLURM_JOB_ID.out
     mv /home/$USER/logs/$SLURM_JOB_ID.out	/home/$USER/simulations/$work_dir/data/$SLURM_JOB_ID.out

     cp /home/$USER/logs/$SLURM_JOB_ID.err	/home/$USER/logs/finished/$SLURM_JOB_ID.err
     mv /home/$USER/logs/$SLURM_JOB_ID.err	/home/$USER/simulations/$work_dir/data/$SLURM_JOB_ID.err

     exit
}

# call "clean_up" function when this script exits, it is run even if SLURM cancels the job
trap 'clean_up' EXIT

##########################################################################################

##### Activate AMUSE environment #####

# Load the same models used for installing AMUSE
module purge
module load gnu12/12.3.0
module load gsl/2.7.1
module load openmpi4/4.1.6


# Activate AMUSE python environment
export PYTHON_ENV="Amuse-env"
echo python envionment: $PYTHON_ENV

source /home/$USER/$PYTHON_ENV/bin/activate

# Allow AMUSE to use more workers per core
export OMPI_MCA_rmaps_base_oversubscribe=1

##### pipeline below #####

# Make a folder in /hddstore/uvanetid/tmp.XXXX by using mktemp
# You can have multiple scripts at once and each writes its output to a different folder.
# By echoing both the folder name and the slurm node name, the slurm output file will
# tell you where to find your data if something goes wrong and your program crashes.

mkdir -p /hddstore/$USER
export OUTPUT_FOLDER=$(mktemp -d -p /hddstore/$USER)
mkdir -p $OUTPUT_FOLDER/simulations

echo output folder: $OUTPUT_FOLDER
echo node: $SLURMD_NODENAME

# If you need input data, copy it to the compute node
# echo Copying files
# cp  -r /home/$USER/simulations/Mp1.5_Ms1.4_e0.5_a30_N250k_RL1.0  $OUTPUT_FOLDER/simulations

# Switch to the output folder before running the program
cd $OUTPUT_FOLDER

# Run your program
date
echo running main.py
python3.11 /home/$USER/code/main.py

