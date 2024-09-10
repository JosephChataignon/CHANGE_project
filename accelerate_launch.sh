#!/bin/bash
#SBATCH --mail-user=joseph.chataignon@unibe.ch
#SBATCH --mail-type=fail,end
#SBATCH -o /storage/homefs/jc23c442/logs/slurm_jobid-%j.log # output reports directory
#SBATCH --partition=gpu
#SBATCH --qos=job_gpu
#SBATCH --gres=gpu:teslap100:4
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00

echo "date: $(date)"

echo "loading workspace"
export HPC_WORKSPACE=wbkolleg_dh_nlp
module load Workspace
echo "loading CUDA drivers"
module load CUDA


# Access the first argument passed to the script
python_script=$1

# Check if argument is empty
if [ -z "$python_script" ]; then
    echo "Please provide an argument"
    echo "Usage: pysbatch <python_script>"
    exit 1
fi

fullscriptpath=$(realpath $python_script)
echo "script path provided: $python_script"

if [[ $fullscriptpath == *"/storage/homefs/jc23c442/CHANGE_project"* ]]; then
    echo "script from the CHANGE project, I will do a git commit"
    cd /storage/homefs/jc23c442/CHANGE_project
    git add .
    git commit -m "automatic commit from pysbatch execution"
    git pull
    git push
fi
echo "The code of the latest commit is available at:"
echo "https://github.com/JosephChataignon/CHANGE_project/commit/$(git rev-parse HEAD)"

echo " === "
echo "Executing Python script: $fullscriptpath , in Apptainer container: ubuntu_env.sif"
apptainer exec --nv \
    --bind /storage/research/wbkolleg_dh_1:/research_storage \
    --bind /software.9:/software.9 \
    ~/ubuntu_env.sif \
    accelerate launch --config_file accelerate_config_4gpu.yaml "$fullscriptpath"
