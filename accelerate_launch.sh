#!/bin/bash
#SBATCH --mail-user=joseph.chataignon@unibe.ch
#SBATCH --mail-type=fail,end
#SBATCH -o /storage/homefs/jc23c442/logs/slurm_jobid-%j.log # output reports directory
#SBATCH --partition=gpu
#SBATCH --qos=job_gpu
#SBATCH --gres=gpu:teslap100:2
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00

echo "date: $(date)"

# Function to detect the environment
detect_environment() {
    # Check for HPC-specific environment variables or paths
    if [[ -d /storage/homefs/jc23c442 ]] || [[ "$HOSTNAME" == "submit"* ]]; then
        echo "ubelix"
    # Check for lab server-specific environment variables or paths
    elif [[ "$HOSTNAME" == "srv" && "$USER" == "joseph" ]]; then
        echo "dhserver"
    else
        echo "unknown"
    fi
}
ENVIRONMENT=$(detect_environment)

# Execute commands based on the detected environment
case $ENVIRONMENT in
    ubelix)
        CHANGE_PROJ_DIR="/storage/homefs/jc23c442/CHANGE_project"
        STORAGE_DIR="/storage/research/wbkolleg_dh_1"
        SOFTWARE_BIND="--bind /software.9:/software.9"
        ACCELERATE_CONFIG="accelerate_config_HPC.yaml"
        echo "loading workspace"
        export HPC_WORKSPACE=wbkolleg_dh_nlp
        module load Workspace
        echo "loading CUDA drivers"
        module load CUDA
        ;;
    dhserver)
        CHANGE_PROJ_DIR="/home/joseph/CHANGE_project"
        STORAGE_DIR="/mnt/wbkolleg_dh_1"
        SOFTWARE_BIND="--bind /usr/local/cuda-12.1:/usr/local/cuda"
        ACCELERATE_CONFIG="accelerate_config_DHserver.yaml"
        ;;
    *)
        echo "Accelerate_launch: unknown environment"
        exit 1
        ;;
esac



# Access the first argument passed to the script
python_script=$1

# Check if argument is empty
if [ -z "$python_script" ]; then
    echo "Please provide an argument"
    echo "Usage: accelerate_launch <python_script>"
    exit 1
fi

fullscriptpath=$(realpath $python_script)
echo "script path provided: $python_script"

if [[ $fullscriptpath == *"/CHANGE_project"* ]]; then
    echo "script from the CHANGE project, I will do a git commit"
    cd "$CHANGE_PROJ_DIR"
    git add .
    git commit -m "automatic commit from script execution"
    git pull
    git push
fi
echo "The code of the latest commit is available at:"
echo "https://github.com/JosephChataignon/CHANGE_project/commit/$(git rev-parse HEAD)"

echo " === "
echo "Executing Python script: $fullscriptpath , in Apptainer container: ubuntu_env.sif"
# embeddings_finetune works with SentenceTransformers which is incompatible with Accelerate
if [[ "$fullscriptpath" == *"/embeddings_finetune.py" ]]; then
    echo "Execution with Torchrun."
    apptainer exec --nv \
        $SOFTWARE_BIND \
        --bind "$STORAGE_DIR":"$STORAGE_DIR" \
        ~/ubuntu_env.sif \
        torchrun --nproc_per_node=2  "$fullscriptpath"
else
    echo "Execution with Accelerate."
    apptainer exec --nv \
        $SOFTWARE_BIND \
        --bind "$STORAGE_DIR":"$STORAGE_DIR" \
        ~/ubuntu_env.sif \
        accelerate launch --config_file "$CHANGE_PROJ_DIR/$ACCELERATE_CONFIG" "$fullscriptpath"
fi

