#!/bin/bash
#SBATCH --job-name=tvsum-download
#SBATCH --account=tesis
#SBATCH --partition=standard
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=jobs/logs/download_%j.out
#SBATCH --error=jobs/logs/download_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sebastian.quispe@khipu.utec.edu.pe

# TVSum Dataset Download Job
# Downloads the TVSum dataset from MIT CSAIL

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Started at: $(date)"
echo "=========================================="

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p data/raw/tvsum data/raw/videos jobs/logs

# Download TVSum dataset
echo "Downloading TVSum dataset..."
wget -c http://people.csail.mit.edu/yalesong/tvsum/tvsum50_ver_1_1.tgz -O /tmp/tvsum50_ver_1_1.tgz

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to download TVSum dataset"
    echo "Try downloading manually from: http://people.csail.mit.edu/yalesong/tvsum/"
    exit 1
fi

echo "Extracting TVSum dataset..."
tar -xzf /tmp/tvsum50_ver_1_1.tgz -C data/raw/tvsum/

# The zip usually contains a tvsum50/ directory
echo "Dataset extracted. Contents:"
ls -la data/raw/tvsum/

# The tar extracts to tvsum50_ver_1_1/ directory
echo "Moving videos to unified location..."
if [ -d "data/raw/tvsum/tvsum50_ver_1_1/video" ]; then
    mv data/raw/tvsum/tvsum50_ver_1_1/video/* data/raw/videos/ 2>/dev/null
elif [ -d "data/raw/tvsum/tvsum50_ver_1_1/videos" ]; then
    mv data/raw/tvsum/tvsum50_ver_1_1/videos/* data/raw/videos/ 2>/dev/null
fi

# Also copy annotations if they exist
if [ -f "data/raw/tvsum/tvsum50_ver_1_1/ydata-tvsum50.mat" ]; then
    cp data/raw/tvsum/tvsum50_ver_1_1/ydata-tvsum50.mat data/raw/tvsum/annotations/
    echo "Annotations copied to data/raw/tvsum/annotations/"
fi

echo "=========================================="
echo "Finished at: $(date)"
echo "Videos in data/raw/videos/:"
ls data/raw/videos/ | head -20
echo "=========================================="
