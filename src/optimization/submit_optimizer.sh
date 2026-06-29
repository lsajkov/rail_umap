#!/bin/bash
#SBATCH -J optimize_UMAP
#SBATCH -A m1727
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 128
#SBATCH -t 12:00:00
#SBATCH --array=0-11
#SBATCH -o logs/optimize_UMAP_%A_%a.out
#SBATCH -e logs/optimize_UMAP_%A_%a.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate rail

CONFIGS=(LSST         LSST      \
         LSSTRoman    LSSTRoman  \
         LSSTRomanHSC LSSTRomanHSC \
         HSC          HSC        \
         RomanHSC     RomanHSC   \
         Roman        Roman)

DATA_CUTS=(10000 100000 \
           10000 100000 \
           10000 100000 \
           10000 100000 \
           10000 100000 \
           10000 100000 )

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
DATA_CUT=${DATA_CUTS[$SLURM_ARRAY_TASK_ID]}

N_WORKERS=8
TRIALS_PER_WORKER=$((100 / N_WORKERS))

cd /global/homes/s/sajkov/rail_umap/src/optimization
mkdir -p logs

for i in $(seq 1 $N_WORKERS); do
    python UMAP_optimizer.py 26Jun26 $CONFIG $DATA_CUT $TRIALS_PER_WORKER &
done
wait