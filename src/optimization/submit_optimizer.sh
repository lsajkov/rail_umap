#!/bin/bash
#SBATCH -J optimize_UMAP
#SBATCH -A m1727
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -t 24:00:00
#SBATCH -o logs/optimize_UMAP_%j.out
#SBATCH -e logs/optimize_UMAP_%j.err

module load python
source activate rail

srun -n 1 python UMAP_optimizer.py 26Jun26 LSST 10000
srun -n 1 python UMAP_optimizer.py 26Jun26 LSST 100000
srun -n 1 python UMAP_optimizer.py 26Jun26 LSST 500000


srun -n 1 python UMAP_optimizer.py 26Jun26 LSSTRoman 10000
srun -n 1 python UMAP_optimizer.py 26Jun26 LSSTRoman 100000
srun -n 1 python UMAP_optimizer.py 26Jun26 LSSTRoman 500000

srun -n 1 python UMAP_optimizer.py 26Jun26 LSSTRomanHSC 10000
srun -n 1 python UMAP_optimizer.py 26Jun26 LSSTRomanHSC 100000
srun -n 1 python UMAP_optimizer.py 26Jun26 LSSTRomanHSC 500000

srun -n 1 python UMAP_optimizer.py 26Jun26 LSST 10000
srun -n 1 python UMAP_optimizer.py 26Jun26 LSST 100000
srun -n 1 python UMAP_optimizer.py 26Jun26 LSST 500000

srun -n 1 python UMAP_optimizer.py 26Jun26 LSST 10000
srun -n 1 python UMAP_optimizer.py 26Jun26 LSST 100000
srun -n 1 python UMAP_optimizer.py 26Jun26 LSST 500000

srun -n 1 python UMAP_optimizer.py 26Jun26 LSST 10000
srun -n 1 python UMAP_optimizer.py 26Jun26 LSST 100000
srun -n 1 python UMAP_optimizer.py 26Jun26 LSST 500000

srun -n 1 python UMAP_optimizer.py 26Jun26 LSST 10000
srun -n 1 python UMAP_optimizer.py 26Jun26 LSST 100000
srun -n 1 python UMAP_optimizer.py 26Jun26 LSST 500000