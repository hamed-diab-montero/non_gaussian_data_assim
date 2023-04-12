#!/bin/sh
#
#SBATCH --job-name="pff_chaos_l96"
#SBATCH --partition=compute
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --account=research-ceg-gse
#SBATCH --mail-user=h.a.diabmontero@tudelft.nl
#SBATCH --mail-type=ALL

echo 'Start of the particle filter experiment'

python3 test_lorenz_96_pff_chaos_v1.py

echo 'Finished particle filter experiment'


