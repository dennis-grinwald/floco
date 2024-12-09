#!/bin/bash
#SBATCH --job-name=fold
#SBATCH --partition=cpu-2h
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32G
#SBATCH --output=logs/job-%j.out

cp /home/space/datasets-sqfs/FLIIDNIID/Data.sqfs /tmp/

apptainer run -B /tmp/Data.sqfs:/cluster:image-src=/ --nv fl_bench.sif python generate_data.py -d ${1} -f ${2} -cn ${3} --val_ratio ${4} --test_ratio ${5} --seed ${6}