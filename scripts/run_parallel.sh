#!/usr/bin/bash
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00
#SBATCH -A mor100
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --export=ALL

# Parallel programs run on a dedicated node using all (or at least half) of the cores

NUM_THREADS=128  # 64 may be a good choice as well

SCRATCH="/scratch/$USER/job_$SLURM_JOB_ID"

for FILE in random100.npy random1000.npy random10000.npy; do
    echo nbody-p
    for i in `seq 3`
    do
        ./nbody-p 0.01 10 1000 "$SFILE" "$SCRATCH/temp-$i.npy" $NUM_THREADS
    done
    echo

    echo nbody-p3
    for i in `seq 3`
    do
        ./nbody-p3 0.01 10 1000 "$SFILE" "$SCRATCH/temp-$i.npy" $NUM_THREADS
    done
    echo

    echo
done
