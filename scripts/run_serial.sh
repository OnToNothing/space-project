#!/usr/bin/bash
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G
#SBATCH -t 12:00:00
#SBATCH -A mor100
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --export=ALL

# Serial programs run on shared nodes using a few cores to run individual processes in parallel

SCRATCH="/scratch/$USER/job_$SLURM_JOB_ID"

for FILE in random100.npy random1000.npy random10000.npy; do
    echo $FILE
    cp "$HOME/nbody-examples/$FILE" "$SCRATCH"
    SFILE="$SCRATCH/$FILE"
    echo nbody-s
    for i in `seq 3`
    do
        ./nbody-s 0.01 10 1000 "$SFILE" "$SCRATCH/temp-$i.npy" &  # & means to background the process
    done
    wait # waits for all background processes
    echo

    echo nbody-s3
    for i in `seq 3`
    do
        ./nbody-s3 0.01 10 1000 "$SFILE" "$SCRATCH/temp-$i.npy" &
    done
    wait
    echo

    echo
done
