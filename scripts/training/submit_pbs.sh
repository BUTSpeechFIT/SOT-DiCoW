#!/bin/bash
#PBS -N dicow
#PBS -v CFG
#PBS -q gpu
#PBS -l select=1:ncpus=5:mem=120gb:ngpus=1:gpu_mem=40gb:mpiprocs=1:scratch_local=10gb
#PBS -l walltime=22:00:00
#PBS -o dicow.log
#PBS -e dicow.log
#PBS -m abe -M ikocour@fit.vutbr.cz

# e.g. submit_pbs.sh "+decode=mt_asr/mt_nsf ++training.per_device_eval_batch_size=1"
# or
# CFG="+decode=mt_asr/mt_nsf ++training.per_device_eval_batch_size=1" qsub submit_pbs.sh 
set -x
CFG_ARG="$1"
[ -z "$1" ] && CFG_ARG="$CFG"

(
    echo "JobID:   $PBS_JOBID"
    echo "JobHost: $HOSTNAME"
    echo "CFG:     $CFG_ARG"
    echo "PWD:     $PWD"
    set -ux
    export N_GPUS=`nvidia-smi --list-gpus | wc -l` # this work on PBS, bcz of virtualization layer
    # export CUDA_VISIBLE_DEVICES="0"
    cd $PBS_O_HOME

    $PBS_O_HOME/MT-ASR/CHIME2024/scripts/training/run_train.sh "$CFG_ARG"
) >$PBS_O_HOME/MT-ASR/CHIME2024/exp/logs/$PBS_JOBNAME-$PBS_JOBID.log 2>&1
