#!/bin/bash
#PBS -N dicow
#PBS -v CFG
#PBS -q gpu_dgx
#PBS -l select=1:ncpus=40:mem=600gb:ngpus=4:mpiprocs=1:scratch_local=400gb
#PBS -l walltime=47:00:00
#PBS -o dicow_dgx.log
#PBS -e dicow_dgx.log
#PBS -m abe -M ikocour@fit.vutbr.cz

# e.g. submit_pbs_dgx.sh "+decode=mt_asr/mt_nsf ++training.per_device_eval_batch_size=1"
# or
# CFG="+decode=mt_asr/mt_nsf ++training.per_device_eval_batch_size=1" qsub submit_pbs_dgx.sh 
set -x
CFG_ARG="$1"
[ -z "$1" ] && CFG_ARG="$CFG"

trap 'clean_scratch' EXIT TERM

(
    echo "JobID:   $PBS_JOBID"
    echo "JobHost: $HOSTNAME"
    echo "CFG:     $CFG_ARG"
    echo "PWD:     $PWD"
    set -ux
    export N_GPUS=`nvidia-smi --list-gpus | wc -l` # this work on PBS, bcz of virtualization layer
    # export CUDA_VISIBLE_DEVICES="0,1,2,3"

    $PBS_O_HOME/MT-ASR/CHIME2024/scripts/training/run_train.sh "$CFG_ARG"
) >$PBS_O_HOME/MT-ASR/CHIME2024/exp/logs/$PBS_JOBNAME-$PBS_JOBID.log 2>&1
