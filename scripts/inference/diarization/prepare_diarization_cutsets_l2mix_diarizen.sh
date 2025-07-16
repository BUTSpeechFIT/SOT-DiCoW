#!/bin/bash
#$ -N CHIME_DIAR
#$ -cwd
#$ -v SRC_ROOT
#$ -q long.q
#$ -l gpu=1,gpu_ram=20G
#$ -o /mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge/log/$JOB_NAME_$JOB_ID.out
#$ -e /mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge/log/$JOB_NAME_$JOB_ID.err

# As Karel said don't be an idiot and use the same number of GPUs as requested
export N_GPUS=1
export $(/mnt/matylda4/kesiraju/bin/gpus $N_GPUS) || exit 1

SRC_ROOT=/mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge
cd $SRC_ROOT

# Enable opening multiple files
ulimit -n 4096

# Enable to save bigger files
ulimit -f unlimited

# Enable more threads per process by increasing virtual memory (https://stackoverflow.com/questions/344203/maximum-number-of-threads-per-process-in-linux)
ulimit -v unlimited

EXPERIMENT="diarizen_diar_v2"
ORIG_CUTSET_NAME="libri2mix_mix_both_sc_test_cutset"
source "${SRC_ROOT}"/configs/local_paths.sh


ORIG_CUTSET="${MANIFEST_DIR}/${ORIG_CUTSET_NAME}.jsonl.gz"
NEW_CUTSET="${MANIFEST_DIR_EXP}/${ORIG_CUTSET_NAME}_${EXPERIMENT}.jsonl.gz"

$SRC_ROOT/sge_tools/interactive_python_diarizen $SRC_ROOT/utils/diarizen_diar.py \
--model=BUT-FIT/diarizen-wavlm-large-s80-md \
--input_cutset=$ORIG_CUTSET --output_dir=$EXPERIMENT_PATH

$SRC_ROOT/sge_tools/interactive_python_diarizen $SRC_ROOT/utils/prepare_diar_cutset_from_rttm_dir.py \
--lhotse_manifest_path=$ORIG_CUTSET --rttm_dir=$EXPERIMENT_PATH --out_manifest_path=$NEW_CUTSET

$SRC_ROOT/sge_tools/interactive_python_diarizen $SRC_ROOT/utils/compute_der_between_cutsets.py \
--ref_cutset=$ORIG_CUTSET --hyp_cutset=$NEW_CUTSET