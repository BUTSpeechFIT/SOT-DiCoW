#!/bin/bash
set -x
if [ $# -lt 1 ]; then
    echo "Usage: $0  <dest_dir>"
    exit 1
fi

DATA=/auto/brno12-cerit/nfs4/home/ikocour/data/tmp
DIR=$1

mkdir -p $DIR/tmp

# move NSF dataset
for f in $DATA/nsf_part*.tar.gz
do
    {
        cp -u $f $DIR/tmp
        tar -xzf $DIR/tmp/$(basename $f) -C $DIR
        rm $DIR/tmp/$(basename $f)
    } &
    sleep 1
done
wait

# move libri2mix
for f in $DATA/librimix_part*.tar.gz
do
    {
        cp -u $f $DIR/tmp
        tar -xzf $DIR/tmp/$(basename $f) -C $DIR
        rm $DIR/tmp/$(basename $f)
    } &
    sleep 1
done
wait


# modify manifests
SCRIPTS=$(dirname $0)
MANIFEST_DIR=/auto/brno12-cerit/nfs4/home/ikocour/MT-ASR/CHIME2024/manifests
NEW_MANIFEST_DIR=$DIR/manifests
mkdir -p $NEW_MANIFEST_DIR

for f in notsofar_dev_sc_cutset.jsonl.gz notsofar_eval_sc_cutset.jsonl.gz notsofar_dev_sc_cutset_30s.jsonl.gz notsofar_train_sc_cutset_30s.jsonl.gz libri2mix_both_100_train_sc_cutset_30s.jsonl.gz libri2mix_both_360_train_sc_cutset_30s.jsonl.gz libri2mix_clean_100_train_sc_cutset_30s.jsonl.gz libri2mix_clean_360_train_sc_cutset_30s.jsonl.gz libri2mix_mix_clean_sc_dev_cutset.jsonl.gz libri2mix_mix_clean_sc_test_cutset.jsonl.gz
do
    $SCRIPTS/remap_audio_path.py $MANIFEST_DIR/$f /auto/brno12-cerit/nfs4/home/ikocour/data $DIR $NEW_MANIFEST_DIR/$f
done
