#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Usage: $0 <ORIG_MANIFEST_DIR> <NEW_MANIFEST_DIR> [ORIG_AUDIO_PREFIX] [NEW_AUDIO_PREFIX]"
    exit 1
fi

SCRIPTS=`dirname $0`

ORIG_DIR=$1
NEW_DIR=$2
ORIG_PREFIX=$3
NEW_PREFIX=$4

mkdir -p $NEW_DIR/tmp

IGNORE=(
    cuts_train_clean_full_audio.jsonl.gz cuts_train_clean_full.jsonl.gz cuts_train_clean_full_audio.jsonl.gz cuts_train_clean_full_sources.jsonl.gz
    cuts_train_clean_ov40.jsonl.gz cuts_train_clean_ov40_sources.jsonl.gz cuts_train_comb_full_sources.jsonl.gz cuts_train_comb_ov40_sources.jsonl.gz
    cuts_train_rvb_full.jsonl.gz cuts_train_rvb_full_sources.jsonl.gz cuts_train_rvb_ov40.jsonl.gz cuts_train_rvb_ov40_sources.jsonl.gz
)

for f in "$ORIG_DIR"/*.json*; do
    f=`basename $f`
    if [ -f $NEW_DIR/$f ]; then
        echo "$f already exist. Skipping..."
        continue
    fi

    matched=false
    for element in "${IGNORE[@]}"; do
        if [[ "$element" == $f ]]; then
            matched=true
            break
        fi
    done

    if [ $matched == 'true' ]; then
        echo "Ignoring $f..."
        continue
    fi

    echo "Validating $f"
    $SCRIPTS/validate_cutset.py $ORIG_DIR/$f $NEW_DIR/$f
    if [ $? != 0 ]; then
        echo "Found error. Trying to remove features in $f"
        $SCRIPTS/drop_features.py $ORIG_DIR/$f $NEW_DIR/tmp/$f
        if [ $? != 0 ]; then
            if [ -z "$NEW_PREFIX" ]; then
                echo "Please run the CLI with $0 <ORIG_MANIFEST_DIR> <NEW_MANIFEST_DIR> <ORIG_AUDIO_PREFIX> <NEW_AUDIO_PREFIX>"
                exit 1
            fi

            echo "Trying to remap audio paths in $f"
            $SCRIPTS/remap_audio_path.py $NEW_DIR/tmp/$f $ORIG_PREFIX $NEW_PREFIX $NEW_DIR/$f
            if [ $? != 0 ]; then
                echo "$f"
            fi
        else
            mv $NEW_DIR/tmp/$f $NEW_DIR/$f
        fi
    fi
done

rm -rv $NEW_DIR/tmp