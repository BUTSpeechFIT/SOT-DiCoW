#!/usr/bin/env python
from lhotse import fastcopy, CutSet, RecordingSet, validate, fix_manifests
import sys

from data.prepare_data import split_overlapping_segments



if __name__ == '__main__':
    if len(sys.argv) < 5:
        print(f"Usage: {sys.argv[0]} <cutset_path> <max_dur_in_sec> <nj> <new_cutset_path>")
        sys.exit(1)

    cutset = CutSet.from_file(sys.argv[1])
    max_dur = float(sys.argv[2])
    nj = int(sys.argv[3])
    new_cutset = split_overlapping_segments(cutset, max_dur, num_jobs=nj)

    print("Saving new cuts to ", sys.argv[4])
    new_cutset.to_file(sys.argv[4])
    validate(new_cutset, read_data=True)

