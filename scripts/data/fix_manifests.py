#!/usr/bin/env python
from tabnanny import verbose
from lhotse import fastcopy, CutSet, RecordingSet, validate, fix_manifests
import sys

from data.prepare_data import split_overlapping_segments



if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <cutset_path> <new_cutset_path>")
        sys.exit(1)

    cutset: CutSet = CutSet.from_file(sys.argv[1])
    recs, sups, _ = cutset.decompose(output_dir="./manifests/tmp", verbose=True)
    recs, sups = fix_manifests(recs, sups)
    new_cutset = CutSet.from_manifests(recs, sups, output_path=sys.argv[2])

    print("Saving new cuts to ", sys.argv[2])
    validate(new_cutset, read_data=True)

