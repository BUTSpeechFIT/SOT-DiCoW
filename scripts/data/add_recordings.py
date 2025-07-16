#!/usr/bin/env python
from lhotse import fastcopy, CutSet, RecordingSet, validate, SupervisionSet
import sys

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <cutset_path> <rec_path> <new_cutset_path>")
        sys.exit(1)

    cuts = CutSet.from_file(sys.argv[1])
    sups, _, _ = cuts.decompoe()
    recs = RecordingSet.from_file(sys.argv[2])
    cutset = CutSet.from_manifests(recs, sups)
    validate(cutset, read_data=True)
    print("Saving new cuts to ", sys.argv[2])
    cutset.to_file(sys.argv[2])

