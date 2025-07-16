#!/usr/bin/env python
from lhotse import fastcopy, CutSet, RecordingSet, validate
import sys

def drop_features(cutset: CutSet):
    recordings, supervisions, _ = cutset.decompose()  # ignore features
    return CutSet.from_manifests(supervisions=supervisions, recordings=recordings, features=None)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <cutset_path> <new_cutset_path>")
        sys.exit(1)

    cutset = CutSet.from_file(sys.argv[1])
    new_cutset = drop_features(cutset)
    print("Saving new cuts to ", sys.argv[2])
    new_cutset.to_file(sys.argv[2])
    validate(new_cutset, read_data=True)