#!/usr/bin/env python
from lhotse import fastcopy, CutSet, RecordingSet, validate
import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <cutset_path> <new_cutset_path>")
        sys.exit(1)

    cutset = CutSet.from_file(sys.argv[1])
    validate(cutset, read_data=True)
    print("Saving new cuts to ", sys.argv[2])
    cutset.to_file(sys.argv[2])

