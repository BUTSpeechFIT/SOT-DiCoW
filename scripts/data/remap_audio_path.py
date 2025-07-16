#!/usr/bin/env python
from lhotse import fastcopy, CutSet, RecordingSet, validate
import sys


def replace_audio_path(cutset: CutSet, orig_prefix: str, new_prefix: str):
    return cutset.map(
        lambda c: fastcopy(
            c, recording=_replace_audio_path([c.recording], orig_prefix, new_prefix)[0]
        )
    )
    # recordings, supervisions, features = cutset.decompose()  # ignore features
    # new_recordings = _replace_audio_path(recordings, orig_prefix, new_prefix)
    # return CutSet.from_manifests(supervisions=supervisions, recordings=new_recordings, features=features)


def _replace_audio_path(recordings: RecordingSet, orig_prefix: str, new_prefix: str):
    if recordings[0].sources[0].source.startswith(new_prefix):
        return recordings
    new_recs = (
        fastcopy(
            rec,
            sources=[
                fastcopy(src, source=src.source.replace(orig_prefix, new_prefix))
                for src in rec.sources
            ],
        )
        for rec in recordings
    )
    return RecordingSet.from_recordings(new_recs)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(
            f"Usage: {sys.argv[0]} <cutset_path> <orig_audio_prefix> <new_audio_prefix> <new_cutset_path>"
        )
        sys.exit(1)

    cutset = CutSet.from_file(sys.argv[1])
    orig_prefix = sys.argv[2]
    new_preifx = sys.argv[3]
    if isinstance(cutset, RecordingSet):
        new_cutset = _replace_audio_path(cutset, orig_prefix, new_preifx)
    else:
        new_cutset = replace_audio_path(cutset, orig_prefix, new_preifx)
    print("Saving new cuts to ", sys.argv[4])
    new_cutset.to_file(sys.argv[4])
