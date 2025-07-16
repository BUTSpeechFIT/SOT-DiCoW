#!/bin/env python

# This script is intended for training data preparation
# with real diarization.
#
# See also prepare_data.py for eval data preparation
# with real diarization.

from functools import partial
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from lhotse import CutSet, SupervisionSegment, compute_num_samples
from lhotse.cut.base import Cut

from data.mappings import ns_mapping2


def prepare_diar_cutset_for_training(cutset: CutSet, diar_dir: Union[str, Path], output_path: Optional[Union[str, Path]]=None):
    _add_diar_info = partial(add_diar_info, diar_dir=Path(diar_dir))
    new_cset = cutset.map(_add_diar_info)

    if output_path is not None:
        new_cset.to_file(output_path)
    return new_cset

def _prepare_soft_mask(c: Cut, activations_f):
    soft_spk_mask = np.load(activations_f) / 10
    soft_spk_mask = soft_spk_mask.T[..., None].repeat(int(0.02*16000), axis=-1).reshape((soft_spk_mask.shape[1], -1))
    st = compute_num_samples(c.start, c.sampling_rate) if c.start > 0 else 0
    et = compute_num_samples(c.end, c.sampling_rate)
    soft_spk_mask = soft_spk_mask[:, st:et]
    return soft_spk_mask


def _prepare_speaker_map(c: Cut, soft_mask: np.array):
    speakers = sorted(CutSet.from_cuts([c]).speakers)
    spk_map1 = {v: k for k, v in enumerate(speakers)}
    spk_mask = c.speakers_audio_mask(speaker_to_idx_map=spk_map1)
    ref_spkrs, soft_spkrs = linear_sum_assignment(-np.dot(spk_mask, soft_mask.T))
    spk_map = {speakers[k]: v for k, v in zip(ref_spkrs, soft_spkrs)}
    if len(speakers) < soft_mask.shape[0]:
        # we have more speakers in diarization mask
        empty_segment = SupervisionSegment(
            id=f"{c.id}_fake-speaker-0",
            recording_id=c.recording_id,
            start=0.0,
            duration=c.duration,
            speaker=f"{c.id}_fake-speaker-0",
            text="",
        )
        additional_speakers = [i for i in range(soft_mask.shape[0]) if i not in soft_spkrs]
        most_active_additional_speaker = np.argmax(np.sum(soft_mask[additional_speakers, :], axis=1))
        spk_map[f"{c.id}_fake-speaker-0"] = most_active_additional_speaker
        c.supervisions.append(empty_segment)
    elif len(speakers) > soft_mask.shape[0]:
        # we have undetected ref speakers, we need to remove supervisions for them
        c = c.filter_supervisions(lambda s: s.speaker in spk_map)
    return spk_map


def add_diar_info(cut: Cut, diar_dir: Path):
    session_id = ns_mapping2.get(cut.recording_id, cut.recording_id)
    activations_f = diar_dir/"npy"/f"{session_id}_soft_activations.npy"
    soft_mask = _prepare_soft_mask(cut, activations_f)
    spk_map = _prepare_speaker_map(cut, soft_mask)
    spk_map = {k: int(v) for k, v in spk_map.items()}
    norm_constant = 10
    shift = int(0.02 * 16000)  # 320 samples
    if activations_f.is_file():
        cut.custom={
            'soft_activations': str(activations_f),
            'shift_samples': shift,
            'norm_constant': norm_constant,
            'soft_speaker_map': spk_map,
        }
    else:
        print("Warning: ", activations_f, " doesn't exist")
    return cut


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO, force=True
    )

    import argparse
    parser = argparse.ArgumentParser(description="Prepare diarization with soft activations")
    parser.add_argument("cutset", help="Path to NSF GT cutset")
    parser.add_argument("diar_root_dir", help="Path to NSF diarization root dir, such that <root_dir>/npy is a dir")
    parser.add_argument("--output-path", help="Path where new cutset should be stored", default=None)
    parser.add_argument("--nj", default=1, type=int)
    args = parser.parse_args()

    cset = CutSet.from_file(args.cutset)
    prepare_diar_cutset_for_training(cset, args.diar_root_dir, args.output_path)
