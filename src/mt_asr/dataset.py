#!/usr/bin/env python
#
# Multi-Talker ASR dataset
from typing import List, Dict, Union

import torch
import numpy as np
from lhotse import CutSet, SupervisionSegment, fastcopy, MonoCut
from torch.utils.data import Dataset

from transformers.utils import logging
from data.local_datasets import TS_ASR_DatasetSuperclass, is_fake_spkr
from data.collators import DataCollator
from utils.general import round_nearest
from utils.training_args import CustomTrainingArguments, DataArguments, ModelArguments
from models.sot_dicow.config import SOTStyle

logging.set_verbosity_debug()
logger = logging.get_logger("mt_asr")


class MT_ASR_Dataset(Dataset):
    """Dataset for multi-talker automatic speech recognition.

    Handles multiple speakers in audio by filtering and padding supervisions
    to match the model's expected number of speakers.
    """
    def __init__(self, dataset: TS_ASR_DatasetSuperclass, model_spkrs: int):
        self.dataset = dataset
        self.num_spkrs = model_spkrs
        self.mt_cuts = self._prepare_mt_cuts(dataset.cset)
        self.debug = dataset.debug if hasattr(dataset, "debug") else False

    def filter_mt_speakers(self, cut: MonoCut) -> MonoCut:
        """
        Filters the given cut to only include the top N speakers based on their duration in the cut.

        Args:
            cut (MonoCut): The cut to be filtered.

        Returns:
            MonoCut: The filtered cut with only the top N speakers.
        """
        spk_ids = list(CutSet.from_cuts([cut]).speakers)
        spk_ids_map = {spk_id : i for spk_id, i in enumerate(spk_ids)}
        if len(spk_ids) > self.num_spkrs:
            logger.warning("Too many speakers (%d) in cut: %s", len(spk_ids), cut.id)
            # Sort speakers by duration
            spk_durs = cut.speakers_audio_mask(speaker_to_idx_map=spk_ids_map).sum(axis=1)
            sorted_spkr_idxs = np.argsort(-spk_durs) # descending order
            new_spkr_idxs = sorted_spkr_idxs[:self.num_spkrs]
            new_spkr_ids = [spk_ids[idx] for idx in new_spkr_idxs]
            new_cut = cut.filter_supervisions(lambda sup: sup.speaker in new_spkr_ids)
            return new_cut
        else:
            return cut

    def _prepare_mt_cuts(self, original_cuts):
        n_spkr = self.num_spkrs
        mt_cuts = original_cuts.map(self.filter_mt_speakers)

        # Pad each cut with empty supervisions up to n_spkr
        def pad_supervisions(cut: MonoCut):
            current_spkrs = len(CutSet.from_cuts([cut]).speakers)
            if current_spkrs < n_spkr:
                # Create empty supervisions with same duration as cut
                # This should create stno_mask with all 0s for target speaker
                empty_sups = [
                    SupervisionSegment(
                        id=f"{cut.id}_empty_{i}",
                        recording_id=cut.recording_id,
                        start=0,
                        duration=0,
                        text="",
                        speaker=f"ZZZZ_fake_{cut.id}_empty_spkr_{i - current_spkrs}",
                    )
                    for i in range(current_spkrs, n_spkr)
                ]
                new_cut = fastcopy(cut)
                new_cut.supervisions.extend(empty_sups)
                return new_cut
            return cut

        return mt_cuts.map(pad_supervisions)

    def __len__(self):
        if self.debug:
            return 2
        return len(self.mt_cuts)

    @staticmethod
    def _post_process(sample, sid):
        if is_fake_spkr(sid):
            sample["is_valid"] = False
        else:
            sample["is_valid"] = True
        return sample

    def __getitem__(self, idx):
        if idx > len(self):
            raise IndexError("Index out of range")

        cut = self.mt_cuts[idx]
        spk_ids = sorted(list(CutSet.from_cuts([cut]).speakers))
        samples = [self._post_process(self.dataset.cut_to_sample(cut, sid), sid) for sid in spk_ids]

        if len(samples) > self.num_spkrs:
            raise ValueError("Detected more speakers than model can handle")
        return samples

    @property
    def cset(self) -> CutSet:
        """Get the multi-talker cut set."""
        return self.mt_cuts

    @property
    def references(self) -> CutSet:
        """Get the reference cut set from the underlying dataset."""
        return self.dataset.references


class MT_ASR_SOT_Dataset(MT_ASR_Dataset):
    """
    Dataset for serialized output training (SOT)

    Returns multi-speaker samples with single transcription, e.g.
    audio: [N_speakers]
    labels: "speaker1 text <sc1> speaker2 text <sc2> speaker3 text <sc3> speaker4 text <sc4>"
    """
    SOT_STYLES = SOTStyle

    def __init__(self, dataset: TS_ASR_DatasetSuperclass, model_spkrs: int, augment_speaker_order=False, sot_style: Union[str, SOTStyle] = SOTStyle.UTTERANCE, explicit_speaker_tokens=False):
        """
        Args:
            dataset: Underlying dataset.
            model_spkrs: Number of speakers for the model.
            augment_speaker_order: Whether to augment speaker order.
            sot_style: SOT style, one of MT_ASR_SOT_Dataset.SOT_STYLES ("utterance" or "speaker").
        Raises:
            ValueError: If sot_style is not a valid option.
        """
        super().__init__(dataset, model_spkrs)
        if not dataset.use_timestamps:
            raise ValueError("SOT needs timestamps to distinguish speakers!")
        if isinstance(sot_style, str):
            try:
                sot_style = SOTStyle(sot_style)
            except ValueError:
                raise ValueError(f"sot_style must be one of {[e.value for e in SOTStyle]}, got '{sot_style}'")
        elif not isinstance(sot_style, SOTStyle):
            raise ValueError(f"sot_style must be a string or SOTStyle enum, got {type(sot_style)}")
        self.augment_spaker_order = augment_speaker_order
        self.sot_style = sot_style
        self.explicit_speaker_tokens = explicit_speaker_tokens

    def __getitem__(self, idx):
        samples = super().__getitem__(idx)
        cut = self.mt_cuts[idx]

        spk_ids = sorted(list(CutSet.from_cuts([cut]).speakers))
        spk_ids = [spk_id for spk_id in spk_ids if not is_fake_spkr(spk_id)]
        if self.augment_spaker_order:
            spk_ids, samples = self._reorder_speakers(spk_ids, samples)

        new_cut = cut.filter_supervisions(lambda s: not is_fake_spkr(s.speaker))
        sot_tra = self._sot_transcript(new_cut, spk_ids)

        for sample in samples:
            sample["sot_transcript"] = sot_tra

        return samples

    def _sot_transcript(self, cut: MonoCut, spk_ids):
        spk_ids_map = dict(zip(spk_ids, range(len(spk_ids))))
        segments = self._merge_supervisions(
            cut, spk_ids, max_timestamp_pause=self.dataset.max_timestamp_pause
        )
        segments.sort(key=lambda s: s.start)

        match self.sot_style:
            case SOTStyle.UTTERANCE:
                # sort on segment start time
                pass
            case SOTStyle.SPEAKER:
                # # Find the order of speakers based on their first appearance
                # speaker_first_appearance = np.array([float("inf") for _ in spk_ids])
                # for seg in segments:
                #     if seg.start < speaker_first_appearance[spk_ids_map[seg.speaker]]:
                #         speaker_first_appearance[spk_ids_map[seg.speaker]] = seg.start
                #         if (speaker_first_appearance < float("inf")).all():
                #             break
                # speaker_first_appearance = zip(spk_ids, speaker_first_appearance)
                # # Sort speakers by their first appearance
                # ordered_speakers = [spk for spk, _ in sorted(speaker_first_appearance, key=lambda x: x[1])]
                # For each speaker, get their segments sorted by start
                speaker_segments = []
                for spk in spk_ids:
                    spk_segs = [seg for seg in segments if seg.speaker == spk]
                    speaker_segments.extend(spk_segs)
                    # MAYBE ADD SOME SPECIAL TAG HERE
                segments = speaker_segments
            case _:
                raise NotImplementedError(f"Unknown SOTStyle: {self.sot_style}")

        return ("" if self.dataset.use_timestamps else " ").join(
            [
                self._get_segment_text_with_timestamps(
                    segment, self.dataset.use_timestamps, self.dataset.text_norm, spk_ids_map
                )
                for segment in segments
            ]
        )

    def _reorder_speakers(self, spk_ids, samples):
        # Create new order of indices
        new_order = list(range(len(spk_ids)))
        # Shuffle the order if needed
        np.random.shuffle(new_order)
        # Reorder both speaker IDs and samples according to new order
        reordered_samples = [samples[new_order[i]] if i < len(new_order) else samples[i] for i in range(len(samples))]
        # Reorder spk_ids
        reordered_spk_ids = [spk_ids[i] for i in new_order]
        return reordered_spk_ids, reordered_samples

    def _get_segment_text_with_timestamps(self, segment: SupervisionSegment,
                                         use_timestamps: bool, text_norm, sid_map):
        sidx = sid_map[segment.speaker]
        start = self._get_speaker_timestamp_token(segment.start, sidx, is_start=True)
        end = self._get_speaker_timestamp_token(segment.end_, sidx, is_start=False)
        text = text_norm(segment.text_)
        if not text:
            return ""
        if use_timestamps:
            text = start + text + end
        else:
            text = f"<|spk{sidx}|>" + text
        return text

    def _get_speaker_timestamp_token(self, timestamp, sidx, is_start=True):
        ts = round_nearest(timestamp, 0.02)
        if self.explicit_speaker_tokens:
            if is_start:
                return f"<|spk{sidx}|><|{ts:.2f}|>"
            else:
                return f"<|{ts:.2f}|>"
        else:
            if sidx > 0:
                return f"<|{ts:.2f}_spk{sidx}|>"
            else:
                return f"<|{ts:.2f}|>"

    def _merge_supervisions(self, cut: MonoCut, spk_ids, max_timestamp_pause=0.0):
        all_lists = []
        for spk_id in spk_ids:
            new_merged_list = []
            target_spk_cut = cut.filter_supervisions(lambda s: s.speaker == spk_id)
            for supervision in sorted(target_spk_cut.supervisions, key=lambda x: x.start):
                if len(new_merged_list) == 0:
                    supervision.end_ = supervision.end
                    supervision.text_ = supervision.text
                    new_merged_list.append(supervision)
                else:
                    if round(new_merged_list[-1].end_, 2) == round(supervision.start, 2) or supervision.start - \
                            new_merged_list[-1].end_ <= max_timestamp_pause:
                        new_merged_list[-1].end_ = supervision.end
                        new_merged_list[-1].text_ = new_merged_list[-1].text_ + " " + supervision.text
                    else:
                        supervision.end_ = supervision.end
                        supervision.text_ = supervision.text
                        new_merged_list.append(supervision)
            all_lists.extend(new_merged_list)
        return all_lists


class SOTBaselineDataset(MT_ASR_SOT_Dataset):
    """This dataset is intended for use with Baseline SOT Whisper"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        samples = super().__getitem__(idx)
        return samples[:1]


class MT_Data_Collator(DataCollator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
            self, orig_inputs: List[List[Dict[str, Union[List[int], torch.Tensor]]]]
    ) -> Dict[str, torch.Tensor]:
        # B x Speakers
        inputs = [input for group in orig_inputs for input in group]  # flatten
        # Save to the inputs sizes of each group

        processed_inputs = DataCollator.__call__(self, inputs)
        processed_inputs["is_valid"] = torch.tensor([item["is_valid"] for group in orig_inputs for item in group],
                                                    device=processed_inputs['stno_mask'].device)
        processed_inputs["per_group_sizes"] = processed_inputs["is_valid"].reshape(len(orig_inputs), -1).sum(dim=1)
        return processed_inputs


class MT_SOT_Data_Collator(MT_Data_Collator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter_long_samples = False

    def __call__(self, orig_inputs: List[List[Dict[str, Union[List[int], torch.Tensor]]]]):
        sot_tra = [input["sot_transcript"] for group in orig_inputs for input in group]
        processed_inputs = super().__call__(orig_inputs)

        if len(processed_inputs["input_features"]) != len(sot_tra):
            raise ValueError(f"Number of input features ({len(processed_inputs['input_features'])}) does not match number of SOT transcripts ({len(sot_tra)})")

        # TODO: UNCOMMENT THIS IN FUTURE FOR CTC BRANCH
        # longform = [sample['is_long_form'] for group in orig_inputs for sample in group]
        # if not any(longform):
        #     processed_inputs['ctc_labels'] = processed_inputs['labels']  # for future
        #     processed_inputs['ctc_upp_labels'] = processed_inputs['upp_labels']  # for future

        sot_labels = self.tokenizer(sot_tra, padding="longest", max_length=self.max_length, return_tensors="pt")
        processed_inputs["sot_labels"] = sot_labels.input_ids.masked_fill(sot_labels.attention_mask.ne(1), -100)

        is_long_form = all([sample['is_long_form'] for inputs in orig_inputs for sample in inputs])
        if (sot_labels.attention_mask.sum(dim=1) > self.max_length).any() and not is_long_form:
            # truncate the sot_labels
            logger.warning("Truncating SOT labels to max length %d", self.max_length)
            processed_inputs["sot_labels"] = processed_inputs["sot_labels"][:, :self.max_length]

        if (processed_inputs["sot_labels"][:, 0] == self.bos_token_id).all().cpu().item():
            processed_inputs["sot_labels"] = processed_inputs["sot_labels"][:, 1:]
        processed_inputs['sot_upp_labels'] = processed_inputs['sot_labels'].clone().apply_(
            lambda x: self.tokenizer.upper_cased_tokens.get(int(x)) if int(
                x) in self.tokenizer.upper_cased_tokens else x)

        # This is just a hack to force correct batch_size in generate()
        # Note that we keep STNO mask in it's original shape
        # TODO: let's get rid off is_valid field
        first_valid_idx = torch.arange(0, processed_inputs['input_features'].shape[0], len(orig_inputs[0]))
        for key in (['input_features', 'attention_mask', 'sot_labels', 'sot_upp_labels', 'is_valid'] + (['labels', 'upp_labels'] if is_long_form else [])):
            processed_inputs[key] = processed_inputs[key][first_valid_idx]

        if 'forced_decoder_ids' in processed_inputs:
            processed_inputs['forced_decoder_ids'] = processed_inputs['forced_decoder_ids'][first_valid_idx]

        # For NSF we need to keep
        processed_inputs["per_group_sizes"] = processed_inputs["per_group_sizes"].max().repeat(first_valid_idx.size(0))
        return processed_inputs


def build_mt_asr_datasets(dataset, model_args: ModelArguments, data_args: DataArguments, train_args: CustomTrainingArguments, is_train=False):
    """Return MT-ASR dataset wrapper"""
    if isinstance(dataset, dict):
        return {k: build_mt_asr_datasets(d, model_args, data_args, train_args, is_train) for k, d in dataset.items()}
    elif isinstance(dataset, list):
        return [build_mt_asr_datasets(d, model_args, data_args, train_args, is_train) for d in dataset]
    if model_args.mt_sot:
        if train_args.use_fddt and is_train:
            return MT_ASR_SOT_Dataset(
                dataset,
                model_args.mt_num_speakers, # type: ignore
                augment_speaker_order=data_args.mt_sot_augment_speaker_order,
                sot_style=data_args.mt_sot_style,
                explicit_speaker_tokens=data_args.mt_sot_explicit_speaker_tokens,
            )
        if train_args.use_fddt:
            return MT_ASR_SOT_Dataset(
                dataset,
                model_args.mt_num_speakers, # type: ignore
                sot_style=data_args.mt_sot_style,
                explicit_speaker_tokens=data_args.mt_sot_explicit_speaker_tokens,
            )
        # only SOT baseline does not need target amplifiers
        return SOTBaselineDataset(dataset, model_args.mt_num_speakers)
    return MT_ASR_Dataset(dataset, model_args.mt_num_speakers) # type: ignore
