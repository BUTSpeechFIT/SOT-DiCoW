import os
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from pathlib import Path
from typing import List, Union

import lhotse
import numpy as np
import torch
from lhotse import CutSet
from lhotse.cut import Cut
from torch.utils.data import Dataset
from transformers.utils import logging

from utils.general import round_nearest
from utils.training_args import DataArguments, DecodingArguments

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


def fix_audio_path(cutset: CutSet, audio_path_prefix: str, audio_path_prefix_replacement: str):
    for cut in cutset:
        if hasattr(cut, 'recording'):
            for src in cut.recording.sources:
                src.source = src.source.replace(audio_path_prefix, audio_path_prefix_replacement)


def is_fake_spkr(spk_id):
    return spk_id.startswith("fake_") or spk_id.startswith("ZZZZ_fake_")


def add_timestamps(transcript, sample_len, sampling_rate=16_000, precision=0.02):
    return {"transcript": f"<|0.00|>{transcript}<|{round_nearest(sample_len / sampling_rate, precision):.2f}|>"}


class TS_ASR_DatasetSuperclass:
    """
        Contains all dataset-related methods that both, random and segmented datasets use.
    """

    def __init__(self, cutsets, text_norm=lambda x: x, do_augment=False, use_timestamps=False,
                 empty_transcript_ratio=0.00, train_with_diar_outputs=None, musan_noises=None, audio_path_prefix=None,
                 audio_path_prefix_replacement=None,
                 max_timestamp_pause=0.0, vad_from_alignments=False, model_features_subsample_factor=2,
                 dataset_weights=None,
                 feature_extractor=None,
                 global_lang_id=None,
                 *args,
                 **kwargs):

        self.cutsets = cutsets
        self.dataset_weights = dataset_weights
        if dataset_weights is None:
            self.dataset_weights = [1] * len(cutsets)

        assert len(self.cutsets) == len(self.dataset_weights), "cutsets and dataset_weights must have the same length"

        self.cset = reduce(lambda a, b: a + b, self.cutsets)

        self.audio_path_prefix = audio_path_prefix
        self.audio_path_prefix_replacement = audio_path_prefix_replacement
        self.max_timestamp_pause = max_timestamp_pause
        self.use_timestamps = use_timestamps
        self.text_norm = text_norm
        self.feature_extractor = feature_extractor
        self.model_features_subsample_factor = model_features_subsample_factor
        self.global_lang_id = global_lang_id
        self.feature_cache = None
        self.feature_cache_id = None
        if audio_path_prefix and audio_path_prefix_replacement:
            logger.info(f"Fixing audio paths from {audio_path_prefix} to {audio_path_prefix_replacement}")
            for cutset in self.cutsets:
                fix_audio_path(cutset, audio_path_prefix, audio_path_prefix_replacement)
        self.prepare_cuts()

    @staticmethod
    def get_number_of_speakers_from_monocut(cut):
        spks = set()
        for suppervision in cut.supervisions:
            spks.add(suppervision.speaker)
        return len(spks)

    @staticmethod
    def get_cut_spks(cut):
        spks = set()
        for suppervision in cut.supervisions:
            spks.add(suppervision.speaker)
        return sorted(spks)

    def get_segment_text_with_timestamps(self, segment, use_timestamps, text_norm, skip_end_token):
        start = f"<|{round_nearest(segment.start, 0.02):.2f}|>"
        end = f"<|{round_nearest(segment.end_, 0.02):.2f}|>"
        text = text_norm(segment.text_)
        if not text:
            return ""
        if skip_end_token:
            end = ""
        if use_timestamps:
            text = start + text + end
        return text

    def merge_supervisions(self, target_spk_cut):
        new_merged_list = []
        for supervision in sorted(target_spk_cut.supervisions, key=lambda x: x.start):
            if len(new_merged_list) == 0:
                supervision.end_ = supervision.end
                supervision.text_ = supervision.text
                new_merged_list.append(supervision)
            else:
                if round(new_merged_list[-1].end_, 2) == round(supervision.start, 2) or supervision.start - \
                        new_merged_list[-1].end_ <= self.max_timestamp_pause:
                    new_merged_list[-1].end_ = supervision.end
                    new_merged_list[-1].text_ = new_merged_list[-1].text_ + " " + supervision.text
                else:
                    supervision.end_ = supervision.end
                    supervision.text_ = supervision.text
                    new_merged_list.append(supervision)
        return new_merged_list

    def prepare_cuts(self):
        self.to_index_mapping = []
        for cutset, weight in zip(self.cutsets, self.dataset_weights):
            with ThreadPoolExecutor() as executor:
                spk_per_cut = list(executor.map(self.get_number_of_speakers_from_monocut, cutset.cuts))
            spk_per_cut = np.array(spk_per_cut) * weight
            self.to_index_mapping.append(spk_per_cut)
        self.to_index_mapping = np.cumsum(np.concatenate(self.to_index_mapping))

    def get_stno_mask(self, cut: Cut, speaker_id: str):
        speakers = list(sorted(CutSet.from_cuts([cut]).speakers))
        speakers_to_idx = {spk: idx for idx, spk in enumerate(filter(lambda sid: not is_fake_spkr(sid), speakers))}
        for spk in speakers:
            if is_fake_spkr(spk):
                # this will make sure that fake speaker has larger ID than real speakers
                speakers_to_idx[spk] = len(speakers_to_idx)

        spk_mask = cut.speakers_audio_mask(speaker_to_idx_map=speakers_to_idx)

        # Pad to match features
        pad_len = (self.feature_extractor.n_samples - spk_mask.shape[-1]) % self.feature_extractor.n_samples
        spk_mask = np.pad(spk_mask, ((0, 0), (0, pad_len)), mode='constant')

        # Downsample to meet model features sampling rate
        spk_mask = spk_mask.astype(np.float32).reshape(spk_mask.shape[0], -1,
                                                       self.model_features_subsample_factor * self.feature_extractor.hop_length).mean(
            axis=-1)

        speaker_index = speakers_to_idx[speaker_id]

        return self._create_stno_masks(spk_mask, speaker_index)

    @staticmethod
    def _create_stno_masks(spk_mask: np.ndarray, s_index: int):
        non_target_mask = np.ones(spk_mask.shape[0], dtype="bool")
        non_target_mask[s_index] = False
        sil_frames = (1 - spk_mask).prod(axis=0)
        anyone_else = (1 - spk_mask[non_target_mask]).prod(axis=0)
        target_spk = spk_mask[s_index] * anyone_else
        non_target_spk = (1 - spk_mask[s_index]) * (1 - anyone_else)
        overlapping_speech = spk_mask[s_index] - target_spk
        stno_mask = np.stack([sil_frames, target_spk, non_target_spk, overlapping_speech], axis=0)
        return stno_mask

    def get_features(self, cut: Cut):
        if self.feature_cache is not None and cut.id == self.feature_cache_id:
            return self.feature_cache

        samples, sr = cut.load_audio().squeeze(), cut.sampling_rate
        batch = self.feature_extractor(
            samples, return_tensors="pt",
            sampling_rate=sr, return_attention_mask=True,
            truncation=False, padding="longest",
            pad_to_multiple_of=self.feature_extractor.n_samples
        )
        feats, attn_mask = batch['input_features'], batch['attention_mask']
        self.feature_cache = (feats, attn_mask)
        self.feature_cache_id = cut.id
        return feats, attn_mask

    def cut_to_sample(self, cut: Cut, speaker_id: str):
        stno_mask = self.get_stno_mask(cut, speaker_id)
        features, att_mask = self.get_features(cut)

        last_segment_unfinished = cut.per_spk_flags[speaker_id] if hasattr(cut, "per_spk_flags") else False
        target_spk_cut = cut.filter_supervisions(lambda x: x.speaker == speaker_id)
        merged_supervisions = self.merge_supervisions(target_spk_cut)
        transcription = ("" if self.use_timestamps else " ").join(
            [self.get_segment_text_with_timestamps(segment, self.use_timestamps, self.text_norm, (idx == len(merged_supervisions) - 1) and last_segment_unfinished) for idx, segment in
             enumerate(merged_supervisions)])

        outputs = {"input_features": features, "stno_mask": torch.tensor(stno_mask), "attention_mask": att_mask,
                   "transcript": transcription, "is_long_form": False}

        if hasattr(cut, "lang"):
            outputs["language"] = cut.lang
        elif self.global_lang_id:
            outputs["language"] = self.global_lang_id
        else:
            raise ValueError("Please if your dataset does not provide lang ids, set global lang id.")

        return outputs


class TS_ASR_Dataset(TS_ASR_DatasetSuperclass, Dataset):
    def __init__(self, *args, **kwargs):
        TS_ASR_DatasetSuperclass.__init__(self, *args, **kwargs)

    def __len__(self):
        return self.to_index_mapping[-1]

    def __getitem__(self, idx):
        if idx > len(self):
            raise 'Out of range'

        cut_index = np.searchsorted(self.to_index_mapping, idx, side='right')
        cut = self.cset[cut_index]
        spks = self.get_cut_spks(cut)
        local_sid = (idx - self.to_index_mapping[cut_index]) % len(spks)
        sid = spks[local_sid]
        return self.cut_to_sample(cut, sid)



class LhotseLongFormDataset(TS_ASR_Dataset):
    def __init__(self, cutset: CutSet, is_multichannel: bool = False,
                 references: CutSet = None, provide_gt_lang: bool = False,
                 soft_vad_temp=None, debug: bool = False, **kwargs):
        super().__init__(cutsets=[cutset], **kwargs)
        self._references = references

        if self._references is not None:
            rids = set(cut.recording_id for cut in self.references)
            cids = set(cut.recording_id for cut in self.cset)
            if len(rids & cids) == 0:
                raise ValueError("'references' doesn't match inference cuts")  # fail immediately
            if cids != rids:
                logger.warn("'cutset' and 'references' aren't the same sets")

        self.is_multichannel = is_multichannel
        self.soft_vad_temp = soft_vad_temp
        self.provide_gt_lang = provide_gt_lang
        self.debug = debug

    @property
    def references(self) -> CutSet:
        """Returns the reference CutSet for evaluation.

        This property allows using separate reference and hypothesis CutSets, which is useful
        for evaluation scenarios like diarization where we want to score system outputs
        against ground truth references. If no separate references were provided during
        initialization, falls back to using the input CutSet as references.

        Returns:
            CutSet: The reference CutSet containing ground truth transcripts and speaker labels
        """
        if self._references is not None:
            return self._references
        return self.cset

    def cut_to_sample(self, cut: Cut, speaker_id):
        stno_mask = self.get_stno_mask(cut, speaker_id)
        features, att_mask = self.get_features(cut)

        outputs = {"input_features": features, "stno_mask": torch.tensor(stno_mask), "attention_mask": att_mask,
                   "transcript": f'{cut.id},{speaker_id}', "is_long_form": True}
        if self.provide_gt_lang:
            if hasattr(cut, "lang"):
                outputs["language"] = cut.lang
            elif self.global_lang_id:
                outputs["language"] = self.global_lang_id
            else:
                raise ValueError("Please if your dataset does not provide lang ids, set global lang id.")
        return outputs

def get_libri_dataset(txt_norm, train_path=None, dev_path=None):
    from datasets import load_dataset, concatenate_datasets, load_from_disk
    if train_path is None or dev_path is None:
        librispeech = load_dataset("librispeech_asr", name="all", trust_remote_code=True)
        librispeech = librispeech.map(lambda x: {"transcript": txt_norm(x)}, input_columns="text", num_proc=32)
        librispeech = librispeech.select_columns(["audio", "transcript", ])
        libri_train = concatenate_datasets([librispeech['train.clean.100'], librispeech['train.clean.360'],
                                            librispeech['train.other.500']])
        libri_dev = concatenate_datasets(
            [librispeech['validation.clean'], librispeech['validation.other'], librispeech['test.clean'],
             librispeech['test.other']])
    else:
        libri_train = load_from_disk(train_path)
        libri_dev = load_from_disk(dev_path)

    return libri_train, libri_dev


def get_nsf_dataset(text_norm, data_args):
    train_cutsets = reduce(lambda a, b: a + b, [lhotse.load_manifest(cutset) for cutset in data_args.train_cutsets])
    eval_cutsets = reduce(lambda a, b: a + b, [lhotse.load_manifest(cutset) for cutset in data_args.eval_cutsets])

    train_dataset = TS_ASR_Dataset(train_cutsets, do_augment=data_args.do_augment,
                                   use_timestamps=data_args.use_timestamps,
                                   musan_noises=data_args.musan_noises,
                                   text_norm=text_norm,
                                   empty_transcript_ratio=data_args.empty_transcripts_ratio,
                                   train_with_diar_outputs=data_args.train_with_diar_outputs,
                                   audio_path_prefix=data_args.audio_path_prefix,
                                   audio_path_prefix_replacement=data_args.audio_path_prefix_replacement,
                                   vad_from_alignments=data_args.vad_from_alignments,
                                   random_sentence_l_crop_p=data_args.random_sentence_l_crop_p,
                                   random_sentence_r_crop_p=data_args.random_sentence_r_crop_p,
                                   max_l_crop=data_args.max_l_crop,
                                   max_r_crop=data_args.max_r_crop, )

    eval_dataset = TS_ASR_Dataset(eval_cutsets,
                                  text_norm=text_norm,
                                  use_timestamps=data_args.use_timestamps,
                                  audio_path_prefix=data_args.audio_path_prefix,
                                  audio_path_prefix_replacement=data_args.audio_path_prefix_replacement,
                                  )

    return train_dataset, eval_dataset


def build_datasets(cutset_paths: List[Union[str, Path]], data_args: DataArguments, dec_args: DecodingArguments,
                   text_norm, container, diar_cutset_paths=None, **kwargs):
    logger.info('Using LhotseLongFormDataset')
    if cutset_paths is None or len(cutset_paths) == 0:
        raise ValueError("'cutset_paths' is None or empty. Please provide valid 'cutset_paths' for the dataset")
    if not all(Path(p).exists() for p in cutset_paths):
        wrong_paths = os.linesep.join([f"{'✗' if not Path(p).exists() else '✓'} {p}" for p in cutset_paths])
        raise ValueError(f"Some cutset paths do not exist:{os.linesep}{wrong_paths}")

    cutsets = [CutSet.from_file(path) for path in cutset_paths]

    if data_args.merge_eval_cutsets:
        cutsets = [reduce(lambda a, b: a + b, cutsets)]
        cutset_paths = ["reduced_from" + "_".join([os.path.basename(path) for path in cutset_paths])]
    if data_args.use_diar:
        if diar_cutset_paths is None or len(diar_cutset_paths) == 0:
            raise ValueError(
                "'diar_cutset_paths' is None or empty. Please provide valid 'diar_cutset_paths' for the dataset")
        if not all(Path(p).exists() for p in diar_cutset_paths):
            wrong_paths = os.linesep.join(
                [f"{'✗' if not Path(p).exists() else '✓'} {p}" for p in diar_cutset_paths])
            raise ValueError(f"Some diar cutset paths do not exist:{os.linesep}{wrong_paths}")
        refs = cutsets
        cutsets = [CutSet.from_file(path) for path in diar_cutset_paths]
        if data_args.merge_eval_cutsets:
            cutsets = [reduce(lambda a, b: a + b, cutsets)]
    else:
        refs = [None for _ in cutsets]

    return {os.path.basename(path).removesuffix(".jsonl.gz"): LhotseLongFormDataset(cutset=cutset, references=ref,
                                                                                    audio_path_prefix=data_args.audio_path_prefix,
                                                                                    audio_path_prefix_replacement=data_args.audio_path_prefix_replacement,
                                                                                    use_timestamps=data_args.use_timestamps,
                                                                                    text_norm=text_norm,
                                                                                    use_features=data_args.cache_features_for_dev,
                                                                                    feature_extractor=container.feature_extractor,
                                                                                    soft_vad_temp=dec_args.soft_vad_temp,
                                                                                    global_lang_id=data_args.global_lang_id,
                                                                                    provide_gt_lang=data_args.provide_gt_lang,
                                                                                    **kwargs,
                                                                                    ) for cutset, ref, path in
            zip(cutsets, refs, cutset_paths)}


