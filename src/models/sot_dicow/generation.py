# Author: Martin Kocour (BUT)

import copy
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.generation.logits_process import (
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    WhisperNoSpeechDetection
)
from transformers.utils import logging
from models.dicow.generation import DiCoWGenerationMixin
from models.sot_dicow.utils import WhisperSOTTimeStampLogitsProcessor, timestamp_tokens_len, convert_to_spk_mask, convert_to_fddt_mask

logger = logging.get_logger(__name__)


class SOTDiCoWGenerationMixin(DiCoWGenerationMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stno_mask_seek_skip = [None] * 20 # TODO: batch size

    def _retrieve_logit_processors(self, generation_config, logits_processor, begin_index, is_shortform, num_beams, device):
        if generation_config.return_timestamps is True:
            timestamp_processor = WhisperSOTTimeStampLogitsProcessor(
                generation_config,
                begin_index=begin_index,
                mt_num_speakers=self.config.mt_num_speakers,
            )
            logits_processor = (
                [timestamp_processor] if logits_processor is None else [timestamp_processor] + logits_processor
            )

        if generation_config.suppress_tokens is not None:
            suppress_tokens_processor = SuppressTokensLogitsProcessor(generation_config.suppress_tokens, device=device)
            logits_processor = (
                [suppress_tokens_processor]
                if logits_processor is None
                else [suppress_tokens_processor] + logits_processor
            )
            generation_config.suppress_tokens = None

        if generation_config.begin_suppress_tokens is not None:
            begin_suppress_processor = SuppressTokensAtBeginLogitsProcessor(
                generation_config.begin_suppress_tokens, begin_index=begin_index, device=device
            )
            logits_processor = (
                [begin_suppress_processor]
                if logits_processor is None
                else [begin_suppress_processor] + logits_processor
            )
            generation_config.begin_suppress_tokens = None

        if generation_config.no_speech_threshold is not None and not is_shortform:
            no_speech_detector = WhisperNoSpeechDetection(
                no_speech_token=generation_config.no_timestamps_token_id - 1,
                begin_index=begin_index,
                scores_is_logprobs=num_beams > 1,
            )
            logits_processor = (
                [no_speech_detector] if logits_processor is None else [no_speech_detector] + logits_processor
            )
            no_speech_detector.set_model(self)

        return logits_processor

    def _retrieve_segment(
            self,
            seek_sequence,
            seek_outputs,
            time_offset,
            timestamp_begin,
            seek_num_frames,
            time_precision,
            input_stride,
            prev_idx,
            idx,
            return_token_timestamps,
    ):
        # find the predicted "end of segment" predictions of Whisper
        # "end of segment" predictions occur whenever Whisper predicts a timestamp token
        timestamp_tokens: torch.Tensor = seek_sequence.ge(timestamp_begin)
        single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]
        timestamp_segment_indices = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]
        timestamp_segment_indices.add_(1)
        token_timestamps = seek_outputs[idx]["token_timestamps"] if return_token_timestamps else []
        frame_offset = (time_offset / time_precision * input_stride).to(seek_num_frames.dtype).tolist()

        # If whisper predicted a "end of segment" via a timestep token, let's go ever each
        # "end of segment" prediction and slice the decoding into segments accordingly
        if len(timestamp_segment_indices) > 0:
            # if the output contains two consecutive timestamp tokens
            slices = timestamp_segment_indices.tolist()
            segments = []
            if single_timestamp_ending:
                slices.append(len(seek_sequence))

            last_slice = 0
            # Add each segment to list of all segments
            for i, current_slice in enumerate(slices):
                sliced_tokens = seek_sequence[last_slice:current_slice]
                start_timestamp_pos = (sliced_tokens[0].item() - timestamp_begin) % timestamp_tokens_len()
                end_timestamp_pos = (sliced_tokens[-1].item() - timestamp_begin) % timestamp_tokens_len()
                segments.append(
                    {
                        "start_frames": frame_offset[prev_idx] + start_timestamp_pos * input_stride,
                        "end_frames": frame_offset[prev_idx] + end_timestamp_pos * input_stride,
                        "start": time_offset[prev_idx] + start_timestamp_pos * time_precision,
                        "end": time_offset[prev_idx] + end_timestamp_pos * time_precision,
                        "tokens": sliced_tokens,
                        "result": seek_outputs[idx],
                        "channel": (sliced_tokens[0].item() - timestamp_begin) // timestamp_tokens_len(),
                    }
                )
                if return_token_timestamps:
                    segments[-1]["token_timestamps"] = (
                            token_timestamps[last_slice:current_slice] + time_offset[prev_idx]
                    )
                last_slice = current_slice

            if single_timestamp_ending:
                # single timestamp at the end means no speech after the last timestamp.
                segment_offset = seek_num_frames[prev_idx]
            else:
                # otherwise, ignore the unfinished segment and seek to the last timestamp
                # here we throw away all predictions after the last predicted "end of segment"
                # since we are cutting right in the middle of an audio
                last_timestamp_pos = (seek_sequence[last_slice - 1].item() - timestamp_begin) % timestamp_tokens_len()
                segment_offset = last_timestamp_pos * input_stride
        else:
            # If whisper does not predict any "end of segment" token, then
            # the whole decoding is considered a segment and we add it to the list of segments
            timestamps = seek_sequence[timestamp_tokens.nonzero().flatten()]
            start_timestamp_pos = 0.0
            last_timestamp_pos = seek_num_frames[prev_idx] // input_stride
            skip = False
            segment_offset = seek_num_frames[prev_idx]

            if timestamps.numel() > 1:
                start_timestamp_pos = (timestamps[-2].item() - timestamp_begin) % timestamp_tokens_len()
                last_timestamp_pos = (timestamps[-1].item() - timestamp_begin) % timestamp_tokens_len()
            elif timestamps.numel() == 1:
                # no consecutive timestamps but it has a timestamp; use the last one.
                start_timestamp_pos = (timestamps[-1].item() - timestamp_begin) % timestamp_tokens_len()
                if start_timestamp_pos > 200:
                    # segment does not fit into decoding window, so we need to rollback
                    segment_offset = start_timestamp_pos * input_stride - 100  # timestamp might be inaccurate
                    skip = True
            else:
                # empty sequence, or sequence w/o timestamps
                skip = True

            if skip:
                segments = []
            else:
                segments = [
                    {
                        "start_frames": frame_offset[prev_idx] + start_timestamp_pos * input_stride,
                        "end_frames": frame_offset[prev_idx] + last_timestamp_pos * input_stride,
                        "start": time_offset[prev_idx] + start_timestamp_pos * time_precision,
                        "end": time_offset[prev_idx] + last_timestamp_pos * time_precision,
                        "tokens": seek_sequence,
                        "result": seek_outputs[idx],
                        "channel": (seek_sequence[0].item() - timestamp_begin) // timestamp_tokens_len(),
                    }
                ]
                if return_token_timestamps:
                    segments[-1]["token_timestamps"] = token_timestamps + time_offset[prev_idx]
                segment_offset = seek_num_frames[prev_idx]

        if segment_offset < seek_num_frames[prev_idx]:
            # Rollback to a start of the overlapping group
            rollback_segments = [
                seg for seg in segments
                if seg["start_frames"] - frame_offset[prev_idx] >= segment_offset or seg["end_frames"] - frame_offset[prev_idx] >= seek_num_frames[prev_idx] - 1
            ]
            new_segments = [
                seg for seg in segments
                if seg["start_frames"] - frame_offset[prev_idx] < segment_offset and seg["end_frames"] - frame_offset[prev_idx] < seek_num_frames[prev_idx] - 1
            ]
            if len(rollback_segments) > 0:
                # If there are any overlapping segments, move segment offset to the start of the first overlapping segment
                new_segment_offset = min(min(seg['start_frames'] - frame_offset[prev_idx] for seg in rollback_segments), segment_offset)
                if new_segment_offset > 0:
                    skip_segments = [
                        seg for seg in new_segments if seg["end_frames"] - frame_offset[prev_idx] >= segment_offset
                    ]
                    segment_offset = new_segment_offset
                    segments = new_segments
                else:
                    skip_segments = segments
                self._stno_mask_seek_skip[prev_idx] = skip_segments
            else:
                # there is no overlapping segments, so we can just rollback to the offset time
                self._stno_mask_seek_skip[prev_idx] = None

        if segment_offset <= 0:
            msg = f"Segments: {segments}"
            raise ValueError(f"Segment offset: {segment_offset} <= 0. This should not happen!\n{msg}")

        return segments, segment_offset

    def prepare_kwargs_for_generate(
        self,
        segment_input,
        cur_bsz,
        batch_idx_map,
        seek,
        num_segment_frames,
        max_frames,
        kwargs
    ):
        kwargs = copy.copy(kwargs)
        kwargs["attention_mask_enc"] = torch.ones(cur_bsz, segment_input.size(-1), device=segment_input.device)
        seek_vad = seek // 2
        num_frames_vad = num_segment_frames // 2
        max_frames_vad = max_frames // 2
        seek_num_frames = (max_frames_vad - seek_vad).clamp(max=num_frames_vad)

        stno_masks = []
        for i in range(cur_bsz):
            prev_i = batch_idx_map[i]
            end_idx = kwargs["per_group_sizes"].cumsum(0)[prev_i]
            start_idx = (end_idx - kwargs["per_group_sizes"])[prev_i]
            segment_input_slice = kwargs["stno_mask"][start_idx: end_idx, :, seek_vad[prev_i]: seek_vad[prev_i] + seek_num_frames[prev_i]]

            if segment_input_slice.shape[-1] < num_frames_vad:
                orig_len = segment_input_slice.shape[-1]
                # pad to 3000 if necessary
                segment_input_slice = torch.nn.functional.pad(
                    segment_input_slice, pad=(0, num_frames_vad - orig_len)
                )
                # set corresponding padding tokens to 1 in vad mask representing silence
                segment_input_slice[:, 0, orig_len:] = 1.0

            if self._stno_mask_seek_skip[prev_i]:
                spk_mask = convert_to_spk_mask(segment_input_slice)
                for seg in self._stno_mask_seek_skip[prev_i]:
                    channel_idx = seg['channel']
                    if channel_idx >= spk_mask.shape[0]:
                        continue
                    start_idx = int(seg['start_frames'] / 2 - seek_vad[prev_i])
                    start_idx = max(start_idx, 0)
                    end_idx = int(seg['end_frames'] / 2 - seek_vad[prev_i])
                    end_idx = min(end_idx, num_frames_vad)
                    spk_mask[channel_idx, start_idx:end_idx] = 0

                segment_input_slice = spk_mask = convert_to_fddt_mask(spk_mask)
                self._stno_mask_seek_skip[prev_i] = None

            stno_masks.append(segment_input_slice)
        kwargs["stno_mask"] = torch.cat(stno_masks, dim=0)
        self.stno_mask_seek = kwargs["stno_mask"]

        if "per_group_sizes" in kwargs:
            kwargs["per_group_sizes"] = kwargs["per_group_sizes"][batch_idx_map]

        if self.vad_seek_callback is not None:
            self.vad_seek_callback(kwargs["stno_mask"])

        return kwargs

    def _fix_timestamps_from_segmentation(self, sequences):
        timestamp_begin = self.tokenizer.get_vocab()["<|0.00|>"]
        segment_sep_id = self.tokenizer.segment_sep_id
        results = []

        # Filter out segments that are either empty or consist only of the "<|0.00|>" token
        for idx, sequence_segs in enumerate(sequences['segments']):
            sequences['segments'][idx] = [
                seg for seg in sequence_segs
                if len(seg['tokens']) > 0 and (len(seg['tokens']) != 1 or seg['tokens'][0] < timestamp_begin)
            ]

        def _is_same_30s_block(start_time, end_time):
            return (start_time // 30 == end_time // 30) or (
                (start_time // 30 + 1 == end_time // 30) and (end_time % 30 == 0)
            )

        # Iterate over each group of segments (e.g., one per utterance)
        for idx, sequence_segs in enumerate(sequences['segments']):
            if not sequence_segs:
                continue
            result = []

            # Sort segments by start time
            # This is necessary due to tokenization of timestamps
            sequence_segs = sorted(sequence_segs, key=lambda x: x['start_frames'])
            last_start_time = self.round_to_nearest_0_02(sequence_segs[-1]['start'].item())

            groups = [[] for _ in range(int(last_start_time // 30 + 1))]
            for i, seg in enumerate(sequence_segs):
                # Round start and end times to nearest 0.02 seconds
                start_time = self.round_to_nearest_0_02(seg['start'].item())
                end_time = self.round_to_nearest_0_02(seg['end'].item())
                group_idx = int(start_time // 30)
                new_start_time = start_time % 30
                new_end_time = end_time % 30
                tokens = seg['tokens']
                speaker = (tokens[0] - timestamp_begin) // timestamp_tokens_len()

                if start_time < end_time and new_end_time == 0:
                    new_end_time = 30
                groups[group_idx].append((new_start_time, tokens, new_end_time, speaker))

            for group in groups:
                for seg in group:
                    result.append(seg)
                result.append((0, [segment_sep_id], 30, 0))

            # Convert result segments into a token sequence with proper timestamp formatting
            encoded = self.tokenizer(
                "".join([
                    f"<|{seg[0]:.2f}|>{self.tokenizer.decode(seg[1])}<|{seg[2]:.2f}|>" if seg[3] == 0 else
                    f"<|{seg[0]:.2f}_spk{seg[3]}|>{self.tokenizer.decode(seg[1])}<|{seg[2]:.2f}_spk{seg[3]}|>"
                    for seg in result
                ]), add_special_tokens=False,
            )['input_ids']
            results.append(encoded)

        # Pad all sequences to the same length for batching
        sequences = pad_sequence(
            [torch.tensor(res, device=sequences['sequences'].device) for res in results],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        return sequences
