# Author: Martin Kocour (BUT)

import torch
from torch.nn.utils.rnn import pad_sequence

from transformers.utils import logging
from transformers.generation.logits_process import SuppressTokensLogitsProcessor, SuppressTokensAtBeginLogitsProcessor, WhisperNoSpeechDetection

from models.sot_dicow.generation import SOTDiCoWGenerationMixin
from models.sot_dicow.utils import timestamp_tokens_len
from models.explicit_sot_dicow.logits_process import WhisperSOTExplicitSpeakerTimeStampLogitsProcessor, ExplicitSpeakerLogitsProcessor


logger = logging.get_logger("explicit_sot_dicow")


class ExplicitSOTDiCoWGenerationMixin(SOTDiCoWGenerationMixin):
    def _retrieve_logit_processors(self, generation_config, logits_processor, begin_index, is_shortform, num_beams, device):
        if generation_config.return_timestamps is True:
            timestamp_processor = WhisperSOTExplicitSpeakerTimeStampLogitsProcessor(
                generation_config,
                begin_index=begin_index,
                mt_num_speakers=self.config.mt_num_speakers,
            )
            logits_processor = (
                [timestamp_processor] if logits_processor is None else [timestamp_processor] + logits_processor
            )

            speaker_processor = ExplicitSpeakerLogitsProcessor(
                generation_config,
                begin_index=begin_index,
                mt_num_speakers=self.config.mt_num_speakers,
            )
            logits_processor.append(speaker_processor)

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

    def _postprocess_outputs(self, seek_outputs, decoder_input_ids, return_token_timestamps, generation_config):
        seek_outputs, seek_sequence = super()._postprocess_outputs(seek_outputs, decoder_input_ids, return_token_timestamps, generation_config)
        if isinstance(seek_outputs, torch.Tensor):
            # In this block of code we need to remove trailing speaker tokens
            # This is because the trailing speaker tokens are unwanted, and we want to replace them with pad tokens
            # We do this by checking if the speaker token is followed by only EOS or pad tokens
            # If it is, we replace the speaker token with a pad token

            # First, we need to find the speaker token positions
            speaker_token_begin = generation_config.no_timestamps_token_id + 1 + timestamp_tokens_len()
            speaker_mask = seek_outputs.ge(speaker_token_begin)

            if speaker_mask.any():
                # Create a mask for tokens that are EOS or padding
                eos_or_pad_mask = (seek_outputs == generation_config.eos_token_id) | (seek_outputs == generation_config.pad_token_id)

                # Create a mask that identifies positions where all subsequent tokens are EOS or padding
                # We'll work backwards from the end of each sequence using cumprod
                # Flip the mask and use cumprod to find positions where all subsequent tokens are EOS/pad
                # cumprod(flip) gives us a mask where True means all tokens from this position to the end are EOS/pad
                flipped_eos_pad = torch.flip(eos_or_pad_mask, dims=[1])
                cumulative_eos_pad = torch.cumprod(flipped_eos_pad, dim=1)
                cumulative_eos_pad = torch.flip(cumulative_eos_pad, dims=[1])

                # Shift left by 1 to check if all tokens AFTER the current position are EOS/pad
                # This way we don't include the current position in the check
                cumulative_eos_pad = torch.cat([cumulative_eos_pad[:, 1:], torch.zeros_like(cumulative_eos_pad[:, :1])], dim=1)

                # Now find speaker tokens that are followed only by EOS/pad tokens
                # A speaker token should be replaced if all tokens from this position onwards are EOS/pad
                # We want to check if the current speaker position has cumulative_eos_pad=True
                replace_mask = speaker_mask & cumulative_eos_pad

                # Only replace if there are actually tokens to replace
                if replace_mask.any():
                    # Replace speaker tokens with pad_token_id where replace_mask is True
                    seek_outputs[replace_mask.bool()] = generation_config.pad_token_id

            return seek_outputs, seek_outputs
        return seek_outputs, seek_sequence

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
        timestamp_end = timestamp_begin + timestamp_tokens_len() - 1
        speaker_token_begin = timestamp_begin + timestamp_tokens_len()

        timestamp_tokens: torch.Tensor = seek_sequence.ge(timestamp_begin) & seek_sequence.le(timestamp_end)
        spk_timestamp_tokens: torch.Tensor = seek_sequence.ge(timestamp_begin)
        spk_tokens: torch.Tensor = seek_sequence.ge(speaker_token_begin)
        single_timestamp_ending = spk_timestamp_tokens[-2:].tolist() == [False, True] and timestamp_tokens[-2:].tolist() == [False, True]

        segment_indices = torch.where(spk_tokens[1:])[0]
        segment_indices.add_(1)
        token_timestamps = seek_outputs[idx]["token_timestamps"] if return_token_timestamps else []
        frame_offset = (time_offset / time_precision * input_stride).to(seek_num_frames.dtype).tolist()

        # If whisper predicted a "end of segment" via a timestep token, let's go ever each
        # "end of segment" prediction and slice the decoding into segments accordingly
        if len(segment_indices) > 0:
            # if the output contains two consecutive timestamp tokens
            slices = segment_indices.tolist()
            segments = []
            if single_timestamp_ending:
                slices.append(len(seek_sequence))

            last_slice = 0
            # Add each segment to list of all segments
            for i, current_slice in enumerate(slices):
                sliced_tokens = seek_sequence[last_slice:current_slice]
                spk_id = sliced_tokens[0].item() - speaker_token_begin
                start_timestamp_pos = sliced_tokens[1].item() - timestamp_begin
                end_timestamp_pos = sliced_tokens[-1].item() - timestamp_begin
                segments.append(
                    {
                        "start_frames": frame_offset[prev_idx] + start_timestamp_pos * input_stride,
                        "end_frames": frame_offset[prev_idx] + end_timestamp_pos * input_stride,
                        "start": time_offset[prev_idx] + start_timestamp_pos * time_precision,
                        "end": time_offset[prev_idx] + end_timestamp_pos * time_precision,
                        "tokens": sliced_tokens,
                        "result": seek_outputs[idx],
                        "channel": spk_id,
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
                last_timestamp_pos = seek_sequence[last_slice - 1].item() - timestamp_begin
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
                start_timestamp_pos = timestamps[-2].item() - timestamp_begin
                last_timestamp_pos = timestamps[-1].item() - timestamp_begin
            elif timestamps.numel() == 1:
                # no consecutive timestamps but it has a timestamp; use the last one.
                start_timestamp_pos = timestamps[-1].item() - timestamp_begin
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
                        "channel": seek_sequence[0].item() - speaker_token_begin,
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
                # Type annotation to handle the assignment properly
                self._stno_mask_seek_skip[prev_idx] = skip_segments  # type: ignore
            else:
                # there is no overlapping segments, so we can just rollback to the offset time
                self._stno_mask_seek_skip[prev_idx] = None

        if segment_offset <= 0:
            msg = f"Segments: {segments}"
            logger.warning("Segment offset: %s <= 0. This should not happen!\n%s", segment_offset, msg)
            segment_offset = seek_num_frames[prev_idx]

        return segments, segment_offset

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

        # Iterate over each group of segments (e.g., one per utterance)
        for idx, sequence_segs in enumerate(sequences['segments']):
            if not sequence_segs:
                results.append([])
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
                speaker = seg['channel']

                if start_time < end_time and new_end_time == 0:
                    new_end_time = 30
                groups[group_idx].append((new_start_time, tokens, new_end_time, speaker))

            for group in groups:
                for seg in group:
                    result.append(seg)
                result.append(segment_sep_id)

            # Convert result segments into a token sequence with proper timestamp formatting
            encoded = self.tokenizer(
                "".join([
                    f"<|spk{seg[3]}|><|{seg[0]:.2f}|>{self.tokenizer.decode(seg[1], decode_with_speakers=False)}<|{seg[2]:.2f}|>" if isinstance(seg, tuple) else self.tokenizer.decode(seg)
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
