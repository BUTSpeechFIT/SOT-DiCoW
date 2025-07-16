# Author: Martin Kocour (BUT)

import torch
from transformers.generation.logits_process import LogitsProcessor
from models.dicow.utils import WhisperTimeStampLogitsProcessorCustom
from models.sot_dicow.utils import timestamp_tokens_len
from models.sot_dicow.config import SOTStyle


class WhisperSOTExplicitSpeakerTimeStampLogitsProcessor(WhisperTimeStampLogitsProcessorCustom):
    """Logits processor for handling timestamp tokens in multi-speaker transcription with explicit speaker tokens."""
    def __init__(self, *args, mt_num_speakers=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.mt_num_speakers = mt_num_speakers
        self.timestamp_end = self.timestamp_begin + timestamp_tokens_len() - 1
        self.speaker_token_begin = self.timestamp_end + 1

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # suppress <|notimestamps|> which is handled by without_timestamps
        scores_processed = scores.clone()
        scores_processed[:, self.no_timestamps_token_id] = -float("inf")

        # timestamps have to appear in pairs, except directly before eos_token; mask logits accordingly
        # speaker token has to precede timestamp token
        # speaker token has to be followed by timestamp token or speaker token
        for k in range(input_ids.shape[0]):
            sampled_tokens = input_ids[k, self.begin_index:]
            seq = list(sampled_tokens.tolist())

            last_was_timestamp = len(seq) >= 1 and seq[-1] >= self.timestamp_begin and seq[-1] <= self.timestamp_end
            penultimate_was_timestamp = len(seq) < 2 or (seq[-2] >= self.timestamp_begin and seq[-2] <= self.timestamp_end)
            last_was_speaker = len(seq) >= 1 and seq[-1] >= self.speaker_token_begin
            penultimate_was_speaker = len(seq) >= 2 and (seq[-2] >= self.speaker_token_begin)

            if last_was_timestamp:
                if penultimate_was_speaker: # next has to be non-speaker and non-timestamp
                    scores_processed[k, self.timestamp_begin:] = -float("inf")
                else:  # next has to be speaker (or eos)
                    scores_processed[k, self.timestamp_begin:self.speaker_token_begin] = -float("inf")
                    scores_processed[k, : self.eos_token_id] = -float("inf")
            elif last_was_speaker: # next has to be timestamp (or eos)
                scores_processed[k, :self.eos_token_id] = -float("inf")
                scores_processed[k, self.speaker_token_begin:] = -float("inf")
            elif len(seq) > 2: # last was regular token
                # next has to be timestamp or regular token
                scores_processed[k, self.speaker_token_begin:] = -float("inf")

            timestamps = self._timestamps_for_last_speaker(input_ids[k]) # type: ignore
            if timestamps.numel() > 0:
                # `timestamps` shouldn't decrease; forbid timestamp tokens smaller than the last
                # The following lines of code are copied from: https://github.com/openai/whisper/pull/914/files#r1137085090
                if last_was_timestamp and not penultimate_was_timestamp:
                    timestamp_last = timestamps[-1]
                else:
                    # Avoid to emit <|0.00|> again
                    timestamp_last = timestamps[-1] + 1

                # Mask out all timestamps up to the last one (including the last)
                scores_processed[k, self.timestamp_begin: timestamp_last] = -float("inf")

        if input_ids.shape[1] == self.begin_index:
            # no tokens have been sampled yet, so we can allow only speaker tokens
            eos_scores = scores_processed[:, self.eos_token_id].clone()
            scores_processed[:, :self.speaker_token_begin] = -float("inf")
            scores_processed[:, self.eos_token_id] = eos_scores
        elif input_ids.shape[1] == self.begin_index + 1:
            # apply the `max_initial_timestamp` option
            eos_scores = scores_processed[:, self.eos_token_id].clone()
            scores_processed[:, : self.timestamp_begin] = -float("inf")
            scores_processed[:, self.eos_token_id] = eos_scores

            if self.max_initial_timestamp_index is not None:
                last_allowed = self.timestamp_begin + self.max_initial_timestamp_index
                scores_processed[:, last_allowed + 1:self.speaker_token_begin] = -float("inf")
            if self.min_initial_timestamp_index is not None:
                first_allowed = self.timestamp_begin + self.min_initial_timestamp_index
                scores_processed[:, self.timestamp_begin:first_allowed] = -float("inf")

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = torch.nn.functional.log_softmax(scores_processed.float(), dim=-1)
        for k in range(input_ids.shape[0]):
            timestamp_logprob = logprobs[k, self.timestamp_begin:].logsumexp(dim=-1)
            max_text_token_logprob = logprobs[k, : self.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob and self._detect_timestamp_from_logprob:
                scores_processed[k, : self.timestamp_begin] = -float("inf")

        return scores_processed # type: ignore


    def _timestamps_for_last_speaker(self, input_ids: torch.LongTensor):
        """Get timestamps for the last speaker"""

        def default_timestamps():
            # return torch.empty(0, dtype=input_ids.dtype, device=input_ids.device)
            if input_ids.numel() > 0:
                return input_ids[(input_ids >= self.timestamp_begin) & (input_ids <= self.timestamp_end)]
            return torch.empty(0, dtype=input_ids.dtype, device=input_ids.device)

        # Find all speaker tokens
        speakers = input_ids[input_ids >= self.speaker_token_begin]
        if speakers.numel() == 0:
            return default_timestamps()

        last_speaker = speakers[-1]
        speaker_positions = torch.where(input_ids == last_speaker)[0]

        timestamps = []
        for spk_pos in speaker_positions:
            # Look for timestamps after this speaker token until the next speaker token or end
            after_speaker = input_ids[spk_pos + 1:]

            # Find the next speaker token position (if any)
            next_speaker_mask = after_speaker >= self.speaker_token_begin
            if next_speaker_mask.any():
                next_speaker_pos = torch.where(next_speaker_mask)[0][0]
                # Only look at tokens between current speaker and next speaker
                tokens_to_check = after_speaker[:next_speaker_pos]
            else:
                # No next speaker, check all remaining tokens
                tokens_to_check = after_speaker

            # Find timestamps in this range
            timestamp_mask = (tokens_to_check >= self.timestamp_begin) & (tokens_to_check <= self.timestamp_end)
            if timestamp_mask.any():
                speaker_timestamps = tokens_to_check[timestamp_mask]
                timestamps.extend(speaker_timestamps.tolist())

        if timestamps:
            return torch.tensor(timestamps, dtype=input_ids.dtype, device=input_ids.device)
        else:
            return torch.empty(0, dtype=input_ids.dtype, device=input_ids.device)


class ExplicitSpeakerLogitsProcessor(LogitsProcessor):
    """Logits processor for handling speaker tokens in multi-speaker transcription with explicit speaker tokens."""
    def __init__(self, generation_config, begin_index=1, mt_num_speakers=1):
        super().__init__()
        self.mt_num_speakers = mt_num_speakers
        self.begin_index = begin_index
        self.eos_token_id = generation_config.eos_token_id
        self.speaker_token_begin = generation_config.no_timestamps_token_id + 1 + timestamp_tokens_len()
        self.sot_style = SOTStyle(generation_config.mt_sot_style)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stno_mask: torch.Tensor) -> torch.FloatTensor: # type: ignore
        return scores