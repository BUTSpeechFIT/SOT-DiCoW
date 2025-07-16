# Author: Martin Kocour (BUT)

import torch
import torch.nn.functional as F
from models.dicow.utils import WhisperTimeStampLogitsProcessorCustom

def timestamp_tokens_len(mt_num_speakers=1):
    """Calculate total number of timestamp tokens for given number of speakers.

    Args:
        mt_num_speakers: Number of speakers to generate timestamps for.
    Returns:
        Total number of timestamp tokens.
    """
    ts_len = 30 * 50 + 1  # i.e. range(0, 1500 incl)
    return mt_num_speakers * ts_len


def speaker_groups(per_group_sizes, *tensors):
    """Split batch according to the speaker groups"""
    offset = 0
    for group_size in per_group_sizes:
        if group_size == 0:  # when decoding, one of the groups can be already finished, so we skip it
            continue
        # Check that the group size doesn't exceed the remaining elements in the tensors
        if offset + group_size > len(tensors[0]):
            raise ValueError("Group size exceeds the remaining number of elements in the tensors.")

        # Yield group_size followed by the sliced tensors as a tuple
        yield (group_size, *[t[offset:offset + group_size] for t in tensors])
        offset += group_size


def handle_different_batch_size(input_tensor: torch.Tensor, ref_tensor: torch.Tensor, per_group_sizes: torch.Tensor):
    """Adjust input tensor to same batch size"""
    # This is the hack for GenerationMixin.generate() since batch_size is computed from input_features
    # and we don't won't the batch size to be a factor of num_speakers in each group
    # TODO: this might be a problem during Beam Search, since per_group_sizes might differ
    if input_tensor.shape[0] != per_group_sizes.shape[0]:
        raise ValueError(
            f"Expected input_features ({input_tensor.shape[0]}) to have same batch "
            f"size as per_group_sizes ({per_group_sizes.shape[0]})"
        )
    if per_group_sizes.sum() != ref_tensor.shape[0]:
        raise ValueError(
            f"Expected sum of per_group_sizes ({per_group_sizes.sum()}) "
            f"to be equal stno_mask batch size ({ref_tensor.shape[0]})"
        )
    return torch.repeat_interleave(input_tensor, per_group_sizes, dim=0)


def remove_duplicates(x: torch.Tensor, per_group_sizes: torch.Tensor, keep_dim=True, ignore_index=-100):
    """Find duplicates across specified dim"""
    if per_group_sizes.shape[0] == x.shape[0]:
        return x
    cum_pgs = per_group_sizes.cumsum(0)
    nzi = cum_pgs - per_group_sizes
    if not keep_dim:
        return torch.index_select(x, dim=0, index=nzi)
    y = torch.full_like(x, fill_value=ignore_index)
    y[nzi] = x[nzi]
    return y


def convert_to_spk_mask(fddt_mask: torch.Tensor):
    """Convert segment input slice to speaker mask"""
    spk_mask = fddt_mask[:, [1,3], :].sum(1)
    return spk_mask


def convert_to_fddt_mask(spk_mask: torch.Tensor):
    """Convert speaker mask to fddt mask"""
    S, T = spk_mask.shape
    fddt_mask = torch.zeros((S, 4, T), dtype=spk_mask.dtype, device=spk_mask.device)
    for s_idx in range(S):
        non_target_mask = torch.ones(S, dtype=torch.bool)
        non_target_mask[s_idx] = False
        sil_frames = (1 - spk_mask).prod(dim=0)
        anyone_else = (1 - spk_mask[non_target_mask]).prod(dim=0)
        target_spk = spk_mask[s_idx] * anyone_else
        non_target_spk = (1 - spk_mask[s_idx]) * (1 - anyone_else)
        overlapping_speech = spk_mask[s_idx] - target_spk
        stno_mask = torch.stack([sil_frames, target_spk, non_target_spk, overlapping_speech], dim=0)
        fddt_mask[s_idx] = stno_mask
    return fddt_mask


def convert_to_spk_info(decoder_input_ids: torch.Tensor, first_spk_idx: int, num_speakers: int, explicit_speaker_tokens: bool = False):
    """Convert input tensor to speaker info"""
    spk_info = torch.full_like(decoder_input_ids, fill_value=-1)
    spk_mask = decoder_input_ids >= first_spk_idx
    spk_idx = (decoder_input_ids[spk_mask] - first_spk_idx)
    if not explicit_speaker_tokens:
        spk_idx = spk_idx // timestamp_tokens_len()
    spk_info[spk_mask] = spk_idx

    idx = torch.arange(spk_info.size(1), device=spk_info.device).expand_as(spk_info)
    idx_masked = torch.where(spk_mask, idx, torch.full_like(idx, -1))
    last_valid_idx, _ = idx_masked.cummax(dim=1)
    spk_info = spk_info.gather(1, last_valid_idx.clamp(min=0))
    spk_info = spk_info.where(last_valid_idx.ge(0), torch.full_like(spk_info, -1))

    valid_mask = (spk_info >= 0) & (spk_info < num_speakers)
    spk_info[~valid_mask] = 0
    spk_encoding = F.one_hot(spk_info, num_classes=num_speakers).float()
    spk_encoding[~valid_mask] = 0
    return spk_encoding


class WhisperSOTTimeStampLogitsProcessor(WhisperTimeStampLogitsProcessorCustom):
    """Logits processor for handling timestamp tokens in multi-speaker transcription."""
    def __init__(self, *args, mt_num_speakers=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.mt_num_speakers = mt_num_speakers

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # suppress <|notimestamps|> which is handled by without_timestamps
        scores_processed = scores.clone()
        scores_processed[:, self.no_timestamps_token_id] = -float("inf")

        # timestamps have to appear in pairs, except directly before eos_token; mask logits accordingly
        for k in range(input_ids.shape[0]):
            sampled_tokens = input_ids[k, self.begin_index:]
            seq = list(sampled_tokens.tolist())

            last_was_timestamp = len(seq) >= 1 and seq[-1] >= self.timestamp_begin
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= self.timestamp_begin

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    scores_processed[k, self.timestamp_begin:] = -float("inf")
                else:  # cannot be normal text tokens
                    scores_processed[k, : self.eos_token_id] = -float("inf")

            timestamps = sampled_tokens[sampled_tokens.ge(self.timestamp_begin)]
            if timestamps.numel() > 0:
                if last_was_timestamp and not penultimate_was_timestamp:
                    # Next timestamp can be from any speaker
                    # we need to know what was the last TS for each speaker and mask every lower TS
                    for s in range(self.mt_num_speakers):
                        sts_begin = self.timestamp_begin + timestamp_tokens_len(s)
                        sts_end = sts_begin + timestamp_tokens_len() - 1
                        sts = timestamps[(timestamps >= sts_begin) & (timestamps <= sts_end)]
                        if sts.numel() == 0:
                            # no speaker timestamp, anything is possible
                            continue
                        # Mask all timestamps up to the last one (including the last)
                        sts_last = sts[-1]
                        scores_processed[k, sts_begin:sts_last + 1] = -float("inf")
                else:
                    sts_last = timestamps[-1]
                    speaker_last = (sts_last - self.timestamp_begin) // timestamp_tokens_len()
                    # Mask out all lower TS and lower speakers
                    # Avoid to emit same TS again
                    scores_processed[k, self.timestamp_begin: sts_last + 1] = -float("inf")
                    # Mask out all TS for higher speakers
                    nsts_begin = self.timestamp_begin + timestamp_tokens_len(speaker_last + 1)
                    scores_processed[k, nsts_begin:] = -float("inf")

        # apply the `max_initial_timestamp` option
        if input_ids.shape[1] == self.begin_index:
            eos_scores = scores_processed[:, self.eos_token_id].clone()
            scores_processed[:, : self.timestamp_begin] = -float("inf")
            scores_processed[:, self.eos_token_id] = eos_scores

            if self.max_initial_timestamp_index is not None:
                for s in range(self.mt_num_speakers):
                    sts_begin = self.timestamp_begin + timestamp_tokens_len(s)
                    sts_end = sts_begin + timestamp_tokens_len() - 1
                    last_allowed = sts_begin + self.max_initial_timestamp_index
                    scores_processed[:, last_allowed + 1: sts_end] = -float("inf")
            if self.min_initial_timestamp_index is not None:
                for s in range(self.mt_num_speakers):
                    sts_begin = self.timestamp_begin + timestamp_tokens_len(s)
                    first_allowed = sts_begin + self.min_initial_timestamp_index
                    scores_processed[:, sts_begin:first_allowed] = -float("inf")

        # if sum of probability over timestamps is above any other token, sample timestamp
        if self._detect_timestamp_from_logprob:
            logprobs = torch.nn.functional.log_softmax(scores_processed.float(), dim=-1)
            for k in range(input_ids.shape[0]):
                timestamp_logprob = logprobs[k, self.timestamp_begin:].logsumexp(dim=-1)
                max_text_token_logprob = logprobs[k, : self.timestamp_begin].max()
                if timestamp_logprob > max_text_token_logprob:
                    scores_processed[k, : self.timestamp_begin] = -float("inf")

        return scores_processed