# Author: Martin Kocour (BUT)

import re
import torch
from transformers.models.whisper import WhisperTokenizerFast
from models.sot_dicow.utils import timestamp_tokens_len


class WhisperTokenizerForSOT(WhisperTokenizerFast):
    """Tokenizer for serialized output training with multi-speaker transcription."""
    def __init__(self, *args, mt_num_speakers=1, predict_timestamps=True,
        segment_sep_token="<|new_segment|>", **kwargs):
        if not predict_timestamps:
            raise NotImplementedError("Seriliazed output training works only with timestamps tokens!")
        super().__init__(*args, predict_timestamps=True, **kwargs)
        # ignoring the first
        new_tokens = [f"<|{ts:.2f}_spk{sidx}|>" for sidx in range(1, mt_num_speakers)
                      for ts in torch.arange(0, 30.02, 0.02)]
        self.add_tokens(new_tokens, special_tokens=False)
        self.add_tokens(segment_sep_token, special_tokens=True)
        self.timestamp_pat = re.compile(r"<\|(\d+\.\d+)(_spk\d+)?\|>")
        self.mt_num_speakers = mt_num_speakers
        self.segment_sep_token = segment_sep_token

    @property
    def segment_sep_id(self):
        """Get the token ID for the segment separator."""
        return self.convert_tokens_to_ids(self.segment_sep_token)

    def _decode_with_timestamps(self, token_ids, skip_special_tokens=False, time_precision=0.02) -> str:
        """
        Timestamp tokens are above the special tokens' id range and are ignored by `decode()`. This method decodes
        given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        timestamp_begin = self.all_special_ids[-1] + 1
        segment_sep_id = self.segment_sep_id
        timestamp_tokens = torch.tensor(token_ids).ge(timestamp_begin) & torch.tensor(token_ids).lt(segment_sep_id)
        if not timestamp_tokens.any():
            return super()._decode_with_timestamps(token_ids, skip_special_tokens=skip_special_tokens)

        outputs = []
        timestamp_indices = torch.where(timestamp_tokens)[0]
        first_timestamp_idx = timestamp_indices[0]

        if first_timestamp_idx > 0:
            outputs.append(token_ids[:first_timestamp_idx])

        last_timestamp_idx = timestamp_indices[-1]
        last_output = []
        if last_timestamp_idx < len(token_ids) - 1:
            if len(timestamp_indices) % 2 != 0:
                last_output.append(int(token_ids[last_timestamp_idx]))
                last_timestamp_idx -= 1
            last_output.append(token_ids[last_timestamp_idx + 1:])

        prev_segments_len = 0.0
        cur_max_timestamp_pos = 0

        for idx in range(0, len(timestamp_indices), 2):
            if idx + 1 >= len(timestamp_indices):
                break
            sliced_tokens = token_ids[timestamp_indices[idx]:timestamp_indices[idx + 1] + 1]
            start_timestamp_pos = (sliced_tokens[0].item() - timestamp_begin) % timestamp_tokens_len()
            end_timestamp_pos = (sliced_tokens[-1].item() - timestamp_begin) % timestamp_tokens_len()
            start_timestamp = float(start_timestamp_pos * time_precision) + prev_segments_len
            end_timestamp = float(end_timestamp_pos * time_precision) + prev_segments_len
            speaker = (sliced_tokens[0].item() - timestamp_begin) // timestamp_tokens_len()

            if len(sliced_tokens) == 3 and sliced_tokens[1] == segment_sep_id:
                prev_segments_len += 30.0
                continue

            # Adjust timestamps if needed
            if start_timestamp_pos > end_timestamp_pos:
                end_timestamp += 30.0

            # Append formatted output
            outputs.append(f"<|{start_timestamp:.2f}|><|spk{speaker}|>")
            outputs.append(sliced_tokens[1:-1] if len(sliced_tokens) > 2 else [])
            outputs.append(f"<|{end_timestamp:.2f}|><|spk{speaker}|>")
            cur_max_timestamp_pos = start_timestamp_pos # we know that segments are sorted by start timestamp

        if len(last_output) > 0 and isinstance(last_output[0], int):
            start_timestamp_pos = (last_output[0] - timestamp_begin) % timestamp_tokens_len()
            prev_segments_len += 30.0 if cur_max_timestamp_pos > start_timestamp_pos else 0.0
            start_timestamp = float(start_timestamp_pos * time_precision) + prev_segments_len
            speaker = (last_output[0] - timestamp_begin) // timestamp_tokens_len()
            last_output[0] = f"<|{start_timestamp:.2f}|><|spk{speaker}|>"
        outputs += last_output
        outputs = [
            s if isinstance(s, str) else self.decode(s, skip_special_tokens=skip_special_tokens) for s in outputs
        ]
        return "".join(outputs)

    def timestamp_ids(self, time_precision=0.02):
        orig_timestamps = [(f"<|{i * time_precision:.2f}|>") for i in range(1500 + 1)]
        new_timestamps = [(f"<|{i * time_precision:.2f}_spk{s}|>") for i in range(1500 + 1)
                          for s in range(1, self.mt_num_speakers)]
        return self.convert_tokens_to_ids(orig_timestamps + new_timestamps)
