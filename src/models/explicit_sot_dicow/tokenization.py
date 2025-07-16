# Author: Martin Kocour (BUT)

import re
import torch

from transformers.models.whisper import WhisperTokenizerFast
from models.sot_dicow.utils import timestamp_tokens_len


class WhisperTokenizerForExplicitSOT(WhisperTokenizerFast):
    """Tokenizer for serialized output training with explicit speaker tokens."""
    def __init__(self, *args, mt_num_speakers=1, predict_timestamps=True, segment_sep_token="<|new_segment|>", **kwargs):
        # TODO: This tokenizer produces correct transcripts only for SOT style: "speaker"
        super().__init__(*args, predict_timestamps=True, **kwargs)
        if not predict_timestamps:
            raise NotImplementedError("Seriliazed output training works only with timestamps tokens!")

        self.mt_num_speakers = mt_num_speakers
        self.add_tokens([f"<|spk{sidx}|>" for sidx in range(mt_num_speakers)])
        self.add_tokens(segment_sep_token, special_tokens=True)
        self.segment_sep_token = segment_sep_token
        self.speaker_pat = re.compile(r"<\|spk\d+\|>")

    @property
    def segment_sep_id(self):
        """Get the token ID for the segment separator."""
        return self.convert_tokens_to_ids(self.segment_sep_token)

    def decode(self, token_ids, *args, decode_with_speakers=True, **kwargs):
        text = super().decode(token_ids, *args, **kwargs)
        if not decode_with_speakers:
            text = self._filter_speaker_tokens(text)
        return text

    def _filter_speaker_tokens(self, text):
        return re.sub(self.speaker_pat, "", text)

    def _decode_with_timestamps(self, token_ids, skip_special_tokens=False, time_precision=0.02) -> str:
        """
        Timestamp tokens are above the special tokens' id range and are ignored by `decode()`. This method decodes
        given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids)

        is_longform = token_ids.eq(self.segment_sep_id).any()
        if is_longform:
            return self._decode_longform(token_ids, skip_special_tokens=skip_special_tokens, time_precision=time_precision)
        else:
            return self._decode_segment(token_ids, skip_special_tokens=skip_special_tokens, time_precision=time_precision)

    def _decode_longform(self, token_ids, skip_special_tokens=False, time_precision=0.02) -> str:
        """
        Decode longform tokens with timestamps tokens and speaker tokens.
        """
        # token_ids contains segments separated by <|new_segment|>
        # each segment contains speaker tokens and timestamps tokens
        # we need to decode each segment separately
        # and then join them together

        segment_idxs = token_ids.eq(self.segment_sep_id).nonzero(as_tuple=True)[0]
        outputs = []
        time_offset = 0
        prev_idx = 0
        for idx in segment_idxs:
            segment = token_ids[prev_idx:idx]
            outputs.append(self._decode_segment(segment, skip_special_tokens=skip_special_tokens, time_precision=time_precision, time_offset=time_offset))
            time_offset += (timestamp_tokens_len() - 1) * time_precision
            prev_idx = idx + 1
        last_segment = token_ids[prev_idx:]
        outputs.append(self._decode_segment(last_segment, skip_special_tokens=skip_special_tokens, time_precision=time_precision, time_offset=time_offset))
        return "".join(outputs)

    def _decode_segment(self, token_ids: torch.Tensor, skip_special_tokens=False, time_precision=0.02, time_offset=0) -> str:
        """
        Decode a segment with timestamps tokens and speaker tokens.

        Timestamp tokens are above the special tokens' id range and are ignored by `decode()`. This method decodes
        given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        timestamp_begin = self.all_special_ids[-1] + 1
        spk_begin = timestamp_begin + timestamp_tokens_len()
        segment_offset = (timestamp_tokens_len() - 1) * time_precision
        outputs = [[]]

        last_start_time = 0.0
        is_start = True

        for token in token_ids.tolist():
            if token >= timestamp_begin and token < spk_begin:
                timestamp = float((token - timestamp_begin)) * time_precision + time_offset

                if timestamp < last_start_time and not is_start:
                    timestamp += segment_offset

                if is_start:
                    last_start_time = timestamp

                is_start = not is_start

                outputs.append(f"<|{timestamp:.2f}|>")
                outputs.append([])
            elif token >= spk_begin:
                is_start = True
                outputs.append(f"<|spk{token - spk_begin}|>")
                outputs.append([])
            else:
                outputs[-1].append(token)
        outputs = [
            s if isinstance(s, str) else self.decode(s, skip_special_tokens=skip_special_tokens, decode_with_speakers=False, decode_with_timestamps=False) for s in outputs
        ]
        return "".join(outputs)
