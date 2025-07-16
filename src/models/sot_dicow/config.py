# Author: Martin Kocour (BUT)

from enum import Enum
from models.dicow.config import DiCoWConfig

SOT_AGGR_TYPES = ["mean", "sum", "none", "concat"]

class SOTAggregationType(Enum):
    MEAN = "mean"
    SUM = "sum"
    NONE = "none"
    CONCAT = "concat"

class SOTStyle(Enum):
    UTTERANCE = "utterance"
    SPEAKER = "speaker"


class SOTDiCoWConfig(DiCoWConfig):
    """This is a modified version of DiCoW Config for Serialized Output Training (SOT)"""
    def __init__(
            self,
            mt_num_speakers: int = 1,
            mt_sot_aggregate_speakers: bool = False,
            mt_sot_aggregation_type: str | None = None,
            mt_sot_encoder: bool = False,
            mt_sot_transform_speakers: bool = False,
            mt_sot_use_sad: bool = False, # speaker activity detection
            mt_sot_spk_mask_inv_temp: float | None = None,
            mt_sot_speaker_loss_weight: float = 0.0,
            mt_sot_speaker_attn_weight: float = 0.0,
            mt_sot_explicit_speaker_tokens: bool = False,
            mt_sot_style: SOTStyle = SOTStyle.UTTERANCE,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.mt_num_speakers = mt_num_speakers
        self.mt_sot_aggregate_speakers = mt_sot_aggregate_speakers
        self.mt_sot_aggregation_type = mt_sot_aggregation_type
        self.mt_sot_transform_speakers = mt_sot_transform_speakers
        self.mt_sot_use_sad = mt_sot_use_sad
        self.mt_sot_spk_mask_inv_temp = mt_sot_spk_mask_inv_temp
        self.mt_sot_encoder = mt_sot_encoder
        self.mt_sot_speaker_loss_weight = mt_sot_speaker_loss_weight
        self.mt_sot_speaker_attn_weight = mt_sot_speaker_attn_weight
        self.mt_sot_explicit_speaker_tokens = mt_sot_explicit_speaker_tokens
        self.mt_sot_style = mt_sot_style