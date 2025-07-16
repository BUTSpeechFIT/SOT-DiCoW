from typing import Optional, Tuple

import torch

import torch.nn as nn
import torch.nn.functional as F
from models.dicow.modeling_dicow import CustomLinear
from models.sot_dicow.config import SOTDiCoWConfig
from models.sot_dicow.utils import timestamp_tokens_len, convert_to_spk_info

from transformers.models.whisper.modeling_whisper import (
    WhisperAttention,
    WhisperFlashAttention2,
    WhisperDecoderLayer,
    WhisperDecoder,
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from models.sot_dicow.sot_layers import BranchLinear
from transformers.utils import logging

logging.set_verbosity_debug()
logger = logging.get_logger("sot_dicow")


class SOTDecoderAttention(WhisperAttention):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, is_decoder: bool = False, config: SOTDiCoWConfig = None):
        super().__init__(embed_dim, num_heads, dropout, is_decoder, config)
        self.num_speakers = config.mt_num_speakers
        if config.mt_sot_speaker_attn_weight > 0:
            self.speaker_info_q_proj = CustomLinear(self.num_speakers, self.num_speakers, init_eye_val=config.mt_sot_speaker_attn_weight)
            self.speaker_info_k_proj = CustomLinear(self.num_speakers, self.num_speakers, init_eye_val=config.mt_sot_speaker_attn_weight)
            self.speaker_info_v_proj = CustomLinear(self.num_speakers, self.num_speakers, init_eye_val=config.mt_sot_speaker_attn_weight)

    def _forward_spk_info(self, query_states, key_states, value_states, spk_info):
        if spk_info is not None and hasattr(self, "speaker_info_q_proj"):
            B = spk_info.shape[0]

            # Query
            valid_mask = spk_info.sum(dim=-1, keepdim=False) == 0
            spk_q_states = self.speaker_info_q_proj(spk_info)
            spk_q_states[valid_mask] = 0
            spk_q_states_expanded = spk_q_states.unsqueeze(1).expand(-1, self.num_heads, -1, -1).reshape(B * self.num_heads, -1, self.num_speakers)
            new_query_states = torch.cat((query_states, spk_q_states_expanded), dim=-1)

            # Key
            key_spk_info = torch.arange(key_states.shape[1], device=key_states.device) // 1500
            key_spk_encoding = F.one_hot(key_spk_info, num_classes=self.num_speakers).float()
            spk_k_states = self.speaker_info_k_proj(key_spk_encoding)
            spk_k_states_expanded = spk_k_states.unsqueeze(0).unsqueeze(1).expand(B, self.num_heads, -1, -1).reshape(B * self.num_heads, -1, self.num_speakers)
            new_key_states = torch.cat((key_states, spk_k_states_expanded), dim=-1)

            # Value
            spk_v_states = self.speaker_info_v_proj(key_spk_encoding)
            spk_v_states_expanded = spk_v_states.unsqueeze(0).unsqueeze(1).expand(B, self.num_heads, -1, -1).reshape(B * self.num_heads, -1, self.num_speakers)
            new_value_states = torch.cat((value_states, spk_v_states_expanded), dim=-1)
            return new_query_states, new_key_states, new_value_states
        return query_states, key_states, value_states

    def _extract_new_spk_info(self, attn_output, old_spk_info):
        if old_spk_info is not None and hasattr(self, "speaker_info_q_proj"):
            B = old_spk_info.shape[0]
            spk_info = attn_output[..., -self.num_speakers:]
            attn_output = attn_output[..., :-self.num_speakers]
            spk_info = spk_info.view(B, self.num_heads, -1, self.num_speakers)
            spk_info = spk_info.mean(dim=1)
            return attn_output, spk_info
        return attn_output, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        spk_info: Optional[torch.Tensor] = None, # This is new
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        query_states, key_states, value_states = self._forward_spk_info(query_states, key_states, value_states, spk_info) # This is new

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)
        attn_output, new_spk_info = self._extract_new_spk_info(attn_output, spk_info)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value, new_spk_info


class SOTDecoderFlashAttention(WhisperFlashAttention2):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, is_decoder: bool = False, config: SOTDiCoWConfig = None):
        super().__init__(embed_dim, num_heads, dropout, is_decoder, config)
        self.num_speakers = config.mt_num_speakers
        if config.mt_sot_speaker_attn_weight > 0:
            self.speaker_info_q_proj = CustomLinear(self.num_speakers, self.num_speakers, init_eye_val=config.mt_sot_speaker_attn_weight)
            self.speaker_info_k_proj = CustomLinear(self.num_speakers, self.num_speakers, init_eye_val=config.mt_sot_speaker_attn_weight)
            self.speaker_info_v_proj = CustomLinear(self.num_speakers, self.num_speakers, init_eye_val=config.mt_sot_speaker_attn_weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        spk_info: Optional[torch.Tensor] = None, # This is new
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # WhisperFlashAttention2 attention does not support output_attentions
        if output_attentions:
            raise ValueError("WhisperFlashAttention2 attention does not support output_attentions")

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, q_len, _ = hidden_states.size()

        # get query proj
        query_states = self._reshape(self.q_proj(hidden_states), -1, bsz)
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0].transpose(1, 2)
            value_states = past_key_value[1].transpose(1, 2)
        elif is_cross_attention:
            # cross_attentions
            key_states = self._reshape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._reshape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._reshape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._reshape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0].transpose(1, 2), key_states], dim=1)
            value_states = torch.cat([past_key_value[1].transpose(1, 2), value_states], dim=1)
        else:
            # self_attention
            key_states = self._reshape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._reshape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states.transpose(1, 2), value_states.transpose(1, 2))

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output, new_spk_info = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=self.dropout, spk_info=spk_info # This is new
        )

        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, new_spk_info

    def _flash_attention_forward(self, query_states, key_states, value_states, attention_mask, q_len, dropout=0.0, spk_info=None):
        if spk_info is not None and hasattr(self, "speaker_info_q_proj"):
            orig_hidden_dim = value_states.shape[-1]
            # Query
            invalid_mask = spk_info.sum(dim=-1, keepdim=False) == 0
            spk_q_states = self.speaker_info_q_proj(spk_info)
            spk_q_states[invalid_mask] = 0
            spk_q_states_expanded = spk_q_states.unsqueeze(2).expand(-1, -1, query_states.shape[2], -1)
            new_query_states = torch.cat((query_states, spk_q_states_expanded), dim=-1)

            # Key
            key_spk_info = torch.arange(key_states.shape[1], device=key_states.device) // 1500
            key_spk_encoding = F.one_hot(key_spk_info, num_classes=self.num_speakers).float()
            spk_k_states = self.speaker_info_k_proj(key_spk_encoding)
            spk_k_states_expanded = spk_k_states.unsqueeze(0).unsqueeze(2).expand(key_states.shape[0], -1, key_states.shape[2], -1)
            new_key_states = torch.cat((key_states, spk_k_states_expanded), dim=-1)

            # Value
            spk_v_states = self.speaker_info_v_proj(key_spk_encoding)
            spk_v_states_expanded = spk_v_states.unsqueeze(0).unsqueeze(2).expand(value_states.shape[0], -1, value_states.shape[2], -1)
            new_value_states = torch.cat((value_states, spk_v_states_expanded), dim=-1)

            attn_output = super()._flash_attention_forward(new_query_states, new_key_states, new_value_states, attention_mask, q_len, dropout)
            new_spk_info = attn_output[..., orig_hidden_dim:]
            attn_output = attn_output[..., :orig_hidden_dim]
            new_spk_info = new_spk_info.mean(dim=2)
            return attn_output, new_spk_info

        attn_output = super()._flash_attention_forward(query_states, key_states, value_states, attention_mask, q_len, dropout)
        return attn_output, None


SOT_ATTENTION_CLASSES = {
    "eager": SOTDecoderAttention,
    "flash_attention_2": SOTDecoderFlashAttention,
}


class SOTDecoderLayer(WhisperDecoderLayer):
    def __init__(self, config: SOTDiCoWConfig):
        super().__init__(config)
        self.encoder_attn = SOT_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        spk_info: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value, new_spk_info = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
                spk_info=spk_info, # This is new
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        if new_spk_info is not None:
            outputs += (new_spk_info,)

        return outputs


class WhisperSOTEmbedding(nn.Embedding):
    """Embedding layer that handles speaker-specific timestamp embeddings."""
    def __init__(self, num_embeddings: int, embedding_dim: int, *args, mt_num_speakers=1, **kwargs):
        super().__init__(num_embeddings, embedding_dim, *args, **kwargs)
        self.mt_num_speakers = mt_num_speakers
        self.spkr_ts_weight = BranchLinear(mt_num_speakers, embedding_dim)
        self.first_ts_idx = num_embeddings - timestamp_tokens_len()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.clone()
        ts_mask = input >= self.first_ts_idx
        ts_states = input[ts_mask] # e.g. 50365, 51865, 51866, etc
        spk_idx = (ts_states - self.first_ts_idx) // timestamp_tokens_len()  # e.g. 0, 0, 1
        ts_norm = (ts_states - self.first_ts_idx) %  timestamp_tokens_len()  # e.g. 0, 1500, 0
        input[ts_mask] = ts_norm + self.first_ts_idx  # e.g. 50365, 51865, 50365
        orig_emb = super().forward(input)  # B x T x H
        orig_emb[ts_mask] = self.spkr_ts_weight(orig_emb[ts_mask], spk_idx)
        return orig_emb


class SOTDiCoWDecoder(WhisperDecoder):
    config_class = SOTDiCoWConfig

    """Whisper decoder modified for multi-speaker transcription."""
    def __init__(self, config: SOTDiCoWConfig):
        super().__init__(config)
        if not config.mt_sot_explicit_speaker_tokens:
            # implicit speaker tokens
            self.embed_tokens = WhisperSOTEmbedding(
                config.vocab_size, config.d_model, self.padding_idx,
                mt_num_speakers=config.mt_num_speakers,
            )
        self.layers = nn.ModuleList([SOTDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.first_ts_idx = config.vocab_size - timestamp_tokens_len()
        self.num_speakers = config.mt_num_speakers

    def set_input_embeddings(self, value):
        raise NotImplementedError("Setting embeding is not supported")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        spk_info=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # This is new
        if spk_info is None:
            spk_info = convert_to_spk_info(input_ids, self.first_ts_idx, self.num_speakers, self.config.mt_sot_explicit_speaker_tokens)

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and head_mask is None and not output_attentions:
            # output_attentions=True & head_mask can not be supported when using SDPA.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        # embed positions
        if input_ids is not None:
            positions = self.embed_positions(
                input_ids, past_key_values_length=past_key_values_length, position_ids=position_ids
            )
        else:
            positions = self.embed_positions(
                inputs_embeds, past_key_values_length=past_key_values_length, position_ids=position_ids
            )

        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`..."
                )
                use_cache = False
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (len(self.layers)), (
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    None,  # encoder attention mask
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,  # past_key_value
                    output_attentions,
                    use_cache,
                    spk_info, # This is new
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    spk_info=spk_info, # This is new
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

            if spk_info is not None:
                spk_info = layer_outputs[-1]

        hidden_states = self.layer_norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )