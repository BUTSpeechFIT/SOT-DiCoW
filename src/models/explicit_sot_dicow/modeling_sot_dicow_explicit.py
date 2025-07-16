import torch
from torch.nn import CrossEntropyLoss
import wandb

from transformers.models.speech_encoder_decoder.modeling_speech_encoder_decoder import (
    shift_tokens_right,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.utils import logging

from models.dicow.config import Seq2SeqLMOutputLosses
from models.dicow.modeling_dicow import DiCoWForConditionalGeneration
from models.sot_dicow.modeling_sot_dicow import BranchLinear, BranchDiagonalLinear, SOTAggregationLayer
from models.sot_dicow.modeling_sot_dicow import SOTDiCoWEncoder, SOTDiCoWModel, SOTDiCoWDecoder
from models.sot_dicow.utils import convert_to_spk_info
from .config import ExplicitSOTDiCoWConfig
from .generation import ExplicitSOTDiCoWGenerationMixin


class ExplicitSOTDiCoWModel(SOTDiCoWModel):
    """Whisper model modified for multi-speaker transcription."""
    config_class = ExplicitSOTDiCoWConfig

    def __init__(self, config: ExplicitSOTDiCoWConfig):
        super().__init__(config)
        self.encoder = SOTDiCoWEncoder(config)
        self.decoder = SOTDiCoWDecoder(config)


class ExplicitSOTDiCoWForConditionalGeneration(ExplicitSOTDiCoWGenerationMixin, DiCoWForConditionalGeneration):
    """Whisper model for conditional generation with multi-speaker transcription."""
    config_class = ExplicitSOTDiCoWConfig

    def __init__(self, config: ExplicitSOTDiCoWConfig):
        super().__init__(config)
        self.model = ExplicitSOTDiCoWModel(config)
        self.ctc_weight = config.ctc_weight
        self.speaker_loss_weight = config.mt_sot_speaker_loss_weight
        self.first_spk_idx = config.vocab_size
        self.vocab_size = config.vocab_size + config.mt_num_speakers

    def _init_weights(self, module: torch.nn.Module) -> None:
        super()._init_weights(module)
        std = self.config.init_std
        if isinstance(module, SOTAggregationLayer):
            module._init_weights()
        elif isinstance(module, BranchLinear) or isinstance(module, BranchDiagonalLinear):
            module._init_weights()

    @property
    def max_supported_speakers(self) -> int:
        """Get the maximum number of speakers supported by the model."""
        return self.config.mt_num_speakers

    def forward(
            self,
            input_features: torch.FloatTensor | None = None,
            stno_mask: torch.FloatTensor | None = None,
            per_group_sizes: torch.LongTensor | None = None,
            attention_mask_enc: torch.LongTensor | None = None,
            attention_mask: torch.LongTensor | None = None,
            decoder_input_ids: torch.LongTensor | None = None,
            decoder_attention_mask: torch.LongTensor | None = None,
            head_mask: torch.Tensor | None = None,
            decoder_head_mask: torch.Tensor | None = None,
            cross_attn_head_mask: torch.Tensor | None = None,
            encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = None,
            past_key_values: tuple[tuple[torch.FloatTensor]] | None = None,
            decoder_inputs_embeds: tuple[torch.FloatTensor] | None = None,
            decoder_position_ids: tuple[torch.LongTensor] | None = None,
            labels: torch.LongTensor | None = None,
            upp_labels: torch.LongTensor | None = None,
            sot_labels: torch.LongTensor | None = None,
            sot_upp_labels: torch.LongTensor | None = None,
            use_cache: bool | None = None,
            output_attentions: bool | None = None,
            output_hidden_states: bool | None = None,
            return_dict: bool | None = None,
            spk_info: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...] | Seq2SeqLMOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(inputs=input_features)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None and self.config.mt_sot_aggregation_type is None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        elif sot_labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    sot_labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            stno_mask=stno_mask,
            per_group_sizes=per_group_sizes,
            spk_info=spk_info,
        )

        dec_lm_logits = self.proj_out(outputs.last_hidden_state)
        enc_lm_logits = outputs.encoder_logits

        loss = None
        ctc_loss = 0

        if labels is not None and self.ctc_weight > 0.0:
            enc_labels = labels.clone()
            for token in self.tokenizer.prefix_tokens:
                if (enc_labels[:, 0] == token).all():
                    enc_labels = enc_labels[:, 1:]
            enc_labels[enc_labels == self.config.eos_token_id] = -100

            ctc_loss = self.get_encoder().get_loss(enc_lm_logits, enc_labels)
            if wandb.run is not None:
                wandb.log({"ctc_loss": ctc_loss})

        if sot_labels is not None:
            if self.speaker_loss_weight > 0:
                spk_ts_loss_weight = torch.ones(self.vocab_size, device=dec_lm_logits.device, dtype=dec_lm_logits.dtype)
                spk_ts_loss_weight[self.first_spk_idx:] = self.speaker_loss_weight
                loss_fct = CrossEntropyLoss(reduction='none', weight=spk_ts_loss_weight)
            else:
                loss_fct = CrossEntropyLoss(reduction='none')
            # move labels to correct device to enable PP
            sot_labels = sot_labels.to(dec_lm_logits.device)
            sot_upp_labels = sot_upp_labels.to(dec_lm_logits.device)
            dec_loss1 = loss_fct(dec_lm_logits.view(-1, self.vocab_size), sot_labels.reshape(-1))
            dec_loss2 = loss_fct(dec_lm_logits.view(-1, self.vocab_size), sot_upp_labels.reshape(-1))
            dec_loss = torch.hstack((dec_loss1[..., None], dec_loss2[..., None])).min(dim=-1).values
            dec_loss = dec_loss.sum() / (sot_labels != -100).sum()
            loss = (1 - self.ctc_weight) * dec_loss + self.ctc_weight * ctc_loss

            if wandb.run is not None:
                wandb.log({"att_loss": dec_loss})

        if not return_dict:
            output = (dec_lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutputLosses(
            loss=loss,
            logits=dec_lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            encoder_logits=enc_lm_logits,
        )

    def prepare_inputs_for_generation(self, decoder_input_ids, *args, past_key_values=None, **kwargs):
        inputs = super().prepare_inputs_for_generation(decoder_input_ids, *args, past_key_values=past_key_values, **kwargs)
        spk_info = convert_to_spk_info(decoder_input_ids, self.first_spk_idx, self.max_supported_speakers, explicit_speaker_tokens=True)

        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            spk_info = spk_info[:, remove_prefix_length:]

        inputs["spk_info"] = spk_info
        return inputs