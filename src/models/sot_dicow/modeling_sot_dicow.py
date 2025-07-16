# Author: Martin Kocour (BUT)
"""Module implementing multi-speaker transcription using
Serialized Output Training (SOT) with Whisper."""

from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

import wandb

from transformers.models.whisper.modeling_whisper import shift_tokens_right
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.utils import logging

from models.dicow.config import Seq2SeqModelOutputLogit
from models.dicow.modeling_dicow import (
    Seq2SeqLMOutputLosses,
    DiCoWEncoder,
    CustomLinear,
    DiCoW,
    DiCoWForConditionalGeneration
)
from models.sot_dicow.config import SOTDiCoWConfig
from models.sot_dicow.utils import timestamp_tokens_len, handle_different_batch_size, convert_to_spk_info
from models.sot_dicow.generation import SOTDiCoWGenerationMixin
from models.sot_dicow.sot_layers import SOT_FDDT, SOTAggregationLayer, BranchLinear, BranchDiagonalLinear
from models.sot_dicow.sot_decoder import SOTDiCoWDecoder, WhisperSOTEmbedding


logging.set_verbosity_debug()
logger = logging.get_logger("sot_dicow")


class SOTDiCoWEncoder(DiCoWEncoder):
    """Diarization Conditioned Whisper Encoder for Seriliazed Output Training"""
    config_class = SOTDiCoWConfig

    def __init__(self, config: SOTDiCoWConfig):
        super().__init__(config)
        if config.mt_sot_aggregate_speakers:
            self.sot_layer = SOTAggregationLayer(config)

        if config.mt_sot_encoder:
            self.initial_fddt = SOT_FDDT(
                config.d_model,
                non_target_rate=config.non_target_fddt_value,
                is_diagonal=config.fddt_is_diagonal,
                bias_only=config.fddt_bias_only,
                use_silence=config.fddt_use_silence,
                use_target=config.fddt_use_target,
                use_overlap=config.fddt_use_overlap,
                use_non_target=config.fddt_use_non_target,
                num_branches=config.mt_num_speakers,
            )
            self.fddts = nn.ModuleList([
                SOT_FDDT(
                    config.d_model,
                    non_target_rate=1.0,
                    is_diagonal=config.fddt_is_diagonal,
                    bias_only=config.fddt_bias_only,
                    use_silence=config.fddt_use_silence,
                    use_target=config.fddt_use_target,
                    use_overlap=config.fddt_use_overlap,
                    use_non_target=config.fddt_use_non_target,
                    num_branches=config.mt_num_speakers,
                )
                for i in range(len(self.fddts))
            ])

        self.use_sad = config.mt_sot_use_sad

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        stno_mask=None,
        per_group_sizes=None
    ):
        if input_features.shape[0] != stno_mask.shape[0]:
            input_features = handle_different_batch_size(input_features, stno_mask, per_group_sizes)
        outputs = super().forward(
            input_features, None, head_mask, output_attentions,
            output_hidden_states, return_dict, stno_mask, per_group_sizes,
        )
        if hasattr(self, "sot_layer"):
            outputs.hidden_states += (
                self.sot_layer(outputs.hidden_states[-1], per_group_sizes, stno_mask if self.use_sad else None),
            )
        return outputs


class SOTDiCoWModel(DiCoW):
    """Whisper model modified for multi-speaker transcription."""
    config_class = SOTDiCoWConfig

    def __init__(self, config: SOTDiCoWConfig):
        super().__init__(config)
        self.encoder = SOTDiCoWEncoder(config)
        self.decoder = SOTDiCoWDecoder(config)

    def forward(
            self,
            input_features: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
            decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            stno_mask: Optional[torch.FloatTensor] = None,
            per_group_sizes: Optional[torch.LongTensor] = None,
            spk_info: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutputLosses]:
        r"""
        Returns:

        Example:
         ```python
         >>> import torch
         >>> from transformers import AutoFeatureExtractor, WhisperModel
         >>> from datasets import load_dataset

         >>> model = WhisperModel.from_pretrained("openai/whisper-base")
         >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
         >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
         >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
         >>> input_features = inputs.input_features
         >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
         >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
         >>> list(last_hidden_state.shape)
         [1, 2, 512]
         ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            input_features = self._mask_input_features(input_features, attention_mask=attention_mask)

            encoder_outputs = self.encoder(
                input_features,
                output_attentions=output_attentions,
                output_hidden_states=True,
                head_mask=head_mask,
                return_dict=return_dict,
                stno_mask=stno_mask,
                per_group_sizes=per_group_sizes
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        # elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        #     raise ValueError("encoder_outputs should be of type BaseModelOutput when return_dict=True.")

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.hidden_states[-1],
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            spk_info=spk_info,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutputLogit(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.hidden_states[-1],
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            encoder_logits=encoder_outputs.logits,
        )


class SOTWhisperOutput(torch.nn.Module):
    """Output layer for multi-speaker transcription that handles speaker-specific logits."""
    def __init__(self, config: SOTDiCoWConfig):
        super().__init__()
        self.mt_num_speakers = config.mt_num_speakers
        self.spkr_ts_weight = CustomLinear(config.d_model, config.mt_num_speakers, bias=False, init_eye_val=1.0)
        self.first_ts_idx = config.vocab_size - timestamp_tokens_len()
        self.norm = torch.log(torch.tensor(config.mt_num_speakers))

    def forward(self, hidden_states: torch.Tensor, orig_output: torch.Tensor) -> torch.Tensor:
        """Generate speaker-specific logits for timestamp tokens.

        Args:
            hidden_states: Hidden states from the decoder.
            orig_output: Original logits before speaker-specific processing.
        Returns:
            Modified logits with speaker-specific timestamp tokens.
        """
        output_vocab = orig_output[..., :self.first_ts_idx] # B x T x N_vocab w.o. ts
        output_ts = orig_output[..., self.first_ts_idx:] # B x T x N_ts

        # Adding more TS tokens redistributes the probability,
        # that's why we need to normalize the TS logits
        # so the overall sum (logsumexp) is same as before
        # B x T x Ns
        output_spkrs = self.spkr_ts_weight(hidden_states)
        # B x T x Ns x Nt
        output_ts = output_ts.unsqueeze(-2) + output_spkrs.unsqueeze(-1) - self.norm.to(output_ts.device)
        B, T, Ns, Nt = output_ts.shape
        output_ts = output_ts.reshape(B, T, Nt * Ns)
        new_output = torch.cat((output_vocab, output_ts), dim=-1)
        return new_output


class SOTDiCoWForConditionalGeneration(SOTDiCoWGenerationMixin, DiCoWForConditionalGeneration):
    """Whisper model for conditional generation with multi-speaker transcription."""
    config_class = SOTDiCoWConfig

    def __init__(self, config: SOTDiCoWConfig):
        super().__init__(config)
        self.model = SOTDiCoWModel(config)
        self.ctc_weight = config.ctc_weight
        self.speaker_loss_weight = config.mt_sot_speaker_loss_weight
        self.first_ts_idx = config.vocab_size - timestamp_tokens_len()
        self.proj_out2 = SOTWhisperOutput(config)
        self.vocab_size = config.vocab_size + timestamp_tokens_len(config.mt_num_speakers - 1)

    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.init_std
        if isinstance(module, WhisperSOTEmbedding):
            with torch.no_grad():
                if isinstance(module.spkr_ts_weight, BranchLinear):
                    module.spkr_ts_weight._init_weights()
                else:
                    # TODO: Remove this in future
                    module.spkr_ts_weight.data = torch.ones_like(module.spkr_ts_weight.data).data

                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, SOTWhisperOutput):
            torch.nn.init.zeros_(module.spkr_ts_weight.weight.data)
        elif isinstance(module, SOTAggregationLayer):
            module._init_weights()
        elif isinstance(module, BranchLinear) or isinstance(module, BranchDiagonalLinear):
            module._init_weights()

    @property
    def max_supported_speakers(self):
        """Get the maximum number of speakers supported by the model."""
        return self.config.mt_num_speakers

    def forward(
            self,
            input_features: Optional[torch.FloatTensor] = None,
            stno_mask: Optional[torch.FloatTensor] = None,
            per_group_sizes: Optional[torch.LongTensor] = None,
            attention_mask_enc: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
            decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            upp_labels: Optional[torch.LongTensor] = None,
            sot_labels: Optional[torch.LongTensor] = None,
            sot_upp_labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            spk_info: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
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

        orig_dec_lm_logits = self.proj_out(outputs.last_hidden_state)
        if hasattr(self, "proj_out2"):
            dec_lm_logits = self.proj_out2(outputs.last_hidden_state, orig_dec_lm_logits)
        else:
            dec_lm_logits = orig_dec_lm_logits
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

        if labels is not None and self.config.mt_sot_aggregation_type is None:
            loss_fct = CrossEntropyLoss(reduction='none')
            labels = labels.to(dec_lm_logits.device)
            upp_labels = upp_labels.to(dec_lm_logits.device)
            dec_loss1 = loss_fct(dec_lm_logits.view(-1, self.vocab_size), labels.reshape(-1))
            dec_loss2 = loss_fct(dec_lm_logits.view(-1, self.vocab_size), upp_labels.reshape(-1))
            dec_loss = torch.hstack((dec_loss1[..., None], dec_loss2[..., None])).min(dim=-1).values.sum() / (labels != 100).sum()
            loss = (1 - self.ctc_weight) * dec_loss + self.ctc_weight * ctc_loss

            if wandb.run is not None:
                wandb.log({"att_loss": dec_loss})

        elif sot_labels is not None:
            if self.speaker_loss_weight > 0:
                spk_ts_loss_weight = torch.ones(self.vocab_size, device=dec_lm_logits.device, dtype=dec_lm_logits.dtype)
                spk_ts_loss_weight[self.first_ts_idx:] = self.speaker_loss_weight
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
        spk_info = convert_to_spk_info(decoder_input_ids, self.first_ts_idx, self.max_supported_speakers, explicit_speaker_tokens=False)

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