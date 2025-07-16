import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutput, BaseModelOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoder, WhisperEncoderLayer, WHISPER_ATTENTION_CLASSES, \
    sinusoids

from models.dicow.config import DiCoWConfig
from mt_asr.interactions import Interaction


class CustomLinear(nn.Linear):
    def __init__(self, *args, init_eye_val=0.0, is_diagonal=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_eye_val = init_eye_val


class CustomDiagonalLinear(nn.Module):
    def __init__(self, d_model, bias=True, init_eye_val=0.0):
        super().__init__()
        self.init_eye_val = init_eye_val
        self.weight = nn.Parameter(torch.full((d_model,), init_eye_val))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, input):
        out = input * self.weight
        if self.bias is not None:
            out += self.bias
        return out


class FDDT(nn.Module):
    def __init__(self, d_model, non_target_rate=0.01, is_diagonal=False, bias_only=False, use_silence=True,
                 use_target=True, use_overlap=True, use_non_target=True):
        super().__init__()
        if use_target:
            self.target_linear = nn.Parameter(torch.zeros(d_model)) if bias_only else (
                CustomDiagonalLinear(d_model, bias=True, init_eye_val=1.0) if is_diagonal else CustomLinear(d_model,
                                                                                                            d_model,
                                                                                                            bias=True,
                                                                                                            init_eye_val=1.0))
        if use_non_target:
            self.non_target_linear = nn.Parameter(torch.zeros(d_model)) if bias_only else (
                CustomDiagonalLinear(d_model, bias=True, init_eye_val=non_target_rate) if is_diagonal else CustomLinear(
                    d_model, d_model, bias=True, init_eye_val=non_target_rate))
        if use_overlap:
            self.overlap_linear = nn.Parameter(torch.zeros(d_model)) if bias_only else (
                CustomDiagonalLinear(d_model, bias=True, init_eye_val=1.0) if is_diagonal else CustomLinear(d_model,
                                                                                                            d_model,
                                                                                                            bias=True,
                                                                                                            init_eye_val=1.0))
        if use_silence:
            self.silence_linear = nn.Parameter(torch.zeros(d_model)) if bias_only else (
                CustomDiagonalLinear(d_model, bias=True, init_eye_val=non_target_rate) if is_diagonal else CustomLinear(
                    d_model, d_model, bias=True, init_eye_val=non_target_rate))

        self.use_silence = use_silence
        self.use_target = use_target
        self.use_overlap = use_overlap
        self.use_non_target = use_non_target
        self.bias_only = bias_only

    def forward(self, hidden_states, stno_mask, per_group_sizes):
        stno_mask = stno_mask.to(hidden_states.device)[..., None]
        if self.bias_only:
            if self.use_silence:
                hidden_states += stno_mask[:, 0, ...] * self.silence_linear
            if self.use_target:
                hidden_states += stno_mask[:, 1, ...] * self.target_linear
            if self.use_non_target:
                hidden_states += stno_mask[:, 2, ...] * self.non_target_linear
            if self.use_overlap:
                hidden_states += stno_mask[:, 3, ...] * self.overlap_linear

        else:
            orig_hidden_states = hidden_states
            hidden_states = (self.silence_linear(
                orig_hidden_states) if self.use_silence else orig_hidden_states) * stno_mask[:, 0, :] + \
                            (self.target_linear(
                                orig_hidden_states) if self.use_target else orig_hidden_states) * stno_mask[:, 1, :] + \
                            (self.non_target_linear(
                                orig_hidden_states) if self.use_non_target else orig_hidden_states) * stno_mask[:, 2,
                                                                                                      :] + \
                            (self.overlap_linear(
                                orig_hidden_states) if self.use_overlap else orig_hidden_states) * stno_mask[:, 3, :]
        return hidden_states


class DiCoWEncoder(WhisperEncoder):
    config_class = DiCoWConfig

    def __init__(self, config: DiCoWConfig):
        super().__init__(config)
        self.ctc_weight = config.ctc_weight
        if config.additional_layer and self.ctc_weight > 0.0:
            self.additional_layer = WhisperEncoderLayer(config)
        if config.additional_self_attention_layer and self.ctc_weight > 0.0:
            self.additional_self_attention_layer = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
                embed_dim=config.d_model,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
                config=config,
            )
        if config.sub_sample and self.ctc_weight > 0.0:
            self.subsample_conv1 = nn.Conv1d(
                in_channels=config.d_model,
                out_channels=config.d_model,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
            self.subsample_conv2 = nn.Conv1d(
                in_channels=config.d_model,
                out_channels=config.d_model,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
        if self.ctc_weight > 0.0:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size + 1, bias=False)
        self.final_dropout = nn.Dropout(config.final_dropout)
        if config.use_fddt:
            num_fddts = self.config.apply_fddt_to_n_layers if self.config.apply_fddt_to_n_layers != -1 else len(
                self.layers)
            self.initial_fddt = FDDT(config.d_model,
                                     non_target_rate=config.non_target_fddt_value,
                                     is_diagonal=config.fddt_is_diagonal,
                                     bias_only=config.fddt_bias_only,
                                     use_silence=config.fddt_use_silence,
                                     use_target=config.fddt_use_target,
                                     use_overlap=config.fddt_use_overlap,
                                     use_non_target=config.fddt_use_non_target)
            self.fddts = nn.ModuleList([
                FDDT(config.d_model,
                     non_target_rate=1.0,
                     is_diagonal=config.fddt_is_diagonal,
                     bias_only=config.fddt_bias_only,
                     use_silence=config.fddt_use_silence,
                     use_target=config.fddt_use_target,
                     use_overlap=config.fddt_use_overlap,
                     use_non_target=config.fddt_use_non_target)

                for i in range(num_fddts)
            ])
        self.first_task_token = self.config.vocab_size - 30 * 50 -1 - 6 # 30 seconds of 50 Hz timestamps -1 to get to 0.0 and -6 number of tasks
        if config.mt_interactions:
            self.interaction = Interaction(config)
        self.post_init()

    @classmethod
    def _load_pretrained_model(
            cls,
            model,
            state_dict,
            loaded_keys,
            resolved_archive_file,
            pretrained_model_name_or_path,
            **kwargs
    ):
        for key in list(state_dict.keys()):
            if key.startswith("encoder."):
                state_dict[key[8:]] = state_dict.pop(key)
                loaded_keys.remove(key)
                loaded_keys.append(key[8:])
        output = super()._load_pretrained_model(
            model,
            state_dict,
            loaded_keys,
            resolved_archive_file,
            pretrained_model_name_or_path,
            **kwargs
        )
        return output

    def get_loss(self, logits, labels):
        if labels.max() >= self.config.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")
        if self.config.remove_timestamps_from_ctc:
            labels = torch.nn.utils.rnn.pad_sequence([label[label < self.first_task_token] for label in labels],
                                                     padding_value=-100).T
        input_lengths = torch.full((logits.shape[0],), fill_value=logits.shape[1],
                                   device=logits.device)

        # assuming that padded tokens are filled with -100
        # when not being attended to
        labels_mask = labels >= 0
        target_lengths = labels_mask.sum(-1)
        # flattened_targets = labels_enc.masked_select(labels_mask)

        # ctc_loss doesn't support fp16
        log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

        with torch.backends.cudnn.flags(enabled=True):
            ctc_loss = nn.functional.ctc_loss(
                log_probs,
                labels,
                input_lengths,
                target_lengths,
                blank=logits.shape[-1] - 1,
                reduction=self.config.ctc_loss_reduction,
                zero_infinity=True,
            )
        return ctc_loss

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
        # For MT-ASR the input has shape (B X S) x F x T
        # we can use torch.view(B, S, F, -1) to obtain
        # new tensor with speaker dim
        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if input_features.shape[-1] != expected_seq_length:
            if input_features.shape[-1] > expected_seq_length:
                return CausalLMOutput(
                    logits=None,
                    hidden_states=None,
                    attentions=None,
                )
            else:
                raise ValueError(
                    f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
                )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight
        if hasattr(self, "shift_embeds") and self.shift_embeds:
            embed_pos = embed_pos[
                torch.clamp(((stno_mask[:, 1, :] + stno_mask[:, 3, :]).cumsum(dim=-1) - 1), min=0).to(torch.long)]

        if self.config.use_fddt:
            inputs_embeds = self.initial_fddt(inputs_embeds, stno_mask, per_group_sizes)

        hidden_states = inputs_embeds + embed_pos

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if self.config.use_fddt and idx < len(self.fddts):
                hidden_states = self.fddts[idx](hidden_states, stno_mask, per_group_sizes)

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        None,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        None,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            outputs = tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        else:
            outputs = BaseModelOutput(
                last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
            )

        if hasattr(self, "interaction"):
            outputs.last_hidden_state = self.interaction(
                outputs.last_hidden_state,
                per_group_sizes
            )
            outputs.hidden_states = (*outputs.hidden_states[:-1], outputs.last_hidden_state)

        if hasattr(self, "additional_layer"):
            inter_output, = self.additional_layer(
                outputs.last_hidden_state,
                attention_mask=None,
                output_attentions=output_attentions,
                layer_head_mask=None,
            )
        elif hasattr(self, "additional_self_attention_layer"):
            inter_output, _, __ = self.additional_self_attention_layer(
                outputs.last_hidden_state,
                attention_mask=None,
                output_attentions=output_attentions,
                layer_head_mask=None,
            )
        else:
            inter_output = outputs.last_hidden_state

        inter_output = self.final_dropout(inter_output)
        if hasattr(self, "subsample_conv2"):
            inter_output = self.subsample_conv2(self.subsample_conv1(inter_output.transpose(1, 2))).transpose(1, 2)
        if self.ctc_weight > 0.0:
            logits = self.lm_head(inter_output)
        else:
            logits = None

        return CausalLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
