from types import MethodType
from typing import Any, Union, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import Seq2SeqTrainer, Trainer
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.training_args import TrainingArguments
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import logging
from transformers.utils import is_datasets_available
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
if is_datasets_available():
    import datasets
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
import wandb

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


class CustomTrainerEncoder(Trainer):
    def __init__(self, container, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_w_cast = None
        self.forward_wo_cast = None
        self.container = container

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
            **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if hasattr(self.container, 'h'):
            self.container.h.set_diar_output(inputs['vad_mask'])

        labels = inputs.pop("labels")

        # else:
        output = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        loss = self.model.get_loss(output[1], labels)

        output = (loss, output[1], labels)
        return output

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        for token in self.tokenizer.prefix_tokens:
            if (labels[:, 0] == token).all():
                labels = labels[:, 1:]
        labels[labels == self.tokenizer.eos_token_id] = -100

        loss = self.model.get_loss(outputs.logits, labels)

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        if hasattr(self.container, 'h'):
            self.container.h.set_diar_output(inputs['vad_mask'])
        out = super().training_step(model, inputs)
        return out


class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, container, *args, params_to_keep_frozen, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_w_cast = None
        self.forward_wo_cast = None
        self.container = container
        self.warmup_phase = True
        self.params_to_keep_frozen = params_to_keep_frozen
        self.metric_key_prefix = ""

    def prediction_step_local(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
            **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        has_labels = False
        inputs = self._prepare_inputs(inputs)

        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        generation_inputs = inputs.copy()
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
                "labels" in generation_inputs
                and "decoder_input_ids" in generation_inputs
                and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {
                k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
            }
        generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        labels = inputs["labels"]
        if labels.shape[-1] < gen_config.max_length:
            labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
            labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)

        return loss, generated_tokens, labels

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
            **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        has_valid_field = "is_valid" in inputs
        if has_valid_field:
            # TODO: Move this logic into MT-ASR Dataset, this just adjust
            # the batch_size to batch-global max number of REAL speakers
            max_spks = inputs["per_group_sizes"].max().item()
            for key in inputs.keys():
                if key == "is_valid" or key == "per_group_sizes":
                    continue
                inputs[key] = inputs[key].reshape(inputs['per_group_sizes'].shape[0], -1, *inputs[key].shape[1:])[:,
                              :max_spks].flatten(0, 1)
            is_valid = inputs.pop("is_valid")
        if hasattr(self.container, 'h'):
            self.container.h.set_diar_output(inputs['vad_mask'])

        if self.args.bf16_full_eval and not prediction_loss_only:
            forward = model.forward
            original_forward = model.__dict__.pop("_original_forward", None)
            if original_forward is not None:
                self.forward_w_cast = forward
                while hasattr(forward, "__wrapped__"):
                    forward = forward.__wrapped__
                    if forward == original_forward:
                        break
                self.forward_wo_cast = MethodType(forward, model)

            model.forward = self.forward_wo_cast

            with torch.autocast(dtype=torch.bfloat16, device_type=self.model.device.type):
                output = self.prediction_step_local(
                    model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
                )
            model.forward = self.forward_w_cast
        else:
            output = self.prediction_step_local(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
            )
        if has_valid_field:
            # TODO: This code just appends back fake speakers
            # let's move this logic outside of trainer, e.g. into collators?
            num_groups = inputs['per_group_sizes'].shape[0]
            loss, generated_tokens, labels = output
            generated_tokens_original = torch.full((is_valid.shape[0], generated_tokens.shape[1]), -100,
                                                   dtype=torch.long, device=generated_tokens.device).reshape(num_groups,
                                                                                                             -1,
                                                                                                             *generated_tokens.shape[
                                                                                                              1:])
            # TODO: This is problematic, due to broadcasting, e.g. if generated_tokens has only 1 speaker (SOT)
            # this will repeat the tokens to `max_spks`
            generated_tokens_original[:, :max_spks] = generated_tokens.reshape(num_groups, -1,
                                                                               *generated_tokens.shape[1:])
            generated_tokens_original = generated_tokens_original.flatten(0, 1)
            labels_original = torch.full((is_valid.shape[0], labels.shape[1]), -100, dtype=torch.long,
                                         device=labels.device).reshape(num_groups, -1, *labels.shape[1:])
            labels_original[:, :max_spks] = labels.reshape(num_groups, -1, *labels.shape[1:])
            labels_original = labels_original.flatten(0, 1)
            output = (loss, generated_tokens_original, labels_original)
        return output

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        # TODO: Move this logic into MT-ASR Dataset, this just adjust
        # the batch_size to batch-global max number of REAL speakers
        if "is_valid" in inputs:
            max_spks = inputs["per_group_sizes"].max().item()
            for key in inputs.keys():
                if key == "is_valid" or key == "per_group_sizes":
                    continue
                inputs[key] = inputs[key].reshape(inputs['per_group_sizes'].shape[0], -1, *inputs[key].shape[1:])[:,
                              :max_spks].flatten(0, 1)
            inputs.pop("is_valid")

        if hasattr(self.container, 'h'):
            self.container.h.set_diar_output(inputs['vad_mask'])

        output = super().training_step(model, inputs)

        if self.warmup_phase and self.state.epoch >= self.args.use_fddt_only_n_epochs and self.state.global_step >= self.args.use_fddt_only_n_steps:
            for name, param in self.model.named_parameters():
                required_grad_before = param.requires_grad
                require_grad = True
                for keyword in self.params_to_keep_frozen:
                    if keyword in name:
                        require_grad = required_grad_before  # We don't want to freeze params that are already unfrozen
                        break
                param.requires_grad = require_grad
                if required_grad_before ^ param.requires_grad:
                    logger.debug(f"Param: {name} now requires grad: {param.requires_grad}.")
            logger.info(f"***** Unfreezing params except {self.params_to_keep_frozen}*****")
            logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

            self.warmup_phase = False
        return output

    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
                if self.state.train_batch_size is not None:
                    self.args.gradient_accumulation_steps *= (self.state.train_batch_size // self._train_batch_size)
                    if args is not None:
                        args.gradient_accumulation_steps = self.args.gradient_accumulation_steps
            self.state.train_batch_size = self._train_batch_size
        out = super()._inner_training_loop(
            batch_size=batch_size, args=args, resume_from_checkpoint=resume_from_checkpoint, trial=trial
        )
        return out

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        self.metric_key_prefix = metric_key_prefix
        output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        return output

def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": 1, # Due to the longform nature of data and possible RAM issues we fall back to single worker be gpu
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)


class WeightDecayCallback(TrainerCallback):
    """Gradually decrease the value of a paramater

    This is usefull for e.g. getting rid of diarization mask in SOT decoder,
    i.e. this will force the model to learn the mask by itself.
    """
    def __init__(self, params: List[torch.nn.Parameter], num_steps: int, begin_step=0, initial_weight=1.0, final_weight=0.0):
        self.params = params
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.num_steps = num_steps
        self.begin_step = begin_step
        self.initial_values = [p.data for p in params]

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step >= self.begin_step and state.global_step - self.begin_step <= self.num_steps:
            current_weight = self.initial_weight + (state.global_step / self.num_steps) * (self.final_weight - self.initial_weight)
            with torch.no_grad():
                for param, value in zip(self.params, self.initial_values):
                    param.data = (value * current_weight).to(param.device, dtype=param.dtype)

        if state.global_step - self.begin_step <= self.num_steps:
            control.should_training_stop = False
            return control

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step - self.begin_step <= self.num_steps:
            control.should_training_stop = False
            return control

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if wandb.run is not None:
            wandb.log(
                {f"train/wd_param_{i}_mean": p.detach().cpu().numpy().mean() for i, p in enumerate(self.params)},
            )
        logger.info(f"{state.global_step}/{state.max_steps} WeightDecay params: {[p.detach().cpu().numpy() for p in self.params]}")

    def state(self):
        return {
            "args": {
                "params": self.params,
                "num_steps": self.num_steps,
                "begin_steps": self.begin_step,
                "initial_weight": self.initial_weight,
                "final_weight": self.final_weight,
            },
            "attributes": {}
        }


class CustomParamLogCallback(TrainerCallback):
    def __init__(self, params: Dict[str, torch.nn.Parameter]):
        for k,p in params.items():
            if p.ndim > 1:
                raise ValueError(f"Unsupported param {k} with dim {p.ndim}")
        self.params = params

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if wandb.run is not None:
            for name, param in self.params.items():
                if param.numel() > 1:
                    msg = {f"train/{name}_{i}": p.detach().cpu() for i,p in enumerate(param)}
                else:
                    msg = {f"train/{name}": param.detach().cpu()}
                wandb.log(msg)
                logger.info(f"{state.global_step}/{state.max_steps} Custom Param Logger: {name}: {param}")


class ProfiledTrainer(CustomTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiler = None  # Set externally

    def training_step(self, model, inputs):
        output = super().training_step(model, inputs)
        if self.profiler is not None:
            self.profiler.step()
        return output
