import os

import lhotse
import torch
import wandb
from safetensors.torch import load_file
from transformers import EarlyStoppingCallback
from transformers.utils import logging

from txt_norm import get_text_norm
from data.local_datasets import build_datasets, TS_ASR_Dataset
from models.containers import WhisperContainer, get_optimizer
from mt_asr.dataset import MT_SOT_Data_Collator, build_mt_asr_datasets
from utils.evaluation import compute_longform_metrics_for_sot
from utils.general import create_lower_uppercase_mapping
from utils.generation import update_generation_config
from utils.trainers import CustomTrainer, WeightDecayCallback, CustomParamLogCallback, ProfiledTrainer
from utils.training_args import Cfg, CustomTrainingArguments
from train import do_eval

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


def main(cfg: Cfg) -> None:
    logger.info(f"Config: {cfg}")
    cfg.save()

    model_args, data_args, decoding_args, training_args = cfg.model, cfg.data, cfg.decoding, cfg.training
    assert model_args.mt_sot
    # 1. Initialize container class
    container = WhisperContainer(model_type=model_args.whisper_model, pretrained_encoder=model_args.pretrained_encoder,
                              ctc_weight=model_args.ctc_weight, shift_pos_embeds=model_args.shift_pos_embeds,
                              training_args=training_args, predict_timestamps=data_args.use_timestamps,
                              fddt_is_diagonal=model_args.fddt_is_diagonal,
                              fddt_bias_only=model_args.fddt_bias_only,
                              fddt_use_silence=model_args.fddt_use_silence,
                              fddt_use_target=model_args.fddt_use_target,
                              fddt_use_overlap=model_args.fddt_use_overlap,
                              fddt_use_non_target=model_args.fddt_use_non_target,
                              non_target_fddt_value=model_args.non_target_fddt_value,
                              remove_timestamps_from_ctc=training_args.remove_timestamps_from_ctc,
                              apply_fddt_to_n_layers=model_args.apply_fddt_to_n_layers,
                              use_fddt=training_args.use_fddt,
                              fddt_init=model_args.fddt_init,
                              params_to_keep_frozen_keywords=model_args.params_to_keep_frozen_keywords,
                              use_initial_fddt=model_args.use_initial_fddt,
                              global_lang_id=data_args.global_lang_id,
                              mt_num_speakers=model_args.mt_num_speakers,
                              mt_interactions=model_args.mt_interactions,
                              mt_sot=model_args.mt_sot,
                              mt_sot_aggregate_speakers=model_args.mt_sot_aggregate_speakers,
                              mt_sot_aggregation_type=model_args.mt_sot_aggregation_type,
                              mt_sot_transform_speakers=model_args.mt_sot_transform_speakers,
                              mt_sot_use_sad=model_args.mt_sot_use_sad,
                              mt_sot_spk_mask_inv_temp=model_args.mt_sot_spk_mask_inv_temp,
                              mt_sot_encoder=model_args.mt_sot_encoder,
                              mt_sot_speaker_loss_weight=model_args.mt_sot_speaker_loss_weight,
                              mt_sot_speaker_attn_weight=model_args.mt_sot_speaker_attn_weight,
                              mt_sot_explicit_speaker_tokens=model_args.mt_sot_explicit_speaker_tokens,
                              mt_sot_style=model_args.mt_sot_style,
                              )

    # 2. Load the training data
    train_cutsets = [lhotse.load_manifest(cutset) for cutset in data_args.train_cutsets]

    # 3. Create dataset instances
    text_norm = get_text_norm(data_args.eval_text_norm)
    train_dataset = TS_ASR_Dataset(
        train_cutsets,
        do_augment=data_args.do_augment,
        dataset_weights=data_args.dataset_weights,
        use_timestamps=data_args.use_timestamps,
        musan_noises=data_args.musan_noises,
        text_norm=get_text_norm(data_args.train_text_norm),
        empty_transcript_ratio=data_args.empty_transcripts_ratio,
        train_with_diar_outputs=data_args.train_with_diar_outputs,
        audio_path_prefix=data_args.audio_path_prefix,
        audio_path_prefix_replacement=data_args.audio_path_prefix_replacement,
        vad_from_alignments=data_args.vad_from_alignments,
        random_sentence_l_crop_p=data_args.random_sentence_l_crop_p,
        random_sentence_r_crop_p=data_args.random_sentence_r_crop_p,
        max_l_crop=data_args.max_l_crop,
        max_r_crop=data_args.max_r_crop,
        feature_extractor=container.feature_extractor,
        global_lang_id=data_args.global_lang_id,
    )

    # Create mapping between lower case and upper case tokens
    create_lower_uppercase_mapping(container.tokenizer)

    dev_datasets = build_datasets(data_args.dev_cutsets, data_args, decoding_args, text_norm, container,
                                data_args.dev_diar_cutsets, debug=bool(training_args.debug))
    eval_dataset = build_datasets(data_args.eval_cutsets, data_args, decoding_args, text_norm, container,
                                 data_args.eval_diar_cutsets, debug=bool(training_args.debug))

    train_dataset = build_mt_asr_datasets(train_dataset, model_args, data_args, training_args, is_train=True)
    dev_datasets = build_mt_asr_datasets(dev_datasets, model_args, data_args, training_args, is_train=False)
    eval_dataset = build_mt_asr_datasets(eval_dataset, model_args, data_args, training_args, is_train=False)

    # 4. Get the model, possibly load pretrained weights and update generation config
    model = container.model
    logger.info(f"Model:\n{model}")

    if model_args.reinit_encoder_from:
        enc_state_dict = load_file(model_args.reinit_encoder_from)
        enc_state_dict_no_fddt = {k: v for k, v in enc_state_dict.items() if 'fddt' not in k}
        logger.info(model.get_encoder().load_state_dict(enc_state_dict_no_fddt, strict=False))

    if model_args.reinit_from:
        if model_args.reinit_from.endswith('.safetensors'):
            state_dict = load_file(model_args.reinit_from)
        else:
            # load all safetensors files in directory and merge to single dictionary
            state_dict = {}
            for file in os.listdir(model_args.reinit_from):
                if file.endswith('.safetensors'):
                    state_dict.update(load_file(os.path.join(model_args.reinit_from, file)))
        state_dict['proj_out.weight'] = state_dict['model.decoder.embed_tokens.weight']
        logger.info('Loading model weights from: ' + model_args.reinit_from)
        state_dict = {k.replace("encoder.target_amplifiers", "encoder.fddts") : v for k,v in state_dict.items()}
        logger.info(model.load_state_dict(state_dict, strict=False))

    update_generation_config(model, training_args, decoding_args)

    # 5. Initialize trainer
    collator = MT_SOT_Data_Collator(
        feature_extractor=container.feature_extractor,
        tokenizer=container.tokenizer,
        bos_token_id=container.model.config.decoder_start_token_id,
        mask_inputs=data_args.mask_inputs,
        max_length=training_args.generation_max_length,
        stno_gaussian_noise_var=data_args.stno_gaussian_noise_var,
        stno_gaussian_noise_prob=data_args.stno_gaussian_noise_prob,
    )

    trainer = CustomTrainer(
    # trainer = ProfiledTrainer(
        model=model,
        args=training_args,
        eval_dataset=dev_datasets,
        data_collator=collator,
        train_dataset=train_dataset,
        tokenizer=container.tokenizer,
        container=container,
        optimizers=(get_optimizer(model, training_args, model_args.prefixes_to_preheat), None),
        callbacks=get_callbacks(model, training_args),
        params_to_keep_frozen=model_args.params_to_keep_frozen_keywords,
    )

    if training_args.use_fddt_only_n_epochs > 0 or training_args.use_fddt_only_n_steps> 0:
        container.model.freeze_except(model_args.prefixes_to_preheat)

    if not model_args.reinit_from:
        container.model.suppress_interactions()

    # 6. Apply custom metric computation if needed
    if training_args.predict_with_generate:
        model.generation_config.ctc_weight = decoding_args.decoding_ctc_weight

        def _compute_metrics(pred, dset=None, split='dev', metrics_list=None):
            step = trainer.state.global_step
            output_dir = f'{trainer.args.output_dir}/{split}/{step}'
            os.makedirs(output_dir, exist_ok=True)
            return compute_longform_metrics_for_sot(
                pred, trainer, output_dir, text_norm,
                training_args.train_metrics_list if metrics_list is None else metrics_list,
                dset
            )

        trainer.compute_metrics = (
            lambda x: _compute_metrics(
                x, dev_datasets[trainer.metric_key_prefix.removeprefix("eval_")],
                split=trainer.metric_key_prefix, metrics_list=training_args.train_metrics_list
            )
        )


    # 7. Train the model
    if not training_args.decode_only:
        # import torch.profiler
        # with torch.profiler.profile(
        #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler("./logdir"),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True,
        # ) as prof:
        #     trainer.profiler = prof
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    do_eval(trainer, _compute_metrics, eval_dataset, decoding_args.decoding_ctc_weight, model,
            training_args.train_metrics_list, "test")


def get_callbacks(model: torch.nn.Module, training_args: CustomTrainingArguments):
    callbacks = []
    if training_args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(training_args.early_stopping_patience))

    if training_args.log_params is not None:
        params_to_log = {}
        for name, param in model.named_parameters():
            for param_name in training_args.log_params:
                if param_name in name:
                    logger.info(f"Param: {name} is set for custom logging")
                    params_to_log[param_name] = param
        callbacks.append(CustomParamLogCallback(params_to_log))

    if training_args.mt_sot_decay_params is not None and training_args.mt_sot_decay_steps > 0:
        params_to_decay = []
        for name, param in model.named_parameters():
            for param_name in training_args.mt_sot_decay_params:
                if param_name in name:
                    logger.info(f"Param: {name} is set for weight decaying")
                    params_to_decay.append(param)
        callbacks.append(WeightDecayCallback(params_to_decay, training_args.mt_sot_decay_steps))
    return callbacks or None
