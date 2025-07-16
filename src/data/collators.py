from dataclasses import dataclass
from typing import Any, List, Dict, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchFeature
from transformers.utils import logging

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")

@dataclass
class DataCollator:
    feature_extractor: Any
    tokenizer: Any
    bos_token_id: Any
    max_length: int
    conv_subsample_factor: int = 2
    mask_inputs: bool = False
    stno_gaussian_noise_var: float = None
    stno_gaussian_noise_prob: float = None
    filter_long_samples: bool = True

    @staticmethod
    def add_gaussian_noise_and_rescale(prob_mask, variance=0.05, fraction=0.5):
        B, C, T = prob_mask.shape
        num_noisy_batches = int(B * fraction)  # Number of batches to modify

        if num_noisy_batches == 0:  # Return original if no batches are selected
            return prob_mask

        # Randomly select which batches to apply noise to
        noisy_indices = torch.randperm(B)[:num_noisy_batches]

        # Create a copy of the original mask to avoid modifying input
        noisy_mask = prob_mask.clone()

        # Apply noise only to the selected batches
        noise = torch.randn((num_noisy_batches, C, T), device=prob_mask.device) * (variance ** 0.5)
        noisy_mask[noisy_indices] += noise  # Add noise

        # Compute the minimum value along C axis
        min_vals = noisy_mask[noisy_indices].amin(dim=1, keepdim=True)

        # Apply shift only where min_vals < 0
        min_vals = torch.clamp(min_vals, max=0)  # Keep only negative values
        noisy_mask[noisy_indices] -= min_vals  # Shift up if needed

        # Normalize to sum to 1 over C axis
        noisy_mask[noisy_indices] /= noisy_mask[noisy_indices].sum(dim=1, keepdim=True)

        return noisy_mask

    @staticmethod
    def is_all_true_or_all_false(lst):
        return all(lst) or not any(lst)

    def __call__(self, inputs: List[Dict[str, Union[List[int], torch.Tensor]]]) -> BatchFeature:
        longform = [sample['is_long_form'] for sample in inputs]
        if len(set(longform)) != 1:
            raise ValueError(f"Some inputs are longform and some are not")

        in_longform = longform[0]

        if not in_longform and self.filter_long_samples:
            # Tokenize the labels
            labels = self.tokenizer([sample["transcript"] for sample in inputs],
                                    padding="longest", max_length=self.max_length, return_tensors="pt")

            # remove samples from batch that are longer than max model input len 448
            lens = labels['attention_mask'].sum(dim=1)
            inputs_to_keep = lens < self.max_length

            inputs_filtered = []
            for mask, sample in zip(inputs_to_keep, inputs):
                if mask:
                    inputs_filtered.append(sample)
            if len(inputs_filtered) == 0:
                inputs_filtered.append(inputs[0]) # it can happen that we have only faulty data in batch, than return at least one element
                logger.warning("Empty batch after filtering.")
            inputs = inputs_filtered

        labels = self.tokenizer([sample["transcript"] for sample in inputs],
                                padding="longest", max_length=self.max_length, return_tensors="pt")
        feats = pad_sequence([
            sample['input_features'].squeeze().T for sample in inputs]).permute(1, 2, 0)
        masks = pad_sequence([
            sample['attention_mask'].T for sample in inputs]).squeeze().T

        stno_masks = pad_sequence([sample['stno_mask'].T for sample in inputs]).permute(1, 2, 0)

        orig_stno_masks_len = [sample['stno_mask'].shape[1] for sample in inputs]
        for i, sample in enumerate(stno_masks):
            stno_masks[i][0, orig_stno_masks_len[i]:] = 1

        batch = BatchFeature({'input_features': feats, 'attention_mask': masks, 'stno_mask': stno_masks})

        languages = [sample.get("language") for sample in inputs]
        if all(languages):
            langs = [f"<|{sample}|>" for sample in languages]
            langs = self.tokenizer.convert_tokens_to_ids(langs)
            if in_longform:
                # we are in generation mode and languages are provided
                batch["forced_decoder_ids"] = torch.tensor(
                    [[self.tokenizer.prefix_tokens[0], language, self.tokenizer.prefix_tokens[2]] for language in
                     langs])
            else:
                # we are in training modify labels with lang
                labels['input_ids'][:, 1] = torch.tensor(langs)
        elif any(languages):
            raise ValueError(
                f"Some inputs have language and some not. Please unify it if you want to condition by language.")

        batch["labels"] = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (batch["labels"][:, 0] == self.bos_token_id).all().cpu().item():
            batch["labels"] = batch["labels"][:, 1:]
        batch['upp_labels'] = batch['labels'].clone().apply_(
            lambda x: self.tokenizer.upper_cased_tokens.get(int(x)) if int(
                x) in self.tokenizer.upper_cased_tokens else x)

        if self.stno_gaussian_noise_var is not None and self.stno_gaussian_noise_var > 0:
            if not ("is_long_form" in inputs[0] and inputs[0]['is_long_form']):
                batch["stno_mask"] = self.add_gaussian_noise_and_rescale(batch["stno_mask"],
                                                                         self.stno_gaussian_noise_var,
                                                                         self.stno_gaussian_noise_prob)
        return batch


@dataclass
class DataCollatorForPretraining(DataCollator):
    use_timestamps: bool = False

    def __call__(self, inputs: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
        # batch = self.feature_extractor([sample["audio"]["array"] for sample in inputs], return_tensors="pt",
        #                                sampling_rate=16_000, return_attention_mask=True)
        # orig_lens = torch.tensor([sample['audio']["array"].shape[-1] for sample in inputs])
        #
        # # Tokenize the labels
        # labels = self.tokenizer(
        #     [add_timestamps(sample["transcript"], orig_lens[i].item())["transcript"] if self.use_timestamps else sample[
        #         "transcript"] for i, sample in enumerate(inputs)],
        #     padding="longest", max_length=self.max_length, return_tensors="pt")
        #
        # batch["labels"] = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        # if (batch["labels"][:, 0] == self.bos_token_id).all().cpu().item():
        #     batch["labels"] = batch["labels"][:, 1:]
        # vad_mask_shape = batch["input_features"].shape
        # batch["vad_mask"] = torch.zeros((vad_mask_shape[0], vad_mask_shape[1], vad_mask_shape[2] // 2),
        #                                 device=batch["input_features"].device, )
        # for idx, input_len in enumerate(orig_lens):
        #     batch["vad_mask"][idx, 1, :input_len] = 1.0
        #
        # return batch
