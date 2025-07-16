# Author: Martin Kocour (BUT)

import torch
from torch import nn

from models.sot_dicow.config import SOTAggregationType, SOTDiCoWConfig
from models.sot_dicow.utils import speaker_groups

class BranchLinear(nn.Module):
    def __init__(self, num_branches, hidden_size):
        super().__init__()
        self.num_branches = num_branches
        self.dim = hidden_size

        # Combine all weights and biases into single tensors
        self.weight = nn.Parameter(torch.empty((num_branches, hidden_size, hidden_size)))
        self.bias = nn.Parameter(torch.empty((num_branches, hidden_size)))
        self._init_weights()

    def _init_weights(self):
        for i in range(self.num_branches):
            torch.nn.init.constant_(self.bias.data[i], 0.0)
            torch.nn.init.eye_(self.weight.data[i])

    def forward(self, x: torch.Tensor, branch_idx: torch.Tensor):
        """
        x: (B, T, dim) or (T, dim)
        branch_idx: (B,) or (T,) or None
        """
        if x.ndim == 2:
            T, D = x.shape
            assert D == self.dim
            assert branch_idx.shape == (T,)

            # Select weights and biases per time step
            W = self.weight[branch_idx]  # (T, D, D)
            b = self.bias[branch_idx]    # (T, D)

            # Apply linear transformation: (T, 1, D) x (T, D, D) -> (T, 1, D)
            x_proj = torch.bmm(x.unsqueeze(1), W).squeeze(1)  # (T, D)
            x_proj = x_proj + b
            return x_proj
        else:
            B, T, D = x.shape
            assert D == self.dim

            W = self.weight  # (N, D, D)
            b = self.bias    # (N, 1, D)
            if branch_idx is not None:
                W = self.weight[branch_idx]
                b = self.bias[branch_idx]

            assert x.shape[0] == W.shape[0]
            x_proj = torch.bmm(x, W)  # (N, T, D)
            x_proj = x_proj + b.unsqueeze(1)
            return x_proj


class BranchDiagonalLinear(nn.Module):
    def __init__(self, num_branches, hidden_size):
        super().__init__()
        self.num_branches = num_branches
        self.dim = hidden_size

        # Combine all weights and biases into single tensors
        self.weight = nn.Parameter(torch.empty((num_branches, hidden_size)))
        self.bias = nn.Parameter(torch.empty((num_branches, hidden_size)))
        self._init_weights()

    def _init_weights(self):
        for i in range(self.num_branches):
            torch.nn.init.constant_(self.bias.data[i], 0.0)
            torch.nn.init.constant_(self.weight.data[i], 1.0)

    def forward(self, x: torch.Tensor, branch_idx: torch.Tensor):
        """
        x: (B, T, dim) or (T, dim)
        branch_idx: (B,) or (T,) or None
        """
        if x.ndim == 2:
            T, D = x.shape
            assert D == self.dim
            assert branch_idx.shape == (T,)

            # Select weights and biases per time step
            W = self.weight[branch_idx]  # (T, D)
            b = self.bias[branch_idx]    # (T, D)

            x_proj = x * W + b
            return x_proj
        else:
            B, T, D = x.shape
            assert D == self.dim

            W = self.weight  # (N, D)
            b = self.bias    # (N, D)
            if branch_idx is not None:
                W = self.weight[branch_idx]
                b = self.bias[branch_idx]

            x_proj = x * W.unsqueeze(1) + b.unsqueeze(1)  # (N, T, D)
            return x_proj


class SOT_FDDT(nn.Module):
    def __init__(self, d_model, non_target_rate=0.01, is_diagonal=False, bias_only=False, use_silence=True,
                 use_target=True, use_overlap=True, use_non_target=True, num_branches=1):
        super().__init__()
        self.silence = BranchDiagonalLinear(num_branches, d_model) if is_diagonal else BranchLinear(num_branches, d_model)
        self.target = BranchDiagonalLinear(num_branches, d_model) if is_diagonal else BranchLinear(num_branches, d_model)
        self.non_target = BranchDiagonalLinear(num_branches, d_model) if is_diagonal else BranchLinear(num_branches, d_model)
        self.overlap = BranchDiagonalLinear(num_branches, d_model) if is_diagonal else BranchLinear(num_branches, d_model)

    def forward(self, hidden_states, stno_mask, per_group_sizes):
        stno_mask = stno_mask.to(hidden_states.device).unsqueeze(-1)
        branch_idx = torch.cat([torch.arange(pgp, device=hidden_states.device) for pgp in per_group_sizes])

        orig_hidden_states = hidden_states
        hidden_states = self.silence(   orig_hidden_states, branch_idx) * stno_mask[:, 0, :] + \
                        self.target(    orig_hidden_states, branch_idx) * stno_mask[:, 1, :] + \
                        self.non_target(orig_hidden_states, branch_idx) * stno_mask[:, 2, :] + \
                        self.overlap(   orig_hidden_states, branch_idx) * stno_mask[:, 3, :]
        return hidden_states


class SOTAggregationLayer(nn.Module):
    """Layer that aggregates hidden states across speaker channels."""
    def __init__(self, config: SOTDiCoWConfig):
        super().__init__()
        self.config = config
        if config.mt_sot_transform_speakers:
            self.speaker_pos_enc = BranchLinear(config.mt_num_speakers, config.d_model)

        # weight the speaker mask (before softmax)
        if config.mt_sot_spk_mask_inv_temp is not None:
            self.spk_mask_weight = nn.Parameter(torch.empty(2))

    def _init_weights(self, val=0.0):
        if hasattr(self, "speaker_pos_enc"):
            self.speaker_pos_enc._init_weights()

        if self.config.mt_sot_spk_mask_inv_temp is not None:
            torch.nn.init.constant_(self.spk_mask_weight.data, val=self.config.mt_sot_spk_mask_inv_temp)

    def _aggregate(self, group, mask):
        aggregate_type = SOTAggregationType(self.config.mt_sot_aggregation_type)
        if aggregate_type is None or aggregate_type == SOTAggregationType.NONE:
            return group * mask

        if aggregate_type == SOTAggregationType.SUM:
            return (group * mask).sum(dim=0, keepdim=True)
        if aggregate_type == SOTAggregationType.MEAN:
            if hasattr(self, "spk_mask_weight"):
                # if mask contains only silence, than 1/N_spkrs
                mask = torch.softmax(mask, 0)
                return (group * mask).sum(dim=0, keepdim=True)

            group = (group * mask).sum(dim=0, keepdim=True)
            div = mask.sum(dim=0, keepdim=True)
            div = div.where(div > 0, 1)  # avoid div by 0
            return group / div
        if aggregate_type == SOTAggregationType.CONCAT:
            # Concatenate across time
            if hasattr(self, "spk_mask_weight"):
                raise NotImplementedError("Spk mask is currently not supported")
            return group.reshape(-1, group.shape[-1]).unsqueeze(0)

    def _forward_spkr_info(self, group_hidden_states: torch.Tensor, spk_mask: torch.Tensor):
        if not hasattr(self, "speaker_pos_enc"):
            return group_hidden_states, spk_mask

        branch_idx = torch.arange(group_hidden_states.shape[0], device=group_hidden_states.device)
        group = self.speaker_pos_enc(group_hidden_states, branch_idx)
        return group, spk_mask

    def _prepare_spk_mask(self, hidden_states: torch.Tensor, stno_mask=None):
        if stno_mask is None:
            return hidden_states.new_ones(hidden_states.shape[:-1]).unsqueeze(-1)

        if hasattr(self, "spk_mask_weight"):
            tmp_mask = stno_mask[:, [1, 3]]
            spk_mask = torch.tensordot(tmp_mask, self.spk_mask_weight, dims=[[1], [0]]) # type: ignore
            return spk_mask.unsqueeze(-1)

        stno_mask = stno_mask[:, 0, ...] + stno_mask[:, 1, ...] + stno_mask[:, 3, ...]
        spk_mask = hidden_states.new_ones(stno_mask.size())
        spk_mask = spk_mask.where(stno_mask > 0, 0)
        return spk_mask.unsqueeze(-1)

    def forward(self, hidden_states: torch.Tensor, per_group_sizes, stno_mask=None):
        """Average hidden states over speaker channels within each group"""
        spk_masks = self._prepare_spk_mask(hidden_states, stno_mask)
        outputs = []
        for group_size, group_hidden_states, group_spk_masks in speaker_groups(per_group_sizes, hidden_states, spk_masks):
            group_hidden_states, group_spk_masks = self._forward_spkr_info(group_hidden_states, group_spk_masks)  # S x T x d_model
            outputs.append(self._aggregate(group_hidden_states, group_spk_masks))
        return torch.cat(outputs, dim=0)