from __future__ import annotations

import torch
import torch.nn as nn

from src.config import DROPOUT_RATES, HIDDEN_DIMS


FEATURE_GROUPS = {
    "identity": [
        "age", "monthly_income", "previous_applications_count",
        "previous_rejections_count", "phone_age_days",
        "region", "employment_type", "email_domain", "phone_type",
    ],
    "loan": [
        "requested_amount", "loan_term_days", "application_hour",
        "application_weekday", "loan_type",
    ],
    "device": [
        "device_reuse_count", "ip_reuse_count",
        "proxy_flag", "vpn_flag", "tor_flag",
        "browser_type", "os_type",
    ],
    "behavioral": [
        "time_to_fill_application_sec", "number_of_corrections",
        "copy_paste_ratio", "typing_speed",
        "night_application_flag", "weekend_flag",
        "multiple_applications_last_1h", "multiple_applications_last_24h",
    ],
    "network": [
        "shared_device_with_other_iin_count", "shared_ip_with_other_iin_count",
        "cluster_id", "cluster_size",
        "age_income_zscore", "region_income_zscore",
        "application_velocity_score",
    ],
}


class _FeatureGroupBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FraudDetector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feature_cols: list[str] | None = None,
        hidden_dims: list[int] | None = None,
        dropout_rates: list[float] | None = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = HIDDEN_DIMS
        if dropout_rates is None:
            dropout_rates = DROPOUT_RATES

        self._group_indices: dict[str, list[int]] = {}
        self._use_groups = feature_cols is not None

        if self._use_groups and feature_cols:
            group_embed_dim = 16
            self.group_blocks = nn.ModuleDict()

            for group_name, group_features in FEATURE_GROUPS.items():
                indices = [i for i, col in enumerate(feature_cols) if col in group_features]
                if not indices:
                    continue
                self._group_indices[group_name] = indices
                self.group_blocks[group_name] = _FeatureGroupBlock(
                    input_dim=len(indices),
                    output_dim=group_embed_dim,
                    dropout=dropout_rates[0],
                )

            ungrouped = []
            grouped_cols = {col for feats in FEATURE_GROUPS.values() for col in feats}
            for i, col in enumerate(feature_cols):
                if col not in grouped_cols:
                    ungrouped.append(i)
            self._group_indices["_ungrouped"] = ungrouped

            fusion_input_dim = len(self.group_blocks) * group_embed_dim + len(ungrouped)
        else:
            fusion_input_dim = input_dim

        fusion_layers: list[nn.Module] = []
        prev_dim = fusion_input_dim

        for h_dim, drop in zip(hidden_dims, dropout_rates):
            fusion_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(drop),
            ])
            prev_dim = h_dim

        self.fusion = nn.Sequential(*fusion_layers)
        self.head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_groups and self._group_indices:
            parts = []
            for group_name, block in self.group_blocks.items():
                idx = self._group_indices[group_name]
                parts.append(block(x[:, idx]))

            ungrouped_idx = self._group_indices.get("_ungrouped", [])
            if ungrouped_idx:
                parts.append(x[:, ungrouped_idx])

            fused = torch.cat(parts, dim=1)
        else:
            fused = x

        return self.head(self.fusion(fused)).squeeze(-1)
