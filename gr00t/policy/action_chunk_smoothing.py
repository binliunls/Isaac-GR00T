# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Smooths the action chunk produced by an inner policy with an interpolated
# Savitzky-Golay filter, run independently per chunk. This reduces high-
# frequency per-step jitter in the commanded trajectory without changing the
# overall plan the policy intends to follow.

from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from gr00t.policy.policy import PolicyWrapper, BasePolicy


class ActionChunkSmoothingWrapper(PolicyWrapper):
    """Wrapper that smooths actions within each action chunk.

    Pipeline (per get_action call):
      1. Forward observation to the inner policy → action chunk (B, T, D) per key.
      2. Up-sample each chunk's time axis by `time_points_scale` via interp1d.
      3. Apply Savitzky-Golay over the dense time axis per (batch, joint).
      4. Down-sample back to the original T time points.

    Args:
        policy:            inner PolicyWrapper / BasePolicy.
        smoothing_type:    interp1d kind for the up-sample/down-sample steps.
        smooth_window:     Savgol window length (must be odd and ≤ dense T).
        smooth_order:      Savgol polynomial order (< smooth_window).
        time_points_scale: how much to densify the time axis before Savgol.
    """

    def __init__(
        self,
        policy: BasePolicy,
        *,
        smoothing_type: str = "linear",
        smooth_window: int = 5,
        smooth_order: int = 3,
        time_points_scale: float = 2.0,
        strict: bool = True,
    ):
        super().__init__(policy, strict=strict)
        if smoothing_type not in ("linear", "cubic", "quadratic"):
            raise ValueError(f"Invalid smoothing_type: {smoothing_type!r}")
        if smooth_order >= max(smooth_window, 1):
            raise ValueError(
                f"smooth_order ({smooth_order}) must be < smooth_window ({smooth_window})"
            )
        self.smoothing_type = smoothing_type
        self.smooth_window = int(smooth_window)
        self.smooth_order = int(smooth_order)
        self.time_points_scale = float(time_points_scale)

    # ── pass-throughs ───────────────────────────────────────────────────────

    def get_modality_config(self):
        return self.policy.get_modality_config()

    def check_observation(self, observation: dict[str, Any]) -> None:
        # Inner policy already validates; treat this wrapper as transparent.
        pass

    def check_action(self, action: dict[str, Any]) -> None:
        pass

    # ── core ────────────────────────────────────────────────────────────────

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Call the inner policy via its public get_action (handles its own validation).
        action_chunk, info = self.policy.get_action(observation, options)
        if not action_chunk:
            return action_chunk, info

        # Determine chunk horizon from the first array-valued key.
        horizon = None
        for v in action_chunk.values():
            if isinstance(v, np.ndarray) and v.ndim == 3:
                horizon = v.shape[1]
                break
        if horizon is None or horizon < 2:
            return action_chunk, info

        orig_t = np.arange(horizon)
        dense_t = np.linspace(0, horizon - 1, max(int(horizon * self.time_points_scale), horizon))

        smoothed_chunk: dict[str, Any] = {}
        for key, actions in action_chunk.items():
            if not (isinstance(actions, np.ndarray) and actions.ndim == 3):
                # leave non-3D entries unchanged (e.g. metadata)
                smoothed_chunk[key] = actions
                continue

            # 1) up-sample
            interp_up = interp1d(
                orig_t, actions, kind=self.smoothing_type, axis=1,
                bounds_error=False, fill_value=np.nan,
            )
            dense_actions = interp_up(dense_t)

            # 2) Savgol over dense time, per (batch, dim)
            if 0 < self.smooth_window <= len(dense_t):
                smoothed_dense = np.empty_like(dense_actions)
                for b in range(actions.shape[0]):
                    for d in range(actions.shape[2]):
                        smoothed_dense[b, :, d] = savgol_filter(
                            dense_actions[b, :, d],
                            window_length=self.smooth_window,
                            polyorder=self.smooth_order,
                        )
            else:
                smoothed_dense = dense_actions

            # 3) down-sample back to original timesteps
            interp_down = interp1d(
                dense_t, smoothed_dense, kind="linear", axis=1,
                bounds_error=False, fill_value=np.nan,
            )
            smoothed_actions = interp_down(orig_t)

            # NaN guard: fall back to original where smoothing produced NaN.
            nan_mask = np.isnan(smoothed_actions)
            if nan_mask.any():
                smoothed_actions = np.where(nan_mask, actions, smoothed_actions)

            smoothed_chunk[key] = smoothed_actions.astype(actions.dtype, copy=False)

        return smoothed_chunk, info
