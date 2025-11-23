# Notebook copied from Amrosm on Kaggle from the publicly available code for the competition
'''
# MABe Challenge - Social Action Recognition in Mice: Nearest neighbors

This is the original notebook for social action recognition with nearest neighbors. 
I've tried to explain what the code does—feel free to ask questions.

The notebook shows how to overcome the five challenges of this competition:
1. Modeling for variable-size sets of mice
2. Multiclass prediction with missing labels
3. Transforming coordinates to an invariant representation
4. A dataset that doesn't fit into memory
5. Modeling for variable sets of body parts

The title of the notebook mentions *Nearest Neighbors* 
because in earlier versions I used nearest neighbors classification, 
an algorithm which doesn't need a lot of tuning. 
The current version uses LightGBM, and maybe I'll ensemble the two later.

References
- Competition: [MABe Challenge - Social Action Recognition in Mice](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection)
- [MABe EDA which makes sense ⭐️⭐️⭐️⭐️⭐️](https://www.kaggle.com/code/ambrosm/mabe-eda-which-makes-sense)
- [MABe Validated baseline without machine learning](https://www.kaggle.com/code/ambrosm/mabe-validated-baseline-without-machine-learning)

This notebook can be run in validate or submission mode. 
If you look at other saved versions of this notebook, you'll see both modes. 
You can switch between the modes by setting the variable `validate_or_submit`:
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import itertools
import warnings
import json
import os
import lightgbm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.model_selection import cross_val_predict, GroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


@dataclass
class CTRGCNConfig:
    """
    Configuration for controlling the CTR-GCN training pipeline.

    Attributes
    ----------
    mode : str
        One of {"dev", "validate", "submit"}.
        - "dev": train on a very small subset (quick tests)
        - "validate": cross-validation on the full dataset
        - "submit": full training on everything for submission

    max_videos : int | None
        If not None, limit the number of videos processed during training.

    max_batches : int | None
        If not None, limit how many batches from generate_mouse_data() are processed.

    max_windows : int | None
        If not None, limit how many sliding windows per batch we convert to tensors.

    use_delta : bool
        Whether to compute Δx, Δy velocity channels.

    two_stream : bool
        If True: return coords and delta as two separate tensors for a two-stream CTR-GCN.
        If False: merge coords + delta into a single stream with extra channels.

    in_channels_coords : int
        Coordinate channels (x, y).

    in_channels_delta : int
        Δx, Δy channels.

    in_channels_single_stream : int
        Combined coords (x, y) + delta (Δx, Δy) for single-stream input.

    use_bone : bool
        Whether to compute bone vectors.

    use_bone_delta : bool
        Whether to compute bone deltas.

    stream_mode : str
        One of {"one", "two", "four"}; controls stream splitting for CTR-GCN.

    in_channels_streamA : int
        Channels for stream A (coords + bone) in two-stream mode.

    in_channels_streamB : int
        Channels for stream B (delta + bone_delta) in two-stream mode.

    in_channels_coords_only : int
        Channels for coords-only stream in four-stream mode.

    in_channels_delta_only : int
        Channels for delta-only stream in four-stream mode.

    in_channels_bone_only : int
        Channels for bone-only stream in four-stream mode.

    in_channels_bone_delta_only : int
        Channels for bone-delta-only stream in four-stream mode.
    """

    mode: str = "dev"
    max_videos: int | None = 3
    max_batches: int | None = 10
    max_windows: int | None = 50
    use_delta: bool = True
    two_stream: bool = False
    in_channels_coords: int = 2
    in_channels_delta: int = 2
    in_channels_single_stream: int = 4
    use_bone: bool = True
    use_bone_delta: bool = True
    stream_mode: str = "one"
    in_channels_streamA: int = 4
    in_channels_streamB: int = 4
    in_channels_coords_only: int = 2
    in_channels_delta_only: int = 2
    in_channels_bone_only: int = 2
    in_channels_bone_delta_only: int = 2

# Functions requiring updates for stream_mode and bone features:
# - prepare_ctr_gcn_input
# - train_ctr_gcn_models
# - CTRGCNTwoStream (extend input channels)
# - (new) CTRGCNFourStream

# ------------------------------------------------------------
# CTR-GCN: Master anatomical ordering for all mouse body parts
# ------------------------------------------------------------
MASTER_MOUSE_JOINT_ORDER = [
    "nose",
    "head",

    "headpiece_topfrontleft",
    "headpiece_topfrontright",
    "headpiece_topbackleft",
    "headpiece_topbackright",
    "headpiece_bottomfrontleft",
    "headpiece_bottomfrontright",
    "headpiece_bottombackleft",
    "headpiece_bottombackright",

    "ear_left",
    "ear_right",

    "neck",

    "forepaw_left",
    "forepaw_right",

    "body_center",
    "lateral_left",
    "lateral_right",

    "spine_1",
    "spine_2",

    "hip_left",
    "hip_right",

    "hindpaw_left",
    "hindpaw_right",

    "tail_base",
    "tail_middle_1",
    "tail_middle_2",
    "tail_midpoint",
    "tail_tip",
]
# ------------------------------------------------------------
# CTR-GCN: Ordering joints + building adjacency matrix
# ------------------------------------------------------------
def get_ordered_joints_and_adjacency(body_parts_tracked):
    """
    Sorts the subset of tracked body parts according to MASTER_MOUSE_JOINT_ORDER
    and builds a simple chain adjacency matrix (i <-> i+1).
    """
    ordered_joints = [
        bp for bp in MASTER_MOUSE_JOINT_ORDER
        if bp in body_parts_tracked
    ]

    V = len(ordered_joints)
    adjacency = np.zeros((V, V), dtype=np.float32)

    for i in range(V - 1):
        adjacency[i, i + 1] = 1.0
        adjacency[i + 1, i] = 1.0

    return ordered_joints, adjacency

# ------------------------------------------------------------
# CTR-GCN: Sliding window extraction (window=90, stride=30)
# ------------------------------------------------------------
def create_sliding_windows_90_30(single_mouse_df):
    """
    Takes a continuous single-mouse coordinate DataFrame and yields
    windows of length 90 with stride 30, preserving frame indices.
    """
    WINDOW = 90
    STRIDE = 30

    n_frames = len(single_mouse_df)
    frames = single_mouse_df.index.to_numpy()

    for start in range(0, n_frames - WINDOW + 1, STRIDE):
        end = start + WINDOW
        window_df = single_mouse_df.iloc[start:end]
        frame_indices = frames[start:end]
        yield window_df, frame_indices


# ------------------------------------------------------------
# CTR-GCN: Build tensors for spatio-temporal model input
# ------------------------------------------------------------
def prepare_ctr_gcn_input(single_mouse_df, ordered_joints, config: CTRGCNConfig | None = None):
    """
    Convert a sliding-windowed single-mouse DataFrame into
    CTR-GCN input tensors of shape (N, 2, V, 90).

    Parameters
    ----------
    single_mouse_df : DataFrame
        Raw continuous coordinate data for a single mouse.
        Columns are a two-level MultiIndex: (bodypart, xy).

    ordered_joints : list[str]
        Anatomically ordered joints for this model.

    Returns
    -------
    tensors : torch.Tensor of shape (N, 2, V, 90)
    frame_ranges : list of np.ndarray
        Each element is the frame indices for that window.

    Note
    ----
    The optional config.max_windows limit is useful for dev/testing to cap work.
    The optional config.use_delta/config.two_stream settings enable velocity and two-stream output for dev/validate/submit.
    """
    mode = "one"
    if config is not None:
        mode = config.stream_mode if hasattr(config, "stream_mode") else "one"
        if getattr(config, "two_stream", False) and mode == "one":
            mode = "two"
    V = len(ordered_joints)
    frame_ranges = []
    window_count = 0
    if config is None or mode == "one":
        window_tensors = []
    elif mode == "two":
        streamA_tensors = []
        streamB_tensors = []
    else:  # four
        coords_tensors = []
        delta_tensors = []
        bone_tensors = []
        bone_delta_tensors = []

    for window_df, frame_indices in create_sliding_windows_90_30(single_mouse_df):
        if config is not None and config.max_windows is not None and window_count >= config.max_windows:
            break
        if len(window_df) != 90:
            # Skip incomplete windows to keep tensor shape consistent.
            continue

        window_np = np.full((2, V, 90), np.nan, dtype=np.float32)
        for j, bp in enumerate(ordered_joints):
            try:
                coords = window_df[bp]
                window_np[0, j, :] = coords['x'].to_numpy(dtype=np.float32, copy=False)
                window_np[1, j, :] = coords['y'].to_numpy(dtype=np.float32, copy=False)
            except KeyError:
                # Missing bodypart columns remain NaN.
                continue

        if config is None:
            window_tensors.append(torch.from_numpy(window_np))
            frame_ranges.append(frame_indices)
            window_count += 1
            continue

        # Compute bone vectors (pad last with zeros)
        bone = np.zeros_like(window_np)
        for j in range(V - 1):
            bone[:, j, :] = window_np[:, j + 1, :] - window_np[:, j, :]

        # Step 1: mean-centering
        mean_val = np.nanmean(window_np, axis=2, keepdims=True)
        window_np = window_np - mean_val
        bone = bone - mean_val

        # Step 2: root-joint normalization
        anchor_name = None
        if "body_center" in ordered_joints:
            anchor_name = "body_center"
        elif "neck" in ordered_joints:
            anchor_name = "neck"
        if anchor_name is not None:
            anchor_idx = ordered_joints.index(anchor_name)
            anchor = window_np[:, anchor_idx:anchor_idx+1, :]
            window_np = window_np - anchor
            bone = bone - anchor

        # Step 3: scale normalization
        scale = np.nanstd(window_np)
        if scale > 0:
            window_np = window_np / scale
            bone = bone / scale

        # Deltas
        if config.use_delta:
            delta = window_np[:, :, 1:] - window_np[:, :, :-1]
            delta = np.concatenate([np.zeros_like(delta[:, :, :1]), delta], axis=2)
        else:
            delta = np.zeros_like(window_np)

        if config.use_bone:
            bone_curr = bone
        else:
            bone_curr = np.zeros_like(window_np)

        if config.use_bone_delta:
            bone_delta = bone_curr[:, :, 1:] - bone_curr[:, :, :-1]
            bone_delta = np.concatenate([np.zeros_like(bone_delta[:, :, :1]), bone_delta], axis=2)
        else:
            bone_delta = np.zeros_like(window_np)

        if mode == "one":
            merged = np.concatenate([window_np, delta, bone_curr, bone_delta], axis=0)
            window_tensors.append(torch.from_numpy(merged.astype(np.float32)))
        elif mode == "two":
            streamA = np.concatenate([window_np, bone_curr], axis=0)
            streamB = np.concatenate([delta, bone_delta], axis=0)
            streamA_tensors.append(torch.from_numpy(streamA.astype(np.float32)))
            streamB_tensors.append(torch.from_numpy(streamB.astype(np.float32)))
        else:  # four
            coords_tensors.append(torch.from_numpy(window_np.astype(np.float32)))
            delta_tensors.append(torch.from_numpy(delta.astype(np.float32)))
            bone_tensors.append(torch.from_numpy(bone_curr.astype(np.float32)))
            bone_delta_tensors.append(torch.from_numpy(bone_delta.astype(np.float32)))

        frame_ranges.append(frame_indices)
        window_count += 1

    if config is None or mode == "one":
        if len(window_tensors) == 0:
            channels = 2 if config is None else config.in_channels_single_stream
            return torch.empty((0, channels, V, 90)), frame_ranges
        return torch.stack(window_tensors, dim=0), frame_ranges
    if mode == "two":
        return streamA_tensors, streamB_tensors, frame_ranges
    return coords_tensors, delta_tensors, bone_tensors, bone_delta_tensors, frame_ranges


# ------------------------------------------------------------
# CTR-GCN: Minimal spatio-temporal GCN model (single-mouse)
# ------------------------------------------------------------
def _normalize_adjacency_chain(adjacency: np.ndarray) -> np.ndarray:
    """
    Add self-loops and row-normalize a simple chain adjacency matrix.

    Parameters
    ----------
    adjacency : np.ndarray of shape (V, V)
        Symmetric adjacency matrix for the joints graph.

    Returns
    -------
    A_norm : np.ndarray of shape (V, V)
        Row-normalized adjacency with self-loops.
    """
    V = adjacency.shape[0]
    A = adjacency.astype(np.float32).copy()
    # Add self-loops
    A += np.eye(V, dtype=np.float32)
    # Row-normalize
    row_sum = A.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    A_norm = A / row_sum
    return A_norm


class GraphConv(nn.Module):
    """
    Simple graph convolution operating on (N, C, V, T) tensors.

    Given an adjacency matrix A (V, V), this layer first aggregates
    information from neighboring joints via A, then applies a 1x1
    convolution over the channel dimension.
    """
    def __init__(self, in_channels: int, out_channels: int, adjacency: np.ndarray):
        super().__init__()
        A_norm = _normalize_adjacency_chain(adjacency)
        self.register_buffer("A", torch.from_numpy(A_norm))  # shape (V, V)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, V, T)
        returns: (N, C_out, V, T)
        """
        # Aggregate neighbor information along the joint dimension
        # Using einsum: (N, C, V, T) x (V, V) -> (N, C, V, T)
        x = torch.einsum("ncvT,vw->ncwT", x, self.A)
        x = self.conv(x)
        return x


class STBlock(nn.Module):
    """
    Spatio-temporal block:
    - Graph convolution over joints
    - Temporal convolution over frames
    - Residual connection (if shapes match)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: np.ndarray,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels, adjacency)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(1, 3),
                padding=(0, 1),
                stride=(1, stride),
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        if (in_channels != out_channels) or (stride != 1):
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(1, stride),
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C_in, V, T)
        returns: (N, C_out, V, T_out)
        """
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        x = self.relu(x)
        return x


class CTRGCNMinimal(nn.Module):
    """
    Minimal CTR-GCN-style model for single-mouse behavior classification.

    This model:
    - Expects input of shape (N, C, V, T) where:
        N = batch size
        C = number of channels (2 for x/y coordinates)
        V = number of joints (len(ordered_joints))
        T = number of frames (e.g., 90)
    - Uses a simple chain adjacency matrix per body-part set.
    - Outputs logits of shape (N, num_classes) for binary or multi-label tasks.
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        adjacency: np.ndarray,
        base_channels: int = 64,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.V = adjacency.shape[0]

        channels = [base_channels] * num_blocks
        blocks = []
        last_c = in_channels
        for i, out_c in enumerate(channels):
            # No temporal downsampling for now (stride = 1)
            blocks.append(
                STBlock(
                    in_channels=last_c,
                    out_channels=out_c,
                    adjacency=adjacency,
                    stride=1,
                    dropout=dropout,
                )
            )
            last_c = out_c

        self.st_blocks = nn.ModuleList(blocks)
        self.fc = nn.Linear(last_c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, V, T)

        Returns
        -------
        logits : torch.Tensor
            Output tensor of shape (N, num_classes).
        """
        # Ensure correct shape
        assert x.ndim == 4, f"Expected (N, C, V, T), got {x.shape}"

        out = x
        for block in self.st_blocks:
            out = block(out)

        # Global average pooling over joints (V) and time (T)
        out = out.mean(dim=(-2, -1))  # (N, C_out)
        logits = self.fc(out)
        return logits


class CTRGCNTwoStream(nn.Module):
    """
    Two-stream CTR-GCN.

    - Stream A processes coordinate channels (x, y or similar).
    - Stream B processes delta/velocity channels (Δx, Δy).
    - Features are fused via summation and fed to a final classifier head.
    - Use with CTRGCNConfig.two_stream=True when supplying separate coord/delta inputs.
    """
    def __init__(
        self,
        adjacency: np.ndarray,
        in_channels_coords: int = 2,
        in_channels_delta: int = 2,
        base_channels: int = 64,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stream_coords = CTRGCNMinimal(
            in_channels=in_channels_coords,
            num_classes=base_channels,
            adjacency=adjacency,
            base_channels=base_channels,
            num_blocks=num_blocks,
            dropout=dropout,
        )
        self.stream_delta = CTRGCNMinimal(
            in_channels=in_channels_delta,
            num_classes=base_channels,
            adjacency=adjacency,
            base_channels=base_channels,
            num_blocks=num_blocks,
            dropout=dropout,
        )
        self.fc = nn.Linear(base_channels, 1)

    def forward(self, coords_x: torch.Tensor, delta_x: torch.Tensor) -> torch.Tensor:
        feat_A = self.stream_coords(coords_x)
        feat_B = self.stream_delta(delta_x)
        fused = feat_A + feat_B
        logits = self.fc(fused)
        return logits


class CTRGCNFourStream(nn.Module):
    """
    Four-stream CTR-GCN:
      Stream 1: coords
      Stream 2: deltas
      Stream 3: bones
      Stream 4: bone_deltas
    Fuses all four via elementwise sum, followed by a linear classifier.
    """
    def __init__(
        self,
        adjacency,
        base_channels=64,
        dropout=0.1,
        num_blocks=3,
    ):
        super().__init__()
        self.stream_coords = CTRGCNMinimal(2, base_channels, adjacency, base_channels, num_blocks, dropout)
        self.stream_delta = CTRGCNMinimal(2, base_channels, adjacency, base_channels, num_blocks, dropout)
        self.stream_bone = CTRGCNMinimal(2, base_channels, adjacency, base_channels, num_blocks, dropout)
        self.stream_bone_delta = CTRGCNMinimal(2, base_channels, adjacency, base_channels, num_blocks, dropout)

        self.fc = nn.Linear(base_channels, 1)

    def forward(self, coords_x, delta_x, bone_x, bone_delta_x):
        f1 = self.stream_coords(coords_x)
        f2 = self.stream_delta(delta_x)
        f3 = self.stream_bone(bone_x)
        f4 = self.stream_bone_delta(bone_delta_x)
        fused = f1 + f2 + f3 + f4
        return self.fc(fused)


def train_ctr_gcn_models(
    batches,
    ordered_joints,
    adjacency,
    config: CTRGCNConfig,
    device: str = "cpu",
):
    """
    Train one CTR-GCN binary classifier per action using sliding-window inputs.

    Parameters
    ----------
    batches : list of tuples
        Each item is (data_df, meta_df, label_df) for a single-mouse batch.
        data_df: raw coordinate data (pvid)
        meta_df: video/agent/target/frame metadata
        label_df: per-frame binary labels for each action

    ordered_joints : list[str]
        Anatomically ordered joints.

    adjacency : np.ndarray
        V x V adjacency matrix for CTR-GCN.

    config : CTRGCNConfig
        Controls max_windows, max_batches, and dev/validate/submit modes.

    device : str
        "cpu", "cuda", or "mps"

    Returns
    -------
    model_dict : dict[str, CTRGCNMinimal]
        A dictionary mapping action → trained CTR-GCN model.
    """
    model_dict: dict[str, CTRGCNMinimal] = {}
    y_dict: dict[str, list] = {}
    mode = config.stream_mode if hasattr(config, "stream_mode") else "one"
    if getattr(config, "two_stream", False) and mode == "one":
        mode = "two"
    if mode == "one":
        X_windows: list[torch.Tensor] = []
    elif mode == "two":
        streamA_windows: list[torch.Tensor] = []
        streamB_windows: list[torch.Tensor] = []
    else:
        coords_windows: list[torch.Tensor] = []
        delta_windows: list[torch.Tensor] = []
        bone_windows: list[torch.Tensor] = []
        bone_delta_windows: list[torch.Tensor] = []

    batch_count = 0
    for data_df, meta_df, label_df in batches:
        if config.max_batches is not None and batch_count >= config.max_batches:
            break

        if mode == "one":
            window_tensor, frame_ranges = prepare_ctr_gcn_input(data_df, ordered_joints, config)
            if window_tensor.shape[0] == 0:
                batch_count += 1
                continue
        elif mode == "two":
            streamA_list, streamB_list, frame_ranges = prepare_ctr_gcn_input(data_df, ordered_joints, config)
            if len(streamA_list) == 0:
                batch_count += 1
                continue
        else:
            coords_list, delta_list, bone_list, bone_delta_list, frame_ranges = prepare_ctr_gcn_input(data_df, ordered_joints, config)
            if len(coords_list) == 0:
                batch_count += 1
                continue

        # Ensure label tracking per action
        for action in label_df.columns:
            if action not in y_dict:
                y_dict[action] = []

        for i, frame_range in enumerate(frame_ranges):
            center_frame = frame_range[len(frame_range) // 2]
            if center_frame not in label_df.index:
                continue
            label_row = label_df.loc[center_frame]
            if mode == "one":
                X_windows.append(window_tensor[i])
            elif mode == "two":
                streamA_windows.append(streamA_list[i])
                streamB_windows.append(streamB_list[i])
            else:
                coords_windows.append(coords_list[i])
                delta_windows.append(delta_list[i])
                bone_windows.append(bone_list[i])
                bone_delta_windows.append(bone_delta_list[i])
            for action in y_dict.keys():
                val = label_row[action] if action in label_row else np.nan
                y_dict[action].append(val)

        batch_count += 1

    if mode == "one" and len(X_windows) == 0:
        return model_dict
    if mode == "two" and len(streamA_windows) == 0:
        return model_dict
    if mode == "four" and len(coords_windows) == 0:
        return model_dict

    batch_size = 16
    if config.mode == "dev":
        epochs = 1
    elif config.mode == "validate":
        epochs = 3
    else:
        epochs = 5

    if mode == "one":
        X = torch.stack(X_windows, dim=0).to(device)
        in_channels_single = X.shape[1]
    elif mode == "two":
        X_streamA = torch.stack(streamA_windows, dim=0).to(device)
        X_streamB = torch.stack(streamB_windows, dim=0).to(device)
    else:
        X_coords = torch.stack(coords_windows, dim=0).to(device)
        X_delta = torch.stack(delta_windows, dim=0).to(device)
        X_bone = torch.stack(bone_windows, dim=0).to(device)
        X_bone_delta = torch.stack(bone_delta_windows, dim=0).to(device)

    batch_size = 16

    for action, labels in y_dict.items():
        y_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
        mask = ~torch.isnan(y_tensor)
        if mask.sum().item() == 0:
            continue
        y_action = y_tensor[mask].unsqueeze(1)

        if mode == "one":
            X_action = X[mask]
            model = CTRGCNMinimal(in_channels=in_channels_single, num_classes=1, adjacency=adjacency).to(device)
        elif mode == "two":
            X_action_A = X_streamA[mask]
            X_action_B = X_streamB[mask]
            model = CTRGCNTwoStream(
                adjacency=adjacency,
                in_channels_coords=X_action_A.shape[1],
                in_channels_delta=X_action_B.shape[1],
            ).to(device)
        else:
            X_action_coords = X_coords[mask]
            X_action_delta = X_delta[mask]
            X_action_bone = X_bone[mask]
            X_action_bone_delta = X_bone_delta[mask]
            model = CTRGCNFourStream(adjacency=adjacency).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()

        for _ in range(epochs):
            model.train()
            for start in range(0, len(y_action), batch_size):
                end = start + batch_size
                batch_y = y_action[start:end]
                if mode == "one":
                    batch_x = X_action[start:end]
                    logits = model(batch_x)
                elif mode == "two":
                    batch_x_A = X_action_A[start:end]
                    batch_x_B = X_action_B[start:end]
                    logits = model(batch_x_A, batch_x_B)
                else:
                    batch_x_coords = X_action_coords[start:end]
                    batch_x_delta = X_action_delta[start:end]
                    batch_x_bone = X_action_bone[start:end]
                    batch_x_bone_delta = X_action_bone_delta[start:end]
                    logits = model(batch_x_coords, batch_x_delta, batch_x_bone, batch_x_bone_delta)

                optimizer.zero_grad()
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        model_dict[action] = model

    return model_dict


validate_or_submit = 'stresstest' # 'validate' or 'submit' or 'stresstest'
verbose = True
cwd = Path.cwd()

class TrainOnSubsetClassifier(ClassifierMixin, BaseEstimator):
    """Fit estimator to a subset of the training data."""
    def __init__(self, estimator, n_samples):
        self.estimator = estimator
        self.n_samples = n_samples

    def fit(self, X, y):
        downsample = len(X) // self.n_samples
        downsample = max(downsample, 1)
        self.estimator.fit(np.array(X, copy=False)[::downsample],
                           np.array(y, copy=False)[::downsample])
        self.classes_ = self.estimator.classes_
        return self

    def predict_proba(self, X):
        if len(self.classes_) == 1:
            return np.full((len(X), 1), 1.0)
        probs = self.estimator.predict_proba(np.array(X))
        return probs
        
    def predict(self, X):
        return self.estimator.predict(np.array(X))
    
"""F Beta customized for the data format of the MABe challenge."""

import json

from collections import defaultdict

import pandas as pd
import polars as pl


class HostVisibleError(Exception):
    pass


def single_lab_f1(lab_solution: pl.DataFrame, lab_submission: pl.DataFrame, beta: float = 1) -> float:
    label_frames: defaultdict[str, set[int]] = defaultdict(set) # key is video/agent/target/action from solution
    prediction_frames: defaultdict[str, set[int]] = defaultdict(set) # key is video/agent/target/action from submission

    for row in lab_solution.to_dicts():
        label_frames[row['label_key']].update(range(row['start_frame'], row['stop_frame']))

    for video in lab_solution['video_id'].unique():
        active_labels: str = lab_solution.filter(pl.col('video_id') == video)['behaviors_labeled'].first()  # ty: ignore
        active_labels: set[str] = set(json.loads(active_labels)) # set of agent,target,action from solution
        predicted_mouse_pairs: defaultdict[str, set[int]] = defaultdict(set) # key is agent,target from submission

        for row in lab_submission.filter(pl.col('video_id') == video).to_dicts(): # every submission row is converted to a dict
            # Since the labels are sparse, we can't evaluate prediction keys not in the active labels.
            if ','.join([str(row['agent_id']), str(row['target_id']), row['action']]) not in active_labels:
                # print(f'ignoring {video}', ','.join([str(row['agent_id']), str(row['target_id']), row['action']]), active_labels)
                continue # these submission rows are ignored
           
            new_frames = set(range(row['start_frame'], row['stop_frame']))
            # Ignore truly redundant predictions.
            new_frames = new_frames.difference(prediction_frames[row['prediction_key']])
            prediction_pair = ','.join([str(row['agent_id']), str(row['target_id'])])
            if predicted_mouse_pairs[prediction_pair].intersection(new_frames):
                # A single agent can have multiple targets per frame (ex: evading all other mice) but only one action per target per frame.
                raise HostVisibleError('Multiple predictions for the same frame from one agent/target pair')
            prediction_frames[row['prediction_key']].update(new_frames)
            predicted_mouse_pairs[prediction_pair].update(new_frames)

    tps = defaultdict(int) # key is action
    fns = defaultdict(int) # key is action
    fps = defaultdict(int) # key is action
    for key, pred_frames in prediction_frames.items():
        action = key.split('_')[-1]
        matched_label_frames = label_frames[key]
        tps[action] += len(pred_frames.intersection(matched_label_frames))
        fns[action] += len(matched_label_frames.difference(pred_frames))
        fps[action] += len(pred_frames.difference(matched_label_frames))

    distinct_actions = set()
    for key, frames in label_frames.items():
        action = key.split('_')[-1]
        distinct_actions.add(action)
        if key not in prediction_frames:
            fns[action] += len(frames)

    action_f1s = []
    for action in distinct_actions:
        # print(f"{tps[action]:8} {fns[action]:8} {fps[action]:8}")
        if tps[action] + fns[action] + fps[action] == 0:
            action_f1s.append(0)
        else:
            action_f1s.append((1 + beta**2) * tps[action] / ((1 + beta**2) * tps[action] + beta**2 * fns[action] + fps[action]))
    return sum(action_f1s) / len(action_f1s)


def mouse_fbeta(solution: pd.DataFrame, submission: pd.DataFrame, beta: float = 1) -> float:
    """
    Doctests:
    >>> solution = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 10, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ... ])
    >>> submission = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 10},
    ... ])
    >>> mouse_fbeta(solution, submission)
    1.0

    >>> solution = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 10, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ... ])
    >>> submission = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'mount', 'start_frame': 0, 'stop_frame': 10}, # Wrong action
    ... ])
    >>> mouse_fbeta(solution, submission)
    0.0

    >>> solution = pd.DataFrame([
    ...     {'video_id': 123, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 9, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ...     {'video_id': 123, 'agent_id': 1, 'target_id': 2, 'action': 'mount', 'start_frame': 15, 'stop_frame': 24, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ... ])
    >>> submission = pd.DataFrame([
    ...     {'video_id': 123, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 9},
    ... ])
    >>> "%.12f" % mouse_fbeta(solution, submission)
    '0.500000000000'

    >>> solution = pd.DataFrame([
    ...     {'video_id': 123, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 9, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ...     {'video_id': 123, 'agent_id': 1, 'target_id': 2, 'action': 'mount', 'start_frame': 15, 'stop_frame': 24, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ...     {'video_id': 345, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 9, 'lab_id': 2, 'behaviors_labeled': '["1,2,attack"]'},
    ...     {'video_id': 345, 'agent_id': 1, 'target_id': 2, 'action': 'mount', 'start_frame': 15, 'stop_frame': 24, 'lab_id': 2, 'behaviors_labeled': '["1,2,attack"]'},
    ... ])
    >>> submission = pd.DataFrame([
    ...     {'video_id': 123, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 9},
    ... ])
    >>> "%.12f" % mouse_fbeta(solution, submission)
    '0.250000000000'

    >>> # Overlapping solution events, one prediction matching both.
    >>> solution = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 10, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 10, 'stop_frame': 20, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ... ])
    >>> submission = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 20},
    ... ])
    >>> mouse_fbeta(solution, submission)
    1.0

    >>> solution = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 10, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 30, 'stop_frame': 40, 'lab_id': 1, 'behaviors_labeled': '["1,2,attack"]'},
    ... ])
    >>> submission = pd.DataFrame([
    ...     {'video_id': 1, 'agent_id': 1, 'target_id': 2, 'action': 'attack', 'start_frame': 0, 'stop_frame': 40},
    ... ])
    >>> mouse_fbeta(solution, submission)
    0.6666666666666666
    """
    if len(solution) == 0 or len(submission) == 0:
        raise ValueError('Missing solution or submission data')

    expected_cols = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']

    for col in expected_cols:
        if col not in solution.columns:
            raise ValueError(f'Solution is missing column {col}')
        if col not in submission.columns:
            raise ValueError(f'Submission is missing column {col}')

    solution: pl.DataFrame = pl.DataFrame(solution)
    submission: pl.DataFrame = pl.DataFrame(submission)
    assert (solution['start_frame'] <= solution['stop_frame']).all()
    assert (submission['start_frame'] <= submission['stop_frame']).all()
    solution_videos = set(solution['video_id'].unique())
    # Need to align based on video IDs as we can't rely on the row IDs for handling public/private splits.
    submission = submission.filter(pl.col('video_id').is_in(solution_videos))

    solution = solution.with_columns(
        pl.concat_str(
            [
                pl.col('video_id').cast(pl.Utf8),
                pl.col('agent_id').cast(pl.Utf8),
                pl.col('target_id').cast(pl.Utf8),
                pl.col('action'),
            ],
            separator='_',
        ).alias('label_key'),
    )
    submission = submission.with_columns(
        pl.concat_str(
            [
                pl.col('video_id').cast(pl.Utf8),
                pl.col('agent_id').cast(pl.Utf8),
                pl.col('target_id').cast(pl.Utf8),
                pl.col('action'),
            ],
            separator='_',
        ).alias('prediction_key'),
    )

    lab_scores = []
    for lab in solution['lab_id'].unique():
        lab_solution = solution.filter(pl.col('lab_id') == lab).clone()
        lab_videos = set(lab_solution['video_id'].unique())
        lab_submission = submission.filter(pl.col('video_id').is_in(lab_videos)).clone()
        lab_scores.append(single_lab_f1(lab_solution, lab_submission, beta=beta))

    return sum(lab_scores) / len(lab_scores)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, beta: float = 1) -> float:
    """
    F1 score for the MABe Challenge
    """
    solution = solution.drop(row_id_column_name, axis='columns', errors='ignore')
    submission = submission.drop(row_id_column_name, axis='columns', errors='ignore')
    return mouse_fbeta(solution, submission, beta=beta)

train = pd.read_csv(cwd / 'Data' / 'train.csv')
train['n_mice'] = 4 - train[['mouse1_strain', 'mouse2_strain', 'mouse3_strain', 'mouse4_strain']].isna().sum(axis=1)
train_without_mabe22 = train.query("~ lab_id.str.startswith('MABe22_')")

test = pd.read_csv(cwd / 'Data' / 'test.csv')

# labs = list(np.unique(train.lab_id))

body_parts_tracked_list = list(np.unique(train.body_parts_tracked))

# behaviors = list(train.behaviors_labeled.drop_duplicates().dropna())
# behaviors = sorted(list({b.replace("'", "") for bb in behaviors for b in json.loads(bb)}))
# behaviors = [b.split(',') for b in behaviors]
# behaviors = pd.DataFrame(behaviors, columns=['agent', 'target', 'action'])


def create_solution_df(dataset):
    """Create the solution dataframe for validating out-of-fold predictions.

    From https://www.kaggle.com/code/ambrosm/mabe-validated-baseline-without-machine-learning/
    
    Parameters:
    dataset: (a subset of) the train dataframe
    
    Return values:
    solution: solution dataframe in the correct format for the score() function
    """
    solution = []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
    
        # Load annotation file
        lab_id = row['lab_id']
        if lab_id.startswith('MABe22'): continue
        video_id = row['video_id']
        path = f"{cwd}/Data/train_annotation/{lab_id}/{video_id}.parquet"
        try:
            annot = pd.read_parquet(path)
        except FileNotFoundError:
            # MABe22 and one more training file lack annotations.
            if verbose: print(f"No annotations for {path}")
            continue
    
        # Add all annotations to the solution
        annot['lab_id'] = lab_id
        annot['video_id'] = video_id
        annot['behaviors_labeled'] = row['behaviors_labeled']
        annot['target_id'] = np.where(annot.target_id != annot.agent_id, annot['target_id'].apply(lambda s: f"mouse{s}"), 'self')
        annot['agent_id'] = annot['agent_id'].apply(lambda s: f"mouse{s}")
        solution.append(annot)
    
    solution = pd.concat(solution)
    return solution

if validate_or_submit == 'validate':
    solution = create_solution_df(train_without_mabe22)


"""
# Stress testing with unusual inputs

After submission, this notebook will see a test set that it has never seen before. 
If the notebook crashes, debugging will be hard. 
It's better to stress-test the notebook before the submission by giving it some unusual inputs. 
The following hidden cell generate synthetic data with missing values, excessively long videos and so on.
"""

if validate_or_submit == 'stresstest':
    n_videos_per_lab = 2
    
    try:
        os.mkdir(f"stresstest_tracking")
    except FileExistsError:
        pass
    
    rng = np.random.default_rng()
    stresstest = pd.concat(
        [train.query("video_id == 1459695188")] # long video from BoisterousParrot
        + [df.sample(min(n_videos_per_lab, len(df)), random_state=1) for (_, df) in train.groupby('lab_id')])
    for _, row in tqdm(stresstest.iterrows(), total=len(stresstest)):
        lab_id = row['lab_id']
        video_id = row['video_id']
        
        # Load video
        path = f"{cwd}/Data/train_tracking/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
    
        if video_id == 1459695188: # long video from BoisterousParrot
            vid = pd.concat([vid] * 3) # provoke out of memory (5 is too much)
            vid['video_frame'] = np.arange(len(vid))
    
        # Drop some complete frames
        dropped_frames = list(rng.choice(np.unique(vid.video_frame), size=100, replace=False))
        vid = vid.query("~ video_frame.isin(@dropped_frames)")
        
        # Drop a complete bodypart
        if rng.uniform() < 0.2:
            dropped_bodypart = rng.choice(np.unique(vid.bodypart), size=1, replace=False)[0]
            vid = vid.query("bodypart != @dropped_bodypart")
        
        # Drop a mouse
        if rng.uniform() < 0.1:
            vid = vid.query("mouse_id != 1")
        
        # Drop random bodyparts from random frames
        if rng.uniform() < 0.7:
            mask = np.ones(len(vid), dtype=bool)
            mask[:int(0.4 * len(mask))] = False
            rng.shuffle(mask)
            vid = vid[mask]
    
        # Set random coordinates of bodyparts to nan
        if rng.uniform() < 0.7:
            mask = np.ones(len(vid), dtype=bool)
            mask[:int(0.2 * len(mask))] = False
            rng.shuffle(mask)
            vid.loc[:, 'x'] = np.where(mask, np.nan, vid.loc[:, 'x'])
            rng.shuffle(mask)
            vid.loc[:, 'y'] = np.where(mask, np.nan, vid.loc[:, 'y'])
    
        # Save the video
        try:
            os.mkdir(f"stresstest_tracking/{lab_id}")
        except FileExistsError:
            pass
        new_path = f"stresstest_tracking/{lab_id}/{video_id}.parquet"
        vid.to_parquet(new_path)

'''
# Challenge 1: Modeling for variable-sized sets of mice

The first challenge we're going to solve is the fact that we have a variable number of mice (2, 3 or 4), 
and that the labeled behaviors apply either to one mouse or a pair of mice.

The following function, `generate_mouse_data()`, solves this challenge. 
It transforms the dataset into batches. There are single-mouse batches and mouse-pair batches. 
Every single-mouse batch has data of only one mouse, every mouse-pair batch has data of exactly two mice. 
A single video frame can end up in several batches. 
If the frame has two visible mice, it can be part of four batches:
- a single-mouse batch for individual behavior of mouse 1
- a single-mouse batch for individual behavior of mouse 2
- a mouse-pair batch for actions of mouse 1 with mouse 2 as target
- a mouse-pair batch for actions of mouse 2 with mouse 1 as target

The features (`data`) will consist of coordinates of body parts; the metadata (`meta`) 
will specify which mouse is / which mice are involved.
'''

drop_body_parts =  ['headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
                    'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright', 
                    'spine_1', 'spine_2',
                    'tail_middle_1', 'tail_middle_2', 'tail_midpoint']

def generate_mouse_data(dataset, traintest, traintest_directory=None, generate_single=True, generate_pair=True, config: CTRGCNConfig | None = None):
    """Generate batches of data in coordinate representation.

    The batches have variable length, and every batch can have other columns
    for the labels, depending on what behaviors
    were labeled for the batch.

    Every video can produce zero, one or two batches.
    
    Parameters
    ----------
    dataset: (subset of) train.csv or test.csv dataframe
    traintest: either 'train' or 'test'

    Yields
    ------
    switch: either 'single' or 'pair'
    data: dataframe containing coordinates of the body parts of a single mouse or of a pair of mice
    meta: dataframe with columns ['video_id', 'agent_id', 'target_id', 'video_frame']
    label: dataframe with labels (0, 1), one column per action, only if traintest == 'train'
    actions: list of actions to be predicted for this batch, only if traintest == 'test'
    """
    assert traintest in ['train', 'test']
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
    # CTR-GCN config limits allow dev/validate/submit modes to reduce data volume without changing default behavior.
    video_count = 0
    batch_count = 0
    for _, row in dataset.iterrows():
        if config is not None and config.max_videos is not None and video_count >= config.max_videos:
            break
        
        # Load the video and pivot it sn that one frame = one row
        lab_id = row.lab_id
        if lab_id.startswith('MABe22'): continue
        video_id = row.video_id

        if type(row.behaviors_labeled) != str:
            # We cannot use videos without labeled behaviors
            print('No labeled behaviors:', lab_id, video_id, type(row.behaviors_labeled), row.behaviors_labeled)
            continue

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
        if len(np.unique(vid.bodypart)) > 5:
            vid = vid.query("~ bodypart.isin(@drop_body_parts)")
        pvid = vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])
        if pvid.isna().any().any():
            if verbose and traintest == 'test': print('video with missing values', video_id, traintest, len(vid), 'frames')
        else:
            if verbose and traintest == 'test': print('video with all values', video_id, traintest, len(vid), 'frames')
        del vid
        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T # mouse_id, body_part, xy
        pvid /= row.pix_per_cm_approx # convert to cm

        # Determine the behaviors of this video
        vid_behaviors = json.loads(row.behaviors_labeled)
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])
        
        # Load the annotations
        if traintest == 'train':
            try:
                annot = pd.read_parquet(path.replace('train_tracking', 'train_annotation'))
            except FileNotFoundError:
                # MABe22 and one more training file lack annotations.
                # We simply drop these videos.
                continue

        video_count += 1

        # Create the single_mouse dataframes: single_mouse, single_mouse_label and single_mouse_meta
        if generate_single:
            vid_behaviors_subset = vid_behaviors.query("target == 'self'") # single-mouse behaviors of this video
            for mouse_id_str in np.unique(vid_behaviors_subset.agent):
                if config is not None and config.max_batches is not None and batch_count >= config.max_batches:
                    return
                try:
                    mouse_id = int(mouse_id_str[-1])
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("agent == @mouse_id_str").action)
                    single_mouse = pvid.loc[:, mouse_id]
                    assert len(single_mouse) == len(pvid)
                    single_mouse_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': mouse_id_str,
                        'target_id': 'self',
                        'video_frame': single_mouse.index
                    })
                    if traintest == 'train':
                        single_mouse_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=single_mouse.index)
                        annot_subset = annot.query("(agent_id == @mouse_id) & (target_id == @mouse_id)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            single_mouse_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'single', single_mouse, single_mouse_meta, single_mouse_label
                        batch_count += 1
                    else:
                        if verbose: print('- test single', video_id, mouse_id)
                        yield 'single', single_mouse, single_mouse_meta, vid_agent_actions
                        batch_count += 1
                except KeyError:
                    pass # If there is no data for the selected agent mouse, we skip the mouse.

        # Create the mouse_pair dataframes: mouse_pair, mouse_label and mouse_meta
        if generate_pair:
            vid_behaviors_subset = vid_behaviors.query("target != 'self'")
            if len(vid_behaviors_subset) > 0:
                for agent, target in itertools.permutations(np.unique(pvid.columns.get_level_values('mouse_id')), 2): # int8
                    if config is not None and config.max_batches is not None and batch_count >= config.max_batches:
                        return
                    agent_str = f"mouse{agent}"
                    target_str = f"mouse{target}"
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("(agent == @agent_str) & (target == @target_str)").action)
                    mouse_pair = pd.concat([pvid[agent], pvid[target]], axis=1, keys=['A', 'B'])
                    assert len(mouse_pair) == len(pvid)
                    mouse_pair_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': agent_str,
                        'target_id': target_str,
                        'video_frame': mouse_pair.index
                    })
                    if traintest == 'train':
                        mouse_pair_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=mouse_pair.index)
                        annot_subset = annot.query("(agent_id == @agent) & (target_id == @target)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            mouse_pair_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'pair', mouse_pair, mouse_pair_meta, mouse_pair_label
                        batch_count += 1
                    else:
                        if verbose: print('- test pair', video_id, agent, target)
                        yield 'pair', mouse_pair, mouse_pair_meta, vid_agent_actions
                        batch_count += 1

"""
# Challenge 2: Multiclass prediction with missing labels

This competition is a multi-class classification task. 
For every video_id/video_frame/agent/target combination, we may predict at most one of several actions. 
Every action is a class, and 'no-action' is an additional class.

We cannot use a standard multi-class estimator from scikit-learn 
because many values in the labels of our dataset are missing. 
For this reason, we train a binary classifier for every action, 
omitting the samples for which the target is unknown. 
Every binary classificator predicts a probability, 
and for the multiclass prediction we predict the class with the highest binary probability, 
if this probability is above a threshold; otherwise, we predict no action.
"""

# Make the multi-class prediction
def predict_multiclass(pred, meta):
    """Derive multiclass predictions from a set of binary predictions.
    
    Parameters
    pred: dataframe of predicted binary probabilities, shape (n_samples, n_actions), index doesn't matter
    meta: dataframe with columns ['video_id', 'agent_id', 'target_id', 'video_frame'], index doesn't matter
    """
    # Find the most probable class, but keep it only if its probability is above the threshold
    threshold = 0.27
    ama = np.argmax(pred, axis=1)
    ama = np.where(pred.max(axis=1) >= threshold, ama, -1)
    ama = pd.Series(ama, index=meta.video_frame)
    # Keep only start and stop frames
    changes_mask = (ama != ama.shift(1)).values
    ama_changes = ama[changes_mask]
    meta_changes = meta[changes_mask]
    # mask selects the start frames
    mask = ama_changes.values >= 0 # start of action
    mask[-1] = False
    submission_part = pd.DataFrame({
        'video_id': meta_changes['video_id'][mask].values,
        'agent_id': meta_changes['agent_id'][mask].values,
        'target_id': meta_changes['target_id'][mask].values,
        'action': pred.columns[ama_changes[mask].values],
        'start_frame': ama_changes.index[mask],
        'stop_frame': ama_changes.index[1:][mask[:-1]]
    })
    stop_video_id = meta_changes['video_id'][1:][mask[:-1]].values
    stop_agent_id = meta_changes['agent_id'][1:][mask[:-1]].values
    stop_target_id = meta_changes['target_id'][1:][mask[:-1]].values
    for i in range(len(submission_part)):
        video_id = submission_part.video_id.iloc[i]
        agent_id = submission_part.agent_id.iloc[i]
        target_id = submission_part.target_id.iloc[i]
        if stop_video_id[i] != video_id or stop_agent_id[i] != agent_id or stop_target_id[i] != target_id:
            new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
            submission_part.iat[i, submission_part.columns.get_loc('stop_frame')] = new_stop_frame
    assert (submission_part.stop_frame > submission_part.start_frame).all(), 'stop <= start'
    if verbose: print('  actions found:', len(submission_part))
    return submission_part

'''
# Challenge 3: Transforming coordinates to an invariant representation

The body part of the mice are given in cartesian coordinates. 
If the mice show some behavior at varying positions and with varying spatial orientation, 
cartesian coordinates are an inadequate representation. 
Our feature engineering transforms the coordinates to distances between body parts. 
Distances are invariant under translation and rotation.

For a single mouse, the distances indicate whether and how much it turns its head, 
shoulders, hip and tail left or right. 
For a pair of mice, the distances indicate how far the head of the first mouse is near 
what part of the second one, and what body parts either mouse turns towards or away from the other one.
'''

def transform_single(single_mouse, body_parts_tracked):
    """Transform from cartesian coordinates to distance representation.

    Parameters:
    single_mouse: dataframe with coordinates of the body parts of one mouse
                  shape (n_samples, n_body_parts * 2)
                  two-level MultiIndex on columns
    body_parts_tracked: list of body parts
    """
    available_body_parts = single_mouse.columns.get_level_values(0)
    X = pd.DataFrame({
            f"{part1}+{part2}": np.square(single_mouse[part1] - single_mouse[part2]).sum(axis=1, skipna=False)
            for part1, part2 in itertools.combinations(body_parts_tracked, 2) if part1 in available_body_parts and part2 in available_body_parts
        })
    X = X.reindex(columns=[f"{part1}+{part2}" for part1, part2 in itertools.combinations(body_parts_tracked, 2)], copy=False)

    if 'ear_left' in single_mouse.columns and 'ear_right' in single_mouse.columns and 'tail_base' in single_mouse.columns:
        shifted = single_mouse[['ear_left', 'ear_right', 'tail_base']].shift(10)
        X = pd.concat([
            X, 
            pd.DataFrame({
                'speed_left': np.square(single_mouse['ear_left'] - shifted['ear_left']).sum(axis=1, skipna=False),
                'speed_right': np.square(single_mouse['ear_right'] - shifted['ear_right']).sum(axis=1, skipna=False),
                'speed_left2': np.square(single_mouse['ear_left'] - shifted['tail_base']).sum(axis=1, skipna=False),
                'speed_right2': np.square(single_mouse['ear_right'] - shifted['tail_base']).sum(axis=1, skipna=False),
            })
        ], axis=1)
    return X

def transform_pair(mouse_pair, body_parts_tracked):
    """Transform from cartesian coordinates to distance representation.

    Parameters:
    mouse_pair: dataframe with coordinates of the body parts of two mice
                  shape (n_samples, 2 * n_body_parts * 2)
                  three-level MultiIndex on columns
    body_parts_tracked: list of body parts
    """
    # drop_body_parts =  ['ear_left', 'ear_right',
    #                     'headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
    #                     'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright', 
    #                     'tail_midpoint']
    # if len(body_parts_tracked) > 5:
    #     body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
    available_body_parts_A = mouse_pair['A'].columns.get_level_values(0)
    available_body_parts_B = mouse_pair['B'].columns.get_level_values(0)
    X = pd.DataFrame({
            f"12+{part1}+{part2}": np.square(mouse_pair['A'][part1] - mouse_pair['B'][part2]).sum(axis=1, skipna=False)
            for part1, part2 in itertools.product(body_parts_tracked, repeat=2) if part1 in available_body_parts_A and part2 in available_body_parts_B
        })
    X = X.reindex(columns=[f"12+{part1}+{part2}" for part1, part2 in itertools.product(body_parts_tracked, repeat=2)], copy=False)

    if ('A', 'ear_left') in mouse_pair.columns and ('B', 'ear_left') in mouse_pair.columns:
        shifted_A = mouse_pair['A']['ear_left'].shift(10)
        shifted_B = mouse_pair['B']['ear_left'].shift(10)
        X = pd.concat([
            X,
            pd.DataFrame({
                'speed_left_A': np.square(mouse_pair['A']['ear_left'] - shifted_A).sum(axis=1, skipna=False),
                'speed_left_AB': np.square(mouse_pair['A']['ear_left'] - shifted_B).sum(axis=1, skipna=False),
                'speed_left_B': np.square(mouse_pair['B']['ear_left'] - shifted_B).sum(axis=1, skipna=False),
            })
        ], axis=1)
    return X

'''
# Cross-validation

We're now almost ready to cross-validate our models. 

The following function gets as input
- a binary classification model
- a 2d array of features (i.e., distances between body parts); after we have dealt with variable-sized mouse sets (challenge 1) and variable-sized bodyparts sets (challenge 5), this array is rectangular.
- a 2d array of binary labels, some elements of which may be missing
- a 2d array of metadata so that we can match the predictions with the original video_id, agent, target and video_frame

It first computes out-of-fold predictions with a set of binary classifiers 
and then transforms these binary predictions into a multiclass prediction (see above).
'''

threshold = 0.27
f1_list = []
def cross_validate_classifier(binary_classifier, X, label, meta):
    """Cross-validate a binary classifier per action and a multi-class classifier over all actions.

    Parameters
    ----------
    binary_classifier: classifier with predict_proba
    X: 2d array-like (distance representation) of shape (n_samples, n_features)
    label: dataframe with binary targets (one column per action, may have missing values), index doesn't matter
    meta: dataframe with columns ['video_id', 'agent_id', 'target_id', 'video_frame'], index doesn't matter

    Output
    ------
    appends to f1_list (binary) and submission_list (multi-class)
    
    """
    # Cross-validate a binary classifier for every action
    oof = pd.DataFrame(index=meta.video_frame) # will get a column per action
    for action in label.columns:
        # Filter for samples (video frames) with a defined target (i.e., target is not nan)
        action_mask = ~ label[action].isna().values
        X_action = X[action_mask]
        y_action = label[action][action_mask].values.astype(int)
        p = y_action.mean()
        baseline_score = p / (1 + p)
        groups_action = meta.video_id[action_mask] # ensure validation has unseen videos
        if len(np.unique(groups_action)) < 5:
            continue # GroupKFold would fail with fewer than n_splits groups

        if not (y_action == 0).all():
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                # Number of classes in training fold (1) does not match total number of classes (2)
                oof_action = cross_val_predict(binary_classifier, X_action, y_action, groups=groups_action, cv=GroupKFold(), method='predict_proba')
            oof_action = oof_action[:, 1]
        else:
            oof_action = np.zeros(len(y_action))
        f1 = f1_score(y_action, (oof_action >= threshold), zero_division=0)
        ch = '>' if f1 > baseline_score else '=' if f1 == baseline_score else '<'
        print(f"  F1: {f1:.3f} {ch} ({baseline_score:.3f}) {action}")
        f1_list.append((body_parts_tracked_str, action, f1)) # type: ignore
        oof_column = np.zeros(len(label))
        oof_column[action_mask] = oof_action
        oof[action] = oof_column

    # Make the multi-class prediction
    submission_part = predict_multiclass(oof, meta)
    submission_list.append(submission_part) # type: ignore


def load_ctr_gcn_models(
    model_dir: str,
    actions: list[str],
    adjacency: np.ndarray,
    config: CTRGCNConfig,
    device: str = "cpu",
) -> dict[str, nn.Module]:
    """
    Load trained CTR-GCN models from disk.

    Parameters
    ----------
    model_dir : str
        Directory containing "{action}.pt" weight files.
    actions : list[str]
        Action names to load models for.
    adjacency : np.ndarray
        Adjacency matrix (V,V) for the joints.
    config : CTRGCNConfig
        Configuration specifying stream_mode and channel sizes.
    device : str
        "cpu", "cuda", or "mps".

    Returns
    -------
    model_dict : dict[str, nn.Module]
        Loaded models, all in eval mode and moved to the correct device.
    """
    model_dict: dict[str, nn.Module] = {}

    mode = getattr(config, "stream_mode", "one")
    if getattr(config, "two_stream", False) and mode == "one":
        mode = "two"

    for action in actions:
        if mode == "one":
            model = CTRGCNMinimal(
                in_channels=config.in_channels_single_stream,
                num_classes=1,
                adjacency=adjacency,
            )
        elif mode == "two":
            model = CTRGCNTwoStream(
                adjacency=adjacency,
                in_channels_coords=config.in_channels_streamA,
                in_channels_delta=config.in_channels_streamB,
            )
        else:
            model = CTRGCNFourStream(adjacency=adjacency)

        weight_path = os.path.join(model_dir, f"{action}.pt")
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        model_dict[action] = model

    return model_dict


def submit_ctr_gcn(
    body_parts_tracked_str: str,
    switch_tr: str,
    model_dict: dict[str, nn.Module],
    config: CTRGCNConfig,
    device: str = "cpu",
) -> None:
    """
    Generate submission parts using pre-trained CTR-GCN models
    (inference-only, no training). Appends submission_part DataFrames
    to the global submission_list, exactly like the LightGBM submit().

    Parameters
    ----------
    body_parts_tracked_str : str
        JSON list of body parts tracked.
    switch_tr : str
        "single" or "pair".
    model_dict : dict[str, nn.Module]
        Maps action → pretrained CTR-GCN model.
    config : CTRGCNConfig
        Contains stream_mode, use_delta, bone flags, etc.
    device : str
        "cpu", "cuda", or "mps".
    """
    body_parts_tracked = json.loads(body_parts_tracked_str)

    if validate_or_submit == "submit":
        test_subset = test[test.body_parts_tracked == body_parts_tracked_str]
        generator = generate_mouse_data(
            test_subset,
            "test",
            generate_single=(switch_tr == "single"),
            generate_pair=(switch_tr == "pair"),
            config=config,
        )
    else:
        test_subset = stresstest.query("body_parts_tracked == @body_parts_tracked_str")
        generator = generate_mouse_data(
            test_subset,
            "test",
            traintest_directory="stresstest_tracking",
            generate_single=(switch_tr == "single"),
            generate_pair=(switch_tr == "pair"),
            config=config,
        )

    if verbose:
        print(f"n_videos: {len(test_subset)}")

    ordered_joints, adjacency = get_ordered_joints_and_adjacency(body_parts_tracked)

    mode = getattr(config, "stream_mode", "one")
    if getattr(config, "two_stream", False) and mode == "one":
        mode = "two"

    for switch_te, data_te, meta_te, actions_te in generator:
        assert switch_te == switch_tr

        actions_available = [a for a in actions_te if a in model_dict]
        if not actions_available:
            if verbose:
                print(f"  No CTR-GCN models for actions {actions_te}")
            continue

        if mode == "one":
            window_tensor, frame_ranges = prepare_ctr_gcn_input(data_te, ordered_joints, config)
            if window_tensor.shape[0] == 0:
                continue
            X = window_tensor.to(device)
        elif mode == "two":
            streamA_list, streamB_list, frame_ranges = prepare_ctr_gcn_input(data_te, ordered_joints, config)
            if len(streamA_list) == 0:
                continue
            X_streamA = torch.stack(streamA_list, dim=0).to(device)
            X_streamB = torch.stack(streamB_list, dim=0).to(device)
        else:
            coords_list, delta_list, bone_list, bone_delta_list, frame_ranges = prepare_ctr_gcn_input(data_te, ordered_joints, config)
            if len(coords_list) == 0:
                continue
            X_coords = torch.stack(coords_list, dim=0).to(device)
            X_delta = torch.stack(delta_list, dim=0).to(device)
            X_bone = torch.stack(bone_list, dim=0).to(device)
            X_bone_delta = torch.stack(bone_delta_list, dim=0).to(device)

        if verbose and len(frame_ranges) == 0:
            print("  No frame ranges produced.")
            continue

        frame_values = meta_te.video_frame.values
        frame_to_idx = {f: i for i, f in enumerate(frame_values)}
        n_frames = len(frame_values)
        n_actions = len(actions_available)

        sum_probs = np.zeros((n_frames, n_actions), dtype=np.float32)
        counts = np.zeros((n_frames, n_actions), dtype=np.float32)

        for action_idx, action_name in enumerate(actions_available):
            model = model_dict[action_name]

            with torch.no_grad():
                if mode == "one":
                    logits = model(X)
                elif mode == "two":
                    logits = model(X_streamA, X_streamB)
                else:
                    logits = model(X_coords, X_delta, X_bone, X_bone_delta)

            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

            for w_idx, frames in enumerate(frame_ranges):
                p = float(probs[w_idx])
                for f in frames:
                    fi = frame_to_idx.get(f)
                    if fi is None:
                        continue
                    sum_probs[fi, action_idx] += p
                    counts[fi, action_idx] += 1.0

        counts_safe = counts.copy()
        counts_safe[counts_safe == 0] = 1.0
        pred_array = sum_probs / counts_safe

        pred_df = pd.DataFrame(
            pred_array,
            index=meta_te.video_frame,
            columns=actions_available,
        )

        submission_part = predict_multiclass(pred_df, meta_te)
        submission_list.append(submission_part)


"""
# =============================
# EXAMPLE: TRAIN → SAVE WEIGHTS
# =============================
# config = CTRGCNConfig(mode="validate")
# batches = [...]  # from generate_mouse_data
# ordered_joints, adjacency = get_ordered_joints_and_adjacency(body_parts_tracked)
# model_dict = train_ctr_gcn_models(batches, ordered_joints, adjacency, config)

# os.makedirs("models", exist_ok=True)
# for action, model in model_dict.items():
#     torch.save(model.state_dict(), f"models/{action}.pt")

# =============================
# EXAMPLE: LOAD WEIGHTS → INFERENCE
# =============================
# loaded_models = load_ctr_gcn_models(
#     "models/",
#     actions=list(model_dict.keys()),
#     adjacency=adjacency,
#     config=config,
#     device="cpu",
# )

# submit_ctr_gcn(
#     body_parts_tracked_str,
#     switch_tr,
#     loaded_models,
#     config,
#     device="cpu",
# )
"""

'''
# Challenge 4: A dataset that doesn't fit into memory

The competition dataset doesn't fit into memory as whole. 
The problem is exacerbated if we compute lots of distance in feature engineering. 
We tackle this challenge with the following measures:
- Training on a subset of the data: The training dataset is highly redundant. In videos taken with 30 frames per second, the difference from one frame to the next is small. We can well afford to subsample the training data.
- Processing the test data in batches: There is no need to have the full test dataset in memory at any time. (This decision has the drawback that the test data are read from disk several times.)
- It helps that we split all data by body_parts_tracked (see challenge 5 below). This way, we don't even need to have the full training dataset in memory.
'''

def submit(body_parts_tracked_str, switch_tr, binary_classifier, X_tr, label, meta):
    """Produce a submission file for the selected subset of the test data.

    Parameters
    ----------
    body_parts_tracked_str: subset of body parts for filtering the test set
    switch_tr: 'single' or 'pair'
    binary_classifier: classifier with predict_proba
    X_tr: training features as 2d array-like of shape (n_samples, n_features)
    label: dataframe with binary targets (one column per action, may have missing values), index doesn't matter
    meta: dataframe with columns ['video_id', 'agent_id', 'target_id', 'video_frame'], index doesn't matter

    Output
    ------
    appends to submission_list
    
    """
    # Fit a binary classifier for every action
    model_list = [] # will get a model per action
    for action in label.columns:
        # Filter for samples (video frames) with a defined target (i.e., target is not nan)
        action_mask = ~ label[action].isna().values
        y_action = label[action][action_mask].values.astype(int)

        if not (y_action == 0).all():
            model = clone(binary_classifier)
            model.fit(X_tr[action_mask], y_action)
            assert len(model.classes_) == 2
            model_list.append((action, model))

    # Compute test predictions in batches
    body_parts_tracked = json.loads(body_parts_tracked_str)
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
    if validate_or_submit == 'submit':
        test_subset = test[test.body_parts_tracked == body_parts_tracked_str]
        generator = generate_mouse_data(test_subset, 'test',
                                        generate_single=(switch_tr == 'single'), 
                                        generate_pair=(switch_tr == 'pair'))
    else:
        test_subset = stresstest.query("body_parts_tracked == @body_parts_tracked_str")
        generator = generate_mouse_data(test_subset, 'test',
                                        traintest_directory='stresstest_tracking',
                                        generate_single=(switch_tr == 'single'),
                                        generate_pair=(switch_tr == 'pair'))
    if verbose: print(f"n_videos: {len(test_subset)}")
    for switch_te, data_te, meta_te, actions_te in generator:
        assert switch_te == switch_tr
        try:
            # Transform from coordinate representation into distance representation
            if switch_te == 'single':
                X_te = transform_single(data_te, body_parts_tracked) # may raise KeyError
            else:
                X_te = transform_pair(data_te, body_parts_tracked) # may raise KeyError
            if verbose and len(X_te) == 0: print("ERROR: X_te is empty")
            del data_te
    
            # Compute binary predictions
            pred = pd.DataFrame(index=meta_te.video_frame) # will get a column per action
            for action, model in model_list:
                if action in actions_te:
                    pred[action] = model.predict_proba(X_te)[:, 1]
            del X_te
            # Compute multiclass predictions
            if pred.shape[1] != 0:
                submission_part = predict_multiclass(pred, meta_te)
                submission_list.append(submission_part) # type: ignore
            else: # this happens if there was no useful training data for the test actions
                if verbose: print(f"  ERROR: no useful training data")
        except KeyError:
            if verbose: print(f'  ERROR: KeyError because of missing bodypart ({switch_tr})')
            del data_te


submission_list = []
for section in range(1, len(body_parts_tracked_list)): # skip index 0 (MABe22)
    body_parts_tracked_str = body_parts_tracked_list[section]
    try:
        body_parts_tracked = json.loads(body_parts_tracked_str)
        print(f"{section}. Processing videos with {body_parts_tracked}")
        if len(body_parts_tracked) > 5:
            body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
    
        # We read all training data which match the body parts tracked
        train_subset = train[train.body_parts_tracked == body_parts_tracked_str]
        single_mouse_list = []
        single_mouse_label_list = []
        single_mouse_meta_list = []
        mouse_pair_list = []
        mouse_pair_label_list = []
        mouse_pair_meta_list = []
    
        for switch, data, meta, label in generate_mouse_data(train_subset, 'train'):
            if switch == 'single':
                single_mouse_list.append(data)
                single_mouse_meta_list.append(meta)
                single_mouse_label_list.append(label)
            else:
                mouse_pair_list.append(data)
                mouse_pair_meta_list.append(meta)
                mouse_pair_label_list.append(label)
    
        # Construct a binary classifier
        binary_classifier = make_pipeline(
            SimpleImputer(),
            TrainOnSubsetClassifier(
                lightgbm.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.03,
                    min_child_samples=40,
                    # early_stopping_round=10, 
                    verbose=-1),
                100000)
        )
    
        # Predict single-mouse actions
        if len(single_mouse_list) > 0:
            # Concatenate all batches
            # The concatenation will generate label dataframes with missing values.
            single_mouse = pd.concat(single_mouse_list)
            single_mouse_label = pd.concat(single_mouse_label_list)
            single_mouse_meta = pd.concat(single_mouse_meta_list)
            del single_mouse_list, single_mouse_label_list, single_mouse_meta_list
            assert len(single_mouse) == len(single_mouse_label)
            assert len(single_mouse) == len(single_mouse_meta)
            
            # Transform the coordinate representation into a distance representation for single_mouse
            X_tr = transform_single(single_mouse, body_parts_tracked)
            del single_mouse
            print(f"{X_tr.shape=}")
    
            if validate_or_submit == 'validate':
                cross_validate_classifier(binary_classifier, X_tr, single_mouse_label, single_mouse_meta)
            else:
                submit(body_parts_tracked_str, 'single', binary_classifier, X_tr, single_mouse_label, single_mouse_meta)
            del X_tr
                
        # Predict mouse-pair actions
        if len(mouse_pair_list) > 0:
            # Concatenate all batches
            # The concatenation will generate label dataframes with missing values.
            mouse_pair = pd.concat(mouse_pair_list)
            mouse_pair_label = pd.concat(mouse_pair_label_list)
            mouse_pair_meta = pd.concat(mouse_pair_meta_list)
            del mouse_pair_list, mouse_pair_label_list, mouse_pair_meta_list
            assert len(mouse_pair) == len(mouse_pair_label)
            assert len(mouse_pair) == len(mouse_pair_meta)
        
            # Transform the coordinate representation into a distance representation for mouse_pair
            # Use a subset of body_parts_tracked to conserve memory
            X_tr = transform_pair(mouse_pair, body_parts_tracked)
            del mouse_pair
            print(f"{X_tr.shape=}")
    
            if validate_or_submit == 'validate':
                cross_validate_classifier(binary_classifier, X_tr, mouse_pair_label, mouse_pair_meta)
            else:
                submit(body_parts_tracked_str, 'pair', binary_classifier, X_tr, mouse_pair_label, mouse_pair_meta)
            del X_tr
                
    except Exception as e:
        print(f'***Exception*** {e}')
    print()


import numpy as np
import pandas as pd

def robustify(
    submission: pd.DataFrame,
    dataset: pd.DataFrame,
    traintest: str,
    traintest_directory: str | None = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Validate and repair a submission file according to competition rules.

    Rules:
    1. Drop rows where start_frame >= stop_frame.
    2. For each (video_id, agent_id, target_id), remove overlapping predictions.
    3. For videos with no predictions, generate rule-based filler predictions.

    Args:
        submission: DataFrame with columns:
            ['video_id','agent_id','target_id','action','start_frame','stop_frame']
        dataset: Competition dataset with video metadata.
        traintest: "train" or "test".
        traintest_directory: Base directory containing *_tracking parquet files.
        verbose: Print status logs.

    Returns:
        A cleaned submission DataFrame.
    """

    if traintest_directory is None:
        traintest_directory = (
            f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
        )

    # --- RULE 1: ensure start_frame < stop_frame ----------------------------------
    old_len = len(submission)
    submission = submission[submission.start_frame < submission.stop_frame]
    # --- FIX FRAME COLUMN TYPES -----------------------------------------------------
    for col in ['start_frame', 'stop_frame']:
        submission[col] = pd.to_numeric(submission[col], errors='coerce')

    bad_rows = submission[submission.start_frame.isna() | submission.stop_frame.isna()]
    if len(bad_rows):
        print(f"ERROR: Dropping {len(bad_rows)} rows with non-numeric frame values")
        submission = submission.dropna(subset=['start_frame','stop_frame'])

    if len(submission) != old_len:
        print("ERROR: Dropped frames with start >= stop")

    # --- RULE 2: ensure no overlapping predictions per (video_id, agent, target) ---
    cleaned_groups = []
    old_len = len(submission)

    for (_, group) in submission.groupby(['video_id', 'agent_id', 'target_id']):
        group = group.sort_values("start_frame")
        keep_mask = np.ones(len(group), dtype=bool)

        last_stop = -np.inf
        for i, (_, row) in enumerate(group.iterrows()):
            if row.start_frame < last_stop:
                keep_mask[i] = False
            else:
                last_stop = row.stop_frame

        cleaned_groups.append(group[keep_mask])

    submission = pd.concat(cleaned_groups, ignore_index=True)

    if len(submission) != old_len:
        print("ERROR: Dropped duplicate or overlapping frames")

    # --- RULE 3: fill missing videos ------------------------------------------------
    filler_rows = []

    for _, row in dataset.iterrows():
        lab_id = row['lab_id']
        if lab_id.startswith("MABe22"):  # Skip validation set
            continue

        video_id = row['video_id']

        # Already have predictions
        if (submission.video_id == video_id).any():
            continue

        if verbose:
            print(f"Video {video_id} has no predictions → filling.")

        # Load parquet
        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)

        # Parse behaviors
        behaviors_raw = eval(row['behaviors_labeled'])
        behaviors_raw = set(b.replace("'", "") for b in behaviors_raw)
        behaviors = pd.DataFrame(
            [b.split(',') for b in sorted(behaviors_raw)],
            columns=["agent_id", "target_id", "action"]
        )

        # Compute frame range
        start_frame = vid.video_frame.min()
        stop_frame = vid.video_frame.max() + 1
        total_frames = stop_frame - start_frame

        # Generate filler predictions
        for (agent, target), actions in behaviors.groupby(["agent_id", "target_id"]):
            n_actions = len(actions)
            batch_len = int(np.ceil(total_frames / n_actions))

            for i, (_, action_row) in enumerate(actions.iterrows()):
                batch_start = start_frame + i * batch_len
                batch_stop = min(batch_start + batch_len, stop_frame)

                filler_rows.append(
                    (video_id, agent, target, action_row["action"], batch_start, batch_stop)
                )

    if filler_rows:
        submission = pd.concat(
            [
                submission,
                pd.DataFrame(
                    filler_rows,
                    columns=['video_id', 'agent_id', 'target_id', 'action',
                             'start_frame', 'stop_frame']
                )
            ],
            ignore_index=True
        )
        print("ERROR: Filled missing videos")

    return submission.reset_index(drop=True)


if validate_or_submit == 'validate':
    # Score the oof predictions with the competition scoring function
    submission = pd.concat(submission_list)
    submission_robust = robustify(submission, train, 'train')
    print(f"# OOF score with competition metric: {score(solution, submission_robust, ''):.4f}")

    f1_df = pd.DataFrame(f1_list, columns=['body_parts_tracked_str', 'action', 'binary F1 score'])
    print(f"# Average of {len(f1_df)} binary F1 scores {f1_df['binary F1 score'].mean():.4f}")
    # with pd.option_context('display.max_rows', 500):
    #     display(f1_df)


if validate_or_submit != 'validate':
    if len(submission_list) > 0:
        submission = pd.concat(submission_list)
    else:
        submission = pd.DataFrame(
            dict(
                video_id=438887472,
                agent_id='mouse1',
                target_id='self',
                action='rear',
                start_frame='278',
                stop_frame='500'
            ), index=[44])
    if validate_or_submit == 'submit':
        submission_robust = robustify(submission, test, 'test')
    else:
        submission_robust = robustify(submission, stresstest, 'stresstest', 'stresstest_tracking')
    submission_robust.index.name = 'row_id'
    submission_robust.to_csv('submission.csv')
    df = pd.read_csv("submission.csv")
    print(df.head())
