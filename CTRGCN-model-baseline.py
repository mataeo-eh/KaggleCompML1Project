from __future__ import annotations
import os
import json
import time
import random
import itertools
import csv
import re
from typing import Any
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import median_filter


start_time = time.time()
# ======================
# CONFIGURATION
# ======================
RUN_MODE = "dev"  # one of: dev, validate, submit, tune, tune_grid
STREAM_MODE = "one"  # one of: "one", "two", "four"
traintest_directory_path = "./Data/train_tracking" # The path for dev testing the code with competition data

GLOBAL_CONFIG = {
    "verbose": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "stream_mode": STREAM_MODE,
    "max_tuning_trials": 60,
    "tuning_min_videos": 3,
    "param_distributions": {
        "window": [32, 64, 96],
        "stride": [16, 32],
        "base_channels": [64, 96, 128],
        "num_blocks": [8, 10, 12],
        "dropout": [0.05, 0.1, 0.2, 0.3],
        "learning_rate": [0.1, 0.05, 0.01],
        "momentum": [0.8, 0.9],
        "weight_decay": [1e-4, 5e-4],
        "batch_size": [32, 64],
        "epochs": [60, 90, 120],
        "alpha_class_balance": [0.5, 1.0, 1.5],
        "decision_threshold" : [0.1, 0.2, 0.25, 0.30, 0.35],
        "temporal_kernel_size": [5, 7, 9],
    },
}

cwd = Path.cwd()
verbose = True



# ======================
# CONFIG DATA CLASS
# ======================
@dataclass
class CTRGCNConfig:
    mode: str = RUN_MODE  # dev, validate, submit, tune, tune_grid
    stream_mode: str = STREAM_MODE  # one, two, four

    show_progress: bool = False # Use tqdm progress tracking


    max_videos: int | None = None
    max_batches: int | None = None
    max_windows: int | None = None

    use_delta: bool = True
    use_bone: bool = True
    use_bone_delta: bool = True

    in_channels_single_stream: int = 8
    in_channels_streamA: int = 4
    in_channels_streamB: int = 4
    in_channels_coords_only: int = 2
    in_channels_delta_only: int = 2
    in_channels_bone_only: int = 2
    in_channels_bone_delta_only: int = 2

    window: int = 64
    stride: int = 32
    base_channels: int = 64
    num_blocks: int = 12
    temporal_kernel_size: int = 9
    use_ctr: bool = True
    use_edge_importance: bool = True
    dropout: float = 0.1
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 120
    grad_clip: float = 1.0
    alpha_balance: float | None = None
    decision_threshold: float = 0.2
    use_random_crop: bool = True
    use_random_rotation: bool = True
    max_rotation_deg: float = 15.0
    crop_jitter_frames: int = 8


# ======================
# DIRECTORY / PARAM HELPERS
# ======================
def get_stream_mode_tag(cfg: CTRGCNConfig) -> str:
    mode = getattr(cfg, "stream_mode", "one")
    assert mode in {"one", "two", "four"}, f"Unsupported stream_mode: {mode}"
    return mode


def slugify_bodyparts(bp_str: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", bp_str).strip("_")


def get_stream_model_dir(cfg: CTRGCNConfig, bp_slug: str | None = None) -> str:
    root = "./CTR-GCN-Models2.0"
    tag = get_stream_mode_tag(cfg)
    sub = {"one": "one_stream", "two": "two_stream", "four": "four_stream"}[tag]
    if bp_slug is None:
        bp_slug = "all_parts"
    bp_slug_len = str(len(bp_slug.split("_")))
    path = os.path.join(root, sub, bp_slug_len)
    os.makedirs(path, exist_ok=True)
    return path


def get_best_params_path_for_stream(cfg: CTRGCNConfig, bp_slug: str | None = None) -> str:
    os.makedirs("tuning_results", exist_ok=True)
    tag = get_stream_mode_tag(cfg)
    if bp_slug is None:
        bp_slug = "all_parts"
    return os.path.join("tuning_results", f"best_params_{tag}_{bp_slug}.csv")


def load_best_params_csv_for_config(config: CTRGCNConfig, bp_slug: str | None = None) -> dict | None:
    path = get_best_params_path_for_stream(config, bp_slug=bp_slug)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df.to_dict(orient="records")[0]
    except Exception as e:
        print(f"Warning: Could not read {path}: {e}")
        return None


def extract_actions_from_behaviors_labeled(raw_str: str) -> list[str]:
    """
    Given behaviors_labeled string like '["1,2,attack", "2,1,evade"]',
    return unique sorted list of action names (last token).
    """
    entries = json.loads(raw_str)
    out = []
    for ent in entries:
        parts = ent.split(",")
        action = parts[-1]
        out.append(action)
    return sorted(list(set(out)))


# ======================
# JOINTS + ADJACENCY
# ======================
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


def get_ordered_joints_and_adjacency(body_parts_tracked):
    ordered_joints = [bp for bp in MASTER_MOUSE_JOINT_ORDER if bp in body_parts_tracked]
    V = len(ordered_joints)
    adjacency = np.zeros((V, V), dtype=np.float32)
    for i in range(V - 1):
        adjacency[i, i + 1] = 1.0
        adjacency[i + 1, i] = 1.0
    return ordered_joints, adjacency


def make_pair_adjacency(adjacency_single: np.ndarray) -> np.ndarray:
    """Block-diagonal adjacency for two mice using the same single-mouse adjacency."""
    zeros = np.zeros_like(adjacency_single)
    return np.block([[adjacency_single, zeros], [zeros, adjacency_single]])


def flatten_pair_dataframe(pair_df: pd.DataFrame, ordered_joints: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Flatten pair data (A/B levels) into single-mouse-style columns with A_/B_ prefixes."""
    part_order = ordered_joints
    sub_a = pair_df["A"]
    sub_b = pair_df["B"]
    sub_a.columns = pd.MultiIndex.from_tuples([(f"A_{bp}", coord) for bp, coord in sub_a.columns])
    sub_b.columns = pd.MultiIndex.from_tuples([(f"B_{bp}", coord) for bp, coord in sub_b.columns])
    flat_df = pd.concat([sub_a, sub_b], axis=1)
    flat_order = [f"A_{bp}" for bp in part_order] + [f"B_{bp}" for bp in part_order]
    return flat_df, flat_order


# ======================
# DATA LOADING (GENERATE MOUSE DATA)
# ======================
drop_body_parts = [
    "headpiece_bottombackleft",
    "headpiece_bottombackright",
    "headpiece_bottomfrontleft",
    "headpiece_bottomfrontright",
    "headpiece_topbackleft",
    "headpiece_topbackright",
    "headpiece_topfrontleft",
    "headpiece_topfrontright",
    "spine_1",
    "spine_2",
    "tail_middle_1",
    "tail_middle_2",
    "tail_midpoint",
    "neck",
]


def filter_tracked_body_parts(body_parts_tracked: list[str]) -> list[str]:
    """Remove known noisy/dropped body parts to align with available tracking columns."""
    return [bp for bp in body_parts_tracked if bp not in drop_body_parts]


def generate_mouse_data(dataset, traintest, traintest_directory=None, generate_single=True, generate_pair=True, config: CTRGCNConfig | None = None):
    assert traintest in ["train", "test"]
    if traintest_directory is None:
        #traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
        traintest_directory = f"./Data/{traintest}_tracking"
    video_count = 0
    batch_count = 0
    for _, row in dataset.iterrows():
        if config is not None and config.max_videos is not None and video_count >= config.max_videos:
            break

        lab_id = row.lab_id
        if lab_id.startswith("MABe22"):
            continue
        video_id = row.video_id

        if type(row.behaviors_labeled) != str:
            # We cannot use videos without labeled behaviors
            print("No labeled behaviors:", lab_id, video_id, row.behaviors_labeled)
            continue

        # Load annotations early for train mode and determine relevant mice
        annot = None
        used_mice: set[int] = set()
        if traintest == "train":
            try:
                annot = pd.read_parquet(f"{traintest_directory}/{lab_id}/{video_id}.parquet".replace("train_tracking", "train_annotation"))
                if "agent_id" in annot.columns:
                    used_mice.update(annot["agent_id"].dropna().astype(int).tolist())
                if "target_id" in annot.columns:
                    used_mice.update(annot["target_id"].dropna().astype(int).tolist())
            except FileNotFoundError:
                continue
            if not used_mice:
                continue
            tracked_parts = set(filter_tracked_body_parts(json.loads(row.body_parts_tracked)))
            path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
            vid = pd.read_parquet(path)
            vid = vid[vid.mouse_id.isin(used_mice)]
            if len(vid) == 0:
                continue
            available_parts = set(vid.bodypart.unique())
            missing_tracked = tracked_parts - available_parts
            if missing_tracked:
                print(f"[generate_mouse_data] Skipping video {video_id} due to missing tracked parts (annotated mice only): {sorted(missing_tracked)}")
                continue
        else:
            path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
            # Peek columns to avoid heavy processing if tracked parts are missing
            vid_peek = pd.read_parquet(path, columns=["bodypart"])
            tracked_parts = set(filter_tracked_body_parts(json.loads(row.body_parts_tracked)))
            available_parts = set(vid_peek.bodypart.unique())
            missing_tracked = tracked_parts - available_parts
            if missing_tracked:
                print(f"[generate_mouse_data] Skipping video {video_id} due to missing tracked parts: {sorted(missing_tracked)}")
                continue
            vid = pd.read_parquet(path)

        if len(np.unique(vid.bodypart)) > 5:
            vid = vid.query("~ bodypart.isin(@drop_body_parts)")
        #Commented out for testing to see if this is affecting GCN skeleton creation
        pvid = vid.pivot(columns=["mouse_id", "bodypart"], index="video_frame", values=["x", "y"])
        if pvid.isna().any().any():
            # Compute missing ratio per joint
            missing_info = []
            skip_video = False
            for col in pvid.columns:
                missing_ratio = pvid[col].isna().mean()
                if missing_ratio >= 0.5:
                    skip_video = True
                missing_info.append((col, missing_ratio))
            if skip_video:
                # discard this video entirely
                del vid
                continue
            # interpolate remaining NaNs
            pvid = pvid.interpolate(method="linear", axis=0)
            pvid = pvid.ffill().bfill()
        del vid

        # Restrict to only mice that actually appear in annotations for this video (train only)
        if traintest == "train" and annot is not None:
            # Keep only columns for relevant mouse_ids
            mask_cols = pvid.columns.get_level_values("mouse_id").isin(used_mice)
            pvid = pvid.loc[:, mask_cols]
            if pvid.shape[1] == 0:
                # No relevant mice present after filtering; skip this video
                continue

        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T # mouse_id, body_part, xy
        pvid /= row.pix_per_cm_approx
        # safety: no NaNs after interpolation
        assert not pvid.isna().any().any()

        behaviors_entries = json.loads(row.behaviors_labeled)
        behaviors_entries = sorted(list({b.replace("'", "") for b in behaviors_entries}))
        vid_behaviors = [b.split(",") for b in behaviors_entries]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=["agent", "target", "action"])
        vid_behavior_actions = extract_actions_from_behaviors_labeled(row.behaviors_labeled)

        video_count += 1

        if generate_single:
            vid_behaviors_subset = vid_behaviors.query("target == 'self'")
            for mouse_id_str in np.unique(vid_behaviors_subset.agent):
                if config is not None and config.max_batches is not None and batch_count >= config.max_batches:
                    return
                try:
                    mouse_id = int(mouse_id_str[-1])
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("agent == @mouse_id_str").action)
                    single_mouse = pvid.loc[:, mouse_id]
                    assert len(single_mouse) == len(pvid)
                    single_mouse_meta = pd.DataFrame(
                        {
                            "video_id": video_id,
                            "agent_id": mouse_id_str,
                            "target_id": "self",
                            "video_frame": single_mouse.index,
                        }
                    )
                    if traintest == "train":
                        single_mouse_label = pd.DataFrame(np.nan, columns=vid_agent_actions, index=single_mouse.index)
                        annot_subset = annot.query("(agent_id == @mouse_id) & (target_id == @mouse_id)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            single_mouse_label.loc[annot_row["start_frame"] : annot_row["stop_frame"], annot_row.action] = 1.0
                        yield "single", single_mouse, single_mouse_meta, single_mouse_label
                        batch_count += 1
                    else:
                        yield "single", single_mouse, single_mouse_meta, vid_agent_actions
                        batch_count += 1
                except KeyError:
                    pass

        if generate_pair:
            try:
                vid_behaviors_subset = vid_behaviors.query("target != 'self'")
                if len(vid_behaviors_subset) > 0:
                    for agent, target in itertools.permutations(np.unique(pvid.columns.get_level_values("mouse_id")), 2):
                        if config is not None and config.max_batches is not None and batch_count >= config.max_batches:
                            return
                        agent_str = f"mouse{agent}"
                        target_str = f"mouse{target}"
                        vid_agent_actions = np.unique(vid_behaviors_subset.query("(agent == @agent_str) & (target == @target_str)").action)
                        mouse_pair = pd.concat([pvid[agent], pvid[target]], axis=1, keys=["A", "B"])
                        assert len(mouse_pair) == len(pvid)
                        mouse_pair_meta = pd.DataFrame(
                            {
                                "video_id": video_id,
                                "agent_id": agent_str,
                                "target_id": target_str,
                                "video_frame": mouse_pair.index,
                            }
                        )
                        if traintest == "train":
                            mouse_pair_label = pd.DataFrame(np.nan, columns=vid_agent_actions, index=mouse_pair.index)
                            annot_subset = annot.query("(agent_id == @agent) & (target_id == @target)")
                            for i in range(len(annot_subset)):
                                annot_row = annot_subset.iloc[i]
                                mouse_pair_label.loc[annot_row["start_frame"] : annot_row["stop_frame"], annot_row.action] = 1.0
                            yield "pair", mouse_pair, mouse_pair_meta, mouse_pair_label
                            batch_count += 1
                        else:
                            yield "pair", mouse_pair, mouse_pair_meta, vid_agent_actions
                            batch_count += 1
            except KeyError:
                pass

# ======================
# SLIDING WINDOWS
# ======================
def create_sliding_windows(single_mouse_df, window: int, stride: int):
    n_frames = len(single_mouse_df)
    frames = single_mouse_df.index.to_numpy()
    for start in range(0, n_frames - window + 1, stride):
        end = start + window
        window_df = single_mouse_df.iloc[start:end]
        frame_indices = frames[start:end]
        yield window_df, frame_indices


# ======================
# CTR-GCN MODEL COMPONENTS
# ======================
def _normalize_adjacency_chain(adjacency: np.ndarray) -> np.ndarray:
    V = adjacency.shape[0]
    A = adjacency.astype(np.float32).copy()
    A += np.eye(V, dtype=np.float32)
    row_sum = A.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    return A / row_sum


import math
class CTRGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adjacency: np.ndarray, coff_embedding: int = 4, use_ctr: bool = True):
        super().__init__()
        A_norm = _normalize_adjacency_chain(adjacency)
        self.A_base = nn.Parameter(torch.from_numpy(A_norm), requires_grad=True)
        self.use_ctr = use_ctr
        inter_channels = max(1, out_channels // coff_embedding)
        self.theta = nn.Conv2d(out_channels, inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(out_channels, inter_channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, edge_importance: torch.Tensor | None = None) -> torch.Tensor:
        # x: (N, C_in, V, T)
        x = self.proj(x)  # (N, C_out, V, T)
        A_eff = self.A_base
        if edge_importance is not None:
            A_eff = A_eff * edge_importance

        if self.use_ctr:
            theta_x = self.theta(x)  # (N, C_mid, V, T)
            phi_x = self.phi(x)
            N, C_mid, V, T = theta_x.shape
            theta_f = theta_x.mean(-1)        # (N, C_mid, V)
            phi_f   = phi_x.mean(-1)          # (N, C_mid, V)

            refine = torch.einsum("ncv,ncw->nvw", theta_f, phi_f)
            refine = refine.mean(0) / math.sqrt(C_mid)
            refine = torch.tanh(refine)  # (V, V)

            A_eff = A_eff + self.alpha * refine

        x = torch.einsum("ncvt,vw->ncwt", x, A_eff)
        x = self.bn(x)
        return x


class CTRGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adjacency: np.ndarray, stride: int = 1, dropout: float = 0.1, temporal_kernel: int = 9, use_ctr: bool = True, edge_importance: bool = True):
        super().__init__()
        self.gcn = CTRGraphConv(in_channels, out_channels, adjacency, use_ctr=use_ctr)
        pad_t = temporal_kernel // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, temporal_kernel), padding=(0, pad_t), stride=(1, stride), bias=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        if (in_channels != out_channels) or (stride != 1):
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1, stride), bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.edge_importance = edge_importance

    def forward(self, x: torch.Tensor, edge_importance: torch.Tensor | None = None) -> torch.Tensor:
        res = self.residual(x)
        x = self.gcn(x, edge_importance if self.edge_importance else None)
        x = self.tcn(x)
        x = x + res
        x = self.relu(x)
        return x


class CTRGCNBackbone(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, adjacency: np.ndarray, channels: list[int] | None = None, dropout: float = 0.1, temporal_kernel: int = 9, use_ctr: bool = True, use_edge_importance: bool = True):
        super().__init__()
        if channels is None:
            channels = [64, 64, 64, 64, 96, 96, 96, 96, 128, 128, 128, 128]
        blocks = []
        last_c = in_channels
        self.edge_importance = nn.ParameterList(
            [nn.Parameter(torch.ones_like(torch.from_numpy(_normalize_adjacency_chain(adjacency))), requires_grad=True) if use_edge_importance else nn.Parameter(torch.ones_like(torch.from_numpy(_normalize_adjacency_chain(adjacency))), requires_grad=False) for _ in channels]
        )
        for idx, out_c in enumerate(channels):
            blocks.append(CTRGCNBlock(last_c, out_c, adjacency, stride=1, dropout=dropout, temporal_kernel=temporal_kernel, use_ctr=use_ctr, edge_importance=use_edge_importance))
            last_c = out_c
        self.st_blocks = nn.ModuleList(blocks)
        self.fc = nn.Linear(last_c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block, imp in zip(self.st_blocks, self.edge_importance):
            out = block(out, imp)
        out = out.mean(dim=-2)  # (N, C_out, T)
        out = out.permute(0, 2, 1)  # (N, T, C_out)
        logits = self.fc(out)
        return logits


class CTRGCNTwoStream(nn.Module):
    def __init__(self, adjacency: np.ndarray, in_channels_coords: int = 2, in_channels_delta: int = 2, base_channels: int = 64, num_blocks: int = 12, dropout: float = 0.1, temporal_kernel: int = 9, use_ctr: bool = True, use_edge_importance: bool = True):
        super().__init__()
        channels = [base_channels] * num_blocks
        self.stream_coords = CTRGCNBackbone(in_channels_coords, base_channels, adjacency, channels=channels, dropout=dropout, temporal_kernel=temporal_kernel, use_ctr=use_ctr, use_edge_importance=use_edge_importance)
        self.stream_delta = CTRGCNBackbone(in_channels_delta, base_channels, adjacency, channels=channels, dropout=dropout, temporal_kernel=temporal_kernel, use_ctr=use_ctr, use_edge_importance=use_edge_importance)
        self.fc = nn.Linear(base_channels, 1)

    def forward(self, coords_x: torch.Tensor, delta_x: torch.Tensor) -> torch.Tensor:
        feat_A = self.stream_coords(coords_x)
        feat_B = self.stream_delta(delta_x)
        fused = feat_A + feat_B
        logits = self.fc(fused)
        return logits


class CTRGCNFourStream(nn.Module):
    def __init__(self, adjacency, base_channels=64, dropout=0.1, num_blocks=12, temporal_kernel: int = 9, use_ctr: bool = True, use_edge_importance: bool = True):
        super().__init__()
        channels = [base_channels] * num_blocks
        self.stream_coords = CTRGCNBackbone(2, base_channels, adjacency, channels=channels, dropout=dropout, temporal_kernel=temporal_kernel, use_ctr=use_ctr, use_edge_importance=use_edge_importance)
        self.stream_delta = CTRGCNBackbone(2, base_channels, adjacency, channels=channels, dropout=dropout, temporal_kernel=temporal_kernel, use_ctr=use_ctr, use_edge_importance=use_edge_importance)
        self.stream_bone = CTRGCNBackbone(2, base_channels, adjacency, channels=channels, dropout=dropout, temporal_kernel=temporal_kernel, use_ctr=use_ctr, use_edge_importance=use_edge_importance)
        self.stream_bone_delta = CTRGCNBackbone(2, base_channels, adjacency, channels=channels, dropout=dropout, temporal_kernel=temporal_kernel, use_ctr=use_ctr, use_edge_importance=use_edge_importance)
        self.fc = nn.Linear(base_channels, 1)

    def forward(self, coords_x, delta_x, bone_x, bone_delta_x):
        f1 = self.stream_coords(coords_x)
        f2 = self.stream_delta(delta_x)
        f3 = self.stream_bone(bone_x)
        f4 = self.stream_bone_delta(bone_delta_x)
        fused = f1 + f2 + f3 + f4
        return self.fc(fused)


# ======================
# INPUT PREPARATION
# ======================
def prepare_ctr_gcn_input(single_mouse_df, ordered_joints, config: CTRGCNConfig | None = None):
    if config is None:
        config = CTRGCNConfig()
    mode = getattr(config, "stream_mode", "one")
    window_len = getattr(config, "window", 90)
    stride = getattr(config, "stride", 30)

    V = len(ordered_joints)
    frame_ranges = []
    window_count = 0
    missing_joint_counts: dict[tuple, int] = {}
    all_nan_windows = 0
    missing_bp_columns: dict[str, int] = {}

    streamA_tensors: list[torch.Tensor] = []
    streamB_tensors: list[torch.Tensor] = []
    coords_tensors: list[torch.Tensor] = []
    delta_tensors: list[torch.Tensor] = []
    bone_tensors: list[torch.Tensor] = []
    bone_delta_tensors: list[torch.Tensor] = []
    window_tensors: list[torch.Tensor] = []

    for window_df, frame_indices in create_sliding_windows(single_mouse_df, window_len, stride):
        if config.max_windows is not None and window_count >= config.max_windows:
            break
        if len(window_df) != window_len:
            continue

        window_np = np.full((2, V, window_len), np.nan, dtype=np.float32)
        for j, bp in enumerate(ordered_joints):
            col_x = (bp, "x")
            col_y = (bp, "y")
            if col_x not in window_df.columns or col_y not in window_df.columns:
                missing_bp_columns[bp] = missing_bp_columns.get(bp, 0) + 1
                continue
            x_vals = window_df[col_x].to_numpy(dtype=np.float32, copy=False)
            y_vals = window_df[col_y].to_numpy(dtype=np.float32, copy=False)
            window_np[0, j, :] = x_vals
            window_np[1, j, :] = y_vals

        # Debug: identify joints with no data in this window and skip the window if everything is missing
        missing_joint_mask = np.isnan(window_np).all(axis=(0, 2))
        if missing_joint_mask.any():
            missing_joints = [ordered_joints[idx] for idx in np.where(missing_joint_mask)[0]]
            missing_joint_counts.setdefault(tuple(missing_joints), 0)
            missing_joint_counts[tuple(missing_joints)] += 1
            # If all joints are missing, drop the window entirely
            if missing_joint_mask.sum() == V:
                all_missing_count = missing_joint_counts.get(("__all_missing__",), 0) + 1
                missing_joint_counts[("__all_missing__",)] = all_missing_count
                continue
        if np.isnan(window_np).all():
            all_nan_windows += 1
            continue

        bone = np.zeros_like(window_np)
        for j in range(V - 1):
            bone[:, j, :] = window_np[:, j + 1, :] - window_np[:, j, :]

        mean_val = np.nanmean(window_np, axis=2, keepdims=True)
        window_np = window_np - mean_val
        bone = bone - mean_val

        anchor_name = None
        if "body_center" in ordered_joints:
            anchor_name = "body_center"
        elif "neck" in ordered_joints:
            anchor_name = "neck"
        if anchor_name is not None:
            anchor_idx = ordered_joints.index(anchor_name)
            anchor = window_np[:, anchor_idx : anchor_idx + 1, :]
            window_np = window_np - anchor
            bone = bone - anchor

        scale = np.nanstd(window_np)
        if scale > 0:
            window_np = window_np / scale
            bone = bone / scale

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
        elif mode == "four":
            coords_tensors.append(torch.from_numpy(window_np.astype(np.float32)))
            delta_tensors.append(torch.from_numpy(delta.astype(np.float32)))
            bone_tensors.append(torch.from_numpy(bone_curr.astype(np.float32)))
            bone_delta_tensors.append(torch.from_numpy(bone_delta.astype(np.float32)))
        else:
            raise ValueError(f"Unsupported stream_mode: {mode}")

        frame_ranges.append(frame_indices)
        window_count += 1

    if verbose:
        pass

    if mode == "one":
        if len(window_tensors) == 0:
            return torch.empty((0, config.in_channels_single_stream, V, window_len)), frame_ranges
        return torch.stack(window_tensors, dim=0), frame_ranges
    if mode == "two":
        return streamA_tensors, streamB_tensors, frame_ranges
    return coords_tensors, delta_tensors, bone_tensors, bone_delta_tensors, frame_ranges


# ======================
# WINDOW COLLECTION
# ======================
def collect_ctr_gcn_windows(batches, ordered_joints, config: CTRGCNConfig):
    mode = getattr(config, "stream_mode", "one")
    window_len = getattr(config, "window", 90)
    if verbose:
        pass

    X_windows_single: list[torch.Tensor] = []
    streamA_windows_single: list[torch.Tensor] = []
    streamB_windows_single: list[torch.Tensor] = []
    coords_windows_single: list[torch.Tensor] = []
    delta_windows_single: list[torch.Tensor] = []
    bone_windows_single: list[torch.Tensor] = []
    bone_delta_windows_single: list[torch.Tensor] = []
    y_dict_single: dict[str, list] = {}

    X_windows_pair: list[torch.Tensor] = []
    streamA_windows_pair: list[torch.Tensor] = []
    streamB_windows_pair: list[torch.Tensor] = []
    coords_windows_pair: list[torch.Tensor] = []
    delta_windows_pair: list[torch.Tensor] = []
    bone_windows_pair: list[torch.Tensor] = []
    bone_delta_windows_pair: list[torch.Tensor] = []
    y_dict_pair: dict[str, list] = {}
    frame_ranges_single: list = []
    frame_ranges_pair: list = []

    batch_count = 0
    for switch, data_df, meta_df, label_df in batches:
        if config.max_batches is not None and batch_count >= config.max_batches:
            break

        current_ordered = ordered_joints
        data_df_prepared = data_df
        if switch == "pair":
            # Flatten pair data to single-mouse-like columns with prefixes A_/B_
            part_order = ordered_joints
            sub_a = data_df["A"]
            sub_b = data_df["B"]
            sub_a.columns = pd.MultiIndex.from_tuples([(f"A_{bp}", coord) for bp, coord in sub_a.columns])
            sub_b.columns = pd.MultiIndex.from_tuples([(f"B_{bp}", coord) for bp, coord in sub_b.columns])
            data_df_prepared = pd.concat([sub_a, sub_b], axis=1)
            current_ordered = [f"A_{bp}" for bp in part_order] + [f"B_{bp}" for bp in part_order]

        if mode == "one":
            window_tensor, frame_ranges = prepare_ctr_gcn_input(data_df_prepared, current_ordered, config)
            if window_tensor.shape[0] == 0:
                batch_count += 1
                continue
        elif mode == "two":
            streamA_list, streamB_list, frame_ranges = prepare_ctr_gcn_input(data_df_prepared, current_ordered, config)
            if len(streamA_list) == 0:
                batch_count += 1
                continue
        else:
            coords_list, delta_list, bone_list, bone_delta_list, frame_ranges = prepare_ctr_gcn_input(data_df_prepared, current_ordered, config)
            if len(coords_list) == 0:
                batch_count += 1
                continue

        # Ensure action lists exist only for actions present in this label_df
        for action in label_df.columns:
            if switch == "single":
                if action not in y_dict_single:
                    y_dict_single[action] = []
            else:
                if action not in y_dict_pair:
                    y_dict_pair[action] = []

        for i, frame_range in enumerate(frame_ranges):
            if mode == "one":
                if switch == "single":
                    X_windows_single.append(window_tensor[i])
                    frame_ranges_single.append(frame_range)
                else:
                    X_windows_pair.append(window_tensor[i])
                    frame_ranges_pair.append(frame_range)
            elif mode == "two":
                if switch == "single":
                    streamA_windows_single.append(streamA_list[i])
                    streamB_windows_single.append(streamB_list[i])
                    frame_ranges_single.append(frame_range)
                else:
                    streamA_windows_pair.append(streamA_list[i])
                    streamB_windows_pair.append(streamB_list[i])
                    frame_ranges_pair.append(frame_range)
            else:
                if switch == "single":
                    coords_windows_single.append(coords_list[i])
                    delta_windows_single.append(delta_list[i])
                    bone_windows_single.append(bone_list[i])
                    bone_delta_windows_single.append(bone_delta_list[i])
                    frame_ranges_single.append(frame_range)
                else:
                    coords_windows_pair.append(coords_list[i])
                    delta_windows_pair.append(delta_list[i])
                    bone_windows_pair.append(bone_list[i])
                    bone_delta_windows_pair.append(bone_delta_list[i])
                    frame_ranges_pair.append(frame_range)
            target_dict = y_dict_single if switch == "single" else y_dict_pair
            actions_present = [a for a in target_dict.keys() if a in label_df.columns]
            for action in actions_present:
                window_labels = label_df[action].reindex(frame_range).to_numpy(dtype=np.float32)
                target_dict[action].append(window_labels)

        batch_count += 1

    result: dict[str, Any] = {}
    if mode == "one":
        single_tuple = ("one", torch.stack(X_windows_single, dim=0)) if len(X_windows_single) > 0 else ("one", torch.empty((0, config.in_channels_single_stream, len(ordered_joints), window_len)))
        pair_tuple = ("one", torch.stack(X_windows_pair, dim=0)) if len(X_windows_pair) > 0 else ("one", torch.empty((0, config.in_channels_single_stream, len(ordered_joints), window_len)))
    elif mode == "two":
        single_tuple = ("two", (streamA_windows_single, streamB_windows_single))
        pair_tuple = ("two", (streamA_windows_pair, streamB_windows_pair))
    else:
        single_tuple = ("four", (coords_windows_single, delta_windows_single, bone_windows_single, bone_delta_windows_single))
        pair_tuple = ("four", (coords_windows_pair, delta_windows_pair, bone_windows_pair, bone_delta_windows_pair))

    result["windows_single"] = single_tuple
    result["windows_pair"] = pair_tuple
    result["labels"] = {"single": y_dict_single, "pair": y_dict_pair}
    result["frame_ranges"] = {"single": frame_ranges_single, "pair": frame_ranges_pair}
    return result


# ======================
# TRAINING / VALIDATION UTIL
# ======================
def compute_validation_f1_from_windows(windows_tuple, y_dict: dict[str, list], adjacency: np.ndarray, config: CTRGCNConfig, device: str, seed: int = 0) -> float:
    mode, data = windows_tuple
    if mode == "one" and data.shape[0] == 0:
        return 0.0
    if mode in ("two", "four"):
        if not data or (isinstance(data, tuple) and len(data[0]) == 0):
            return 0.0

    if mode == "one":
        X = data.to(device)
    elif mode == "two":
        X_streamA = torch.stack(data[0], dim=0).to(device)
        X_streamB = torch.stack(data[1], dim=0).to(device)
    else:
        X_coords = torch.stack(data[0], dim=0).to(device)
        X_delta = torch.stack(data[1], dim=0).to(device)
        X_bone = torch.stack(data[2], dim=0).to(device)
        X_bone_delta = torch.stack(data[3], dim=0).to(device)

    base_channels = getattr(config, "base_channels", 64)
    num_blocks = getattr(config, "num_blocks", 3)
    dropout = getattr(config, "dropout", 0.1)
    lr = getattr(config, "lr", 1e-3)
    alpha_balance = getattr(config, "alpha_balance", None)

    if config.mode == "dev":
        epochs = 1
    elif config.mode == "validate":
        epochs = 3
    else:
        epochs = 5

    f1_scores: list[float] = []
    rng = np.random.default_rng(seed)

    for action, labels in y_dict.items():
        y_tensor = torch.tensor(labels, dtype=torch.float32, device=device)  # (N, T)
        mask = ~torch.isnan(y_tensor)
        if mask.sum().item() < 2:
            continue
        y_action = y_tensor

        if mode == "one":
            X_action = X
        elif mode == "two":
            X_action_A = X_streamA
            X_action_B = X_streamB
        else:
            X_action_coords = X_coords
            X_action_delta = X_delta
            X_action_bone = X_bone
            X_action_bone_delta = X_bone_delta

        if mode == "one":
            x_len = X_action.shape[0]
        elif mode == "two":
            x_len = X_action_A.shape[0]
        else:
            x_len = X_action_coords.shape[0]
        n_samples = min(y_action.shape[0], x_len)
        if n_samples <= 1:
            continue
        if n_samples != y_action.shape[0] and verbose:
            print(f"[compute_validation_f1_from_windows] Mismatch between labels ({y_action.shape[0]}) and windows ({x_len}); using {n_samples}")
        y_action = y_action[:n_samples]
        mask = mask[:n_samples]
        perm = rng.permutation(n_samples)
        split_idx = max(1, int(0.8 * n_samples))
        if split_idx >= n_samples:
            split_idx = n_samples - 1
        train_idx = perm[:split_idx]
        val_idx = perm[split_idx:]

        temporal_kernel = getattr(config, "temporal_kernel_size", 9)
        use_ctr = getattr(config, "use_ctr", True)
        use_edge_importance = getattr(config, "use_edge_importance", True)
        if mode == "one":
            model = CTRGCNBackbone(
                in_channels=config.in_channels_single_stream,
                num_classes=1,
                adjacency=adjacency,
                channels=[base_channels] * num_blocks,
                dropout=dropout,
                temporal_kernel=temporal_kernel,
                use_ctr=use_ctr,
                use_edge_importance=use_edge_importance,
            ).to(device)
        elif mode == "two":
            model = CTRGCNTwoStream(
                adjacency=adjacency,
                in_channels_coords=config.in_channels_streamA,
                in_channels_delta=config.in_channels_streamB,
                base_channels=base_channels,
                num_blocks=num_blocks,
                dropout=dropout,
                temporal_kernel=temporal_kernel,
                use_ctr=use_ctr,
                use_edge_importance=use_edge_importance,
            ).to(device)
        else:
            model = CTRGCNFourStream(
                adjacency=adjacency,
                base_channels=base_channels,
                dropout=dropout,
                num_blocks=num_blocks,
                temporal_kernel=temporal_kernel,
                use_ctr=use_ctr,
                use_edge_importance=use_edge_importance,
            ).to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=getattr(config, "momentum", 0.9), weight_decay=getattr(config, "weight_decay", 1e-4))
        if alpha_balance is not None:
            pos_weight = torch.tensor([alpha_balance], device=device)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = torch.nn.BCEWithLogitsLoss()

        for _ in range(epochs):
            model.train()
            for start in range(0, len(train_idx), 16):
                end = start + 16
                batch_ids = train_idx[start:end]
                batch_y = y_action[batch_ids]
                batch_mask = mask[batch_ids]
                if mode == "one":
                    logits = model(X_action[batch_ids]).squeeze(-1)
                elif mode == "two":
                    logits = model(X_action_A[batch_ids], X_action_B[batch_ids]).squeeze(-1)
                else:
                    logits = model(
                        X_action_coords[batch_ids],
                        X_action_delta[batch_ids],
                        X_action_bone[batch_ids],
                        X_action_bone_delta[batch_ids],
                    ).squeeze(-1)
                valid = batch_mask
                if valid.sum().item() == 0:
                    continue
                logits = logits[valid]
                batch_y = batch_y[valid]
                optimizer.zero_grad()
                loss = criterion(logits, batch_y)
                loss.backward()
                clip_val = getattr(config, "grad_clip", None)
                if clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                optimizer.step()

        model.eval()
        with torch.no_grad():
            if mode == "one":
                logits_val = model(X_action[val_idx]).squeeze(-1)
            elif mode == "two":
                logits_val = model(X_action_A[val_idx], X_action_B[val_idx]).squeeze(-1)
            else:
                logits_val = model(
                    X_action_coords[val_idx],
                    X_action_delta[val_idx],
                    X_action_bone[val_idx],
                    X_action_bone_delta[val_idx],
                ).squeeze(-1)
        mask_val = mask[val_idx]
        if mask_val.sum().item() == 0:
            continue
        probs = torch.sigmoid(logits_val).cpu().numpy()
        y_val = y_action[val_idx].cpu().numpy()
        valid = mask_val.cpu().numpy()
        probs = probs[valid]
        y_val = y_val[valid]
        thresh = getattr(config, "decision_threshold", 0.5)
        preds = (probs > thresh).astype(int)
        if (preds.sum() + y_val.sum()) == 0:
            f1 = 0.0
        else:
            tp = ((preds == 1) & (y_val == 1)).sum()
            fp = ((preds == 1) & (y_val == 0)).sum()
            fn = ((preds == 0) & (y_val == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)

    if len(f1_scores) == 0:
        return 0.0
    return float(sum(f1_scores) / len(f1_scores))


def train_ctr_gcn_models(batches, ordered_joints, adjacency, config: CTRGCNConfig, device: str | None = None):
    # Device resolution
    if device is None:
        device = config.device

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"Training CTR-GCN using device: {device}")

    model_dict_single: dict[str, nn.Module] = {}
    model_dict_pair: dict[str, nn.Module] = {}

    data_dict = collect_ctr_gcn_windows(batches, ordered_joints, config)
    windows_single = data_dict["windows_single"]
    windows_pair = data_dict["windows_pair"]
    labels_single = data_dict["labels"]["single"]
    labels_pair = data_dict["labels"]["pair"]
    base_channels = getattr(config, "base_channels", 64)
    num_blocks = getattr(config, "num_blocks", 3)
    dropout = getattr(config, "dropout", 0.1)
    lr = getattr(config, "lr", 1e-3)
    alpha_balance = getattr(config, "alpha_balance", None)

    for switch_name, streams_tuple, y_dict in [
        ("single", windows_single, labels_single),
        ("pair", windows_pair, labels_pair),
    ]:
        mode, data = streams_tuple
        if mode == "one" and (data is None or data.shape[0] == 0):
            continue
        if mode in ("two", "four") and (not data or (isinstance(data, tuple) and len(data[0]) == 0)):
            continue

        if mode == "one":
            X = data.to(device)
        elif mode == "two":
            X_streamA = torch.stack(data[0], dim=0).to(device)
            X_streamB = torch.stack(data[1], dim=0).to(device)
        else:
            X_coords = torch.stack(data[0], dim=0).to(device)
            X_delta = torch.stack(data[1], dim=0).to(device)
            X_bone = torch.stack(data[2], dim=0).to(device)
            X_bone_delta = torch.stack(data[3], dim=0).to(device)
        if verbose:
            if mode == "one":
                print(f"[train_ctr_gcn_models] switch={switch_name} X shape {X.shape}")
            elif mode == "two":
                print(f"[train_ctr_gcn_models] switch={switch_name} streamA {X_streamA.shape}, streamB {X_streamB.shape}")
            else:
                print(f"[train_ctr_gcn_models] switch={switch_name} coords {X_coords.shape}, delta {X_delta.shape}, bone {X_bone.shape}, bone_delta {X_bone_delta.shape}")

        batch_size = getattr(config, "batch_size", 16)
        if config.mode == "dev":
            epochs = 1
        elif config.mode == "validate":
            epochs = 3
        else:
            epochs = getattr(config, "epochs", 5)

        # Choose adjacency based on switch (single vs pair)
        adjacency_local = adjacency if switch_name == "single" else make_pair_adjacency(adjacency)

        for action, labels in y_dict.items():
            labels_np = np.asarray(labels, dtype=np.float32)   # (N, T)
            y_tensor = torch.from_numpy(labels_np).to(device)  # (N, T)

            mask = ~torch.isnan(y_tensor)
            if mask.sum().item() == 0:
                continue
            y_action = y_tensor

            total_pos = torch.nan_to_num(y_action, nan=0.0).sum().item()
            if verbose:
                print(f"[train] action={action}, windows={y_action.shape[0]}, frames={y_action.shape[1]}, positives={total_pos}")



            temporal_kernel = getattr(config, "temporal_kernel_size", 9)
            use_ctr = getattr(config, "use_ctr", True)
            use_edge_importance = getattr(config, "use_edge_importance", True)

            if mode == "one":
                X_action = X
                x_len = X_action.shape[0]
                model = CTRGCNBackbone(
                    in_channels=config.in_channels_single_stream,
                    num_classes=1,
                    adjacency=adjacency_local,
                    channels=[base_channels] * num_blocks,
                    dropout=dropout,
                    temporal_kernel=temporal_kernel,
                    use_ctr=use_ctr,
                    use_edge_importance=use_edge_importance,
                ).to(device)
            elif mode == "two":
                X_action_A = X_streamA
                X_action_B = X_streamB
                x_len = min(X_action_A.shape[0], X_action_B.shape[0])
                model = CTRGCNTwoStream(
                    adjacency=adjacency_local,
                    in_channels_coords=config.in_channels_streamA,
                    in_channels_delta=config.in_channels_streamB,
                    base_channels=base_channels,
                    num_blocks=num_blocks,
                    dropout=dropout,
                    temporal_kernel=temporal_kernel,
                    use_ctr=use_ctr,
                    use_edge_importance=use_edge_importance,
                ).to(device)
            else:
                X_action_coords = X_coords
                X_action_delta = X_delta
                X_action_bone = X_bone
                X_action_bone_delta = X_bone_delta
                x_len = min(
                    X_action_coords.shape[0],
                    X_action_delta.shape[0],
                    X_action_bone.shape[0],
                    X_action_bone_delta.shape[0],
                )
                model = CTRGCNFourStream(
                    adjacency=adjacency_local,
                    base_channels=base_channels,
                    dropout=dropout,
                    num_blocks=num_blocks,
                    temporal_kernel=temporal_kernel,
                    use_ctr=use_ctr,
                    use_edge_importance=use_edge_importance,
                ).to(device)

            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=getattr(config, "momentum", 0.9), weight_decay=getattr(config, "weight_decay", 1e-4))
            if alpha_balance is not None:
                pos_weight = torch.tensor([alpha_balance], device=device)
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                criterion = torch.nn.BCEWithLogitsLoss()

            # Align sample counts between labels and windows
            n_samples = min(y_action.shape[0], x_len)
            if n_samples == 0:
                continue
            if n_samples != y_action.shape[0] and verbose:
                print(f"[train_ctr_gcn_models] Mismatch between labels ({y_action.shape[0]}) and windows ({x_len}) for action {action}; using {n_samples}")
            y_action = y_action[:n_samples]
            mask = mask[:n_samples]

            for _ in range(epochs):
                model.train()
                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    batch_y = y_action[start:end]
                    batch_mask = mask[start:end]
                    if mode == "one":
                        logits = model(X_action[start:end]).squeeze(-1)
                    elif mode == "two":
                        logits = model(X_action_A[start:end], X_action_B[start:end]).squeeze(-1)
                    else:
                        logits = model(
                            X_action_coords[start:end],
                            X_action_delta[start:end],
                            X_action_bone[start:end],
                            X_action_bone_delta[start:end],
                        ).squeeze(-1)
                    if batch_mask.sum().item() == 0:
                        continue
                    logits = logits[batch_mask]
                    batch_y_masked = batch_y[batch_mask]
                    optimizer.zero_grad()
                    loss = criterion(logits, batch_y_masked)
                    loss.backward()
                    clip_val = getattr(config, "grad_clip", None)
                    if clip_val is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                    optimizer.step()

            if switch_name == "single":
                model_dict_single[action] = model
            else:
                model_dict_pair[action] = model

    return {"single": model_dict_single, "pair": model_dict_pair}


# ======================
# MULTICLASS PREDICTION (FRAME->EVENT)
# ======================
def predict_multiclass(pred, meta, threshold: float = 0.27):
    ama = np.argmax(pred, axis=1)
    ama = np.where(pred.max(axis=1) >= threshold, ama, -1)
    ama = pd.Series(ama, index=meta.video_frame)
    changes_mask = (ama != ama.shift(1)).values
    ama_changes = ama[changes_mask]
    meta_changes = meta[changes_mask]
    mask = ama_changes.values >= 0
    mask[-1] = False
    submission_part = pd.DataFrame(
        {
            "video_id": meta_changes["video_id"][mask].values,
            "agent_id": meta_changes["agent_id"][mask].values,
            "target_id": meta_changes["target_id"][mask].values,
            "action": pred.columns[ama_changes[mask].values],
            "start_frame": ama_changes.index[mask],
            "stop_frame": ama_changes.index[1:][mask[:-1]],
        }
    )
    stop_video_id = meta_changes["video_id"][1:][mask[:-1]].values
    stop_agent_id = meta_changes["agent_id"][1:][mask[:-1]].values
    stop_target_id = meta_changes["target_id"][1:][mask[:-1]].values
    for i in range(len(submission_part)):
        video_id = submission_part.video_id.iloc[i]
        agent_id = submission_part.agent_id.iloc[i]
        target_id = submission_part.target_id.iloc[i]
        if stop_video_id[i] != video_id or stop_agent_id[i] != agent_id or stop_target_id[i] != target_id:
            new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
            submission_part.iat[i, submission_part.columns.get_loc("stop_frame")] = new_stop_frame
    assert (submission_part.stop_frame > submission_part.start_frame).all(), "stop <= start"
    if verbose:
        print("  actions found:", len(submission_part))
    return submission_part


# ======================
# MODEL LOADING / INFERENCE
# ======================
def load_ctr_gcn_models(model_dir: str | None, actions: list[str], adjacency: np.ndarray, config: CTRGCNConfig, device: str | None = None, bp_slug: str | None = None, switch_tr: str = "single") -> dict[str, nn.Module]:
    # Device resolution
    if device is None:
        device = config.device

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"Loading CTR-GCN using device: {device}")

    model_dict: dict[str, nn.Module] = {}

    best_params = load_best_params_csv_for_config(config, bp_slug=bp_slug)
    if best_params is not None:
        for key, value in best_params.items():
            if hasattr(config, key):
                setattr(config, key, value)

    mode = getattr(config, "stream_mode", "one")
    base_channels = getattr(config, "base_channels", 64)
    num_blocks = getattr(config, "num_blocks", 3)
    dropout = getattr(config, "dropout", 0.1)
    adjacency_local = adjacency if switch_tr == "single" else make_pair_adjacency(adjacency)
    base_dir = get_stream_model_dir(config, bp_slug=bp_slug)
    model_dir = os.path.join(base_dir, switch_tr)
    os.makedirs(model_dir, exist_ok=True)

    for action in actions:
        temporal_kernel = getattr(config, "temporal_kernel_size", 9)
        use_ctr = getattr(config, "use_ctr", True)
        use_edge_importance = getattr(config, "use_edge_importance", True)
        if mode == "one":
            model = CTRGCNBackbone(
                in_channels=config.in_channels_single_stream,
                num_classes=1,
                adjacency=adjacency_local,
                channels=[base_channels] * num_blocks,
                dropout=dropout,
                temporal_kernel=temporal_kernel,
                use_ctr=use_ctr,
                use_edge_importance=use_edge_importance,
            )
        elif mode == "two":
            model = CTRGCNTwoStream(
                adjacency=adjacency_local,
                in_channels_coords=config.in_channels_streamA,
                in_channels_delta=config.in_channels_streamB,
                base_channels=base_channels,
                num_blocks=num_blocks,
                dropout=dropout,
                temporal_kernel=temporal_kernel,
                use_ctr=use_ctr,
                use_edge_importance=use_edge_importance,
            )
        elif mode == "four":
            model = CTRGCNFourStream(
                adjacency=adjacency_local,
                base_channels=base_channels,
                dropout=dropout,
                num_blocks=num_blocks,
                temporal_kernel=temporal_kernel,
                use_ctr=use_ctr,
                use_edge_importance=use_edge_importance,
            )
        else:
            raise ValueError(f"Unsupported stream_mode: {mode}")

        weight_path = os.path.join(model_dir, f"{action}.pt")
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        model_dict[action] = model

    return model_dict


def submit_ctr_gcn(body_parts_tracked_str: str, switch_tr: str, model_dict: dict[str, nn.Module], config: CTRGCNConfig, device: str | None = None, bp_slug: str | None = None) -> pd.DataFrame:
    # Device resolution
    if device is None:
        device = config.device

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"Training CTR-GCN using device: {device}")

    best_params = load_best_params_csv_for_config(config, bp_slug=bp_slug)
    if best_params is not None:
        for key, value in best_params.items():
            if hasattr(config, key):
                setattr(config, key, value)

    body_parts_tracked = filter_tracked_body_parts(json.loads(body_parts_tracked_str))
    if RUN_MODE == "submit":
        test_subset = test[test.body_parts_tracked == body_parts_tracked_str]
        generator = generate_mouse_data(test_subset, "test", generate_single=True, generate_pair=True, config=config)
    else:
        test_subset = train.query("body_parts_tracked == @body_parts_tracked_str")
        generator = generate_mouse_data(test_subset, "train", generate_single=True, generate_pair=True, config=config)

    ordered_joints, adjacency = get_ordered_joints_and_adjacency(body_parts_tracked)
    mode = getattr(config, "stream_mode", "one")

    submissions = []
    for switch_te, data_te, meta_te, actions_te in generator:
        if switch_te != switch_tr:
            continue
        actions_available = [a for a in actions_te if a in model_dict]
        if not actions_available:
            continue

        ordered_for_input = ordered_joints
        data_prepared = data_te
        adjacency_use = adjacency if switch_tr == "single" else make_pair_adjacency(adjacency)
        if switch_te == "pair":
            data_prepared, ordered_for_input = flatten_pair_dataframe(data_te, ordered_joints)

        if mode == "one":
            window_tensor, frame_ranges = prepare_ctr_gcn_input(data_prepared, ordered_for_input, config)
            if window_tensor.shape[0] == 0:
                continue
            X = window_tensor.to(device)
        elif mode == "two":
            streamA_list, streamB_list, frame_ranges = prepare_ctr_gcn_input(data_prepared, ordered_for_input, config)
            if len(streamA_list) == 0:
                continue
            X_streamA = torch.stack(streamA_list, dim=0).to(device)
            X_streamB = torch.stack(streamB_list, dim=0).to(device)
        else:
            coords_list, delta_list, bone_list, bone_delta_list, frame_ranges = prepare_ctr_gcn_input(data_prepared, ordered_for_input, config)
            if len(coords_list) == 0:
                continue
            X_coords = torch.stack(coords_list, dim=0).to(device)
            X_delta = torch.stack(delta_list, dim=0).to(device)
            X_bone = torch.stack(bone_list, dim=0).to(device)
            X_bone_delta = torch.stack(bone_delta_list, dim=0).to(device)

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
                    logits = model(X).squeeze(-1)
                elif mode == "two":
                    logits = model(X_streamA, X_streamB).squeeze(-1)
                else:
                    logits = model(X_coords, X_delta, X_bone, X_bone_delta).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()  # (N, T)
            for w_idx, frames in enumerate(frame_ranges):
                for t, f in enumerate(frames):
                    fi = frame_to_idx.get(f)
                    if fi is None or t >= probs.shape[1]:
                        continue
                    p = float(probs[w_idx, t])
                    sum_probs[fi, action_idx] += p
                    counts[fi, action_idx] += 1.0

        counts[counts == 0] = 1.0
        pred_array = sum_probs / counts
        pred_array = median_filter(pred_array, size=(5, 1))
        pred_df = pd.DataFrame(pred_array, index=meta_te.video_frame, columns=actions_available)
        thresh = getattr(config, "decision_threshold", 0.27)
        submission_part = predict_multiclass(pred_df, meta_te, threshold=thresh)
        submissions.append(submission_part)

    if len(submissions) == 0:
        return pd.DataFrame(columns=["video_id", "agent_id", "target_id", "action", "start_frame", "stop_frame"])
    return pd.concat(submissions, ignore_index=True)


# ======================
# MAIN EXECUTION LOGIC (TUNE / TUNE_GRID / VALIDATE / SUBMIT / DEV)
# ======================
train = pd.read_csv(cwd / "Data" / "train.csv")
train["n_mice"] = 4 - train[["mouse1_strain", "mouse2_strain", "mouse3_strain", "mouse4_strain"]].isna().sum(axis=1)
train_without_mabe22 = train.query("~ lab_id.str.startswith('MABe22_')")
test = pd.read_csv(cwd / "Data" / "test.csv")
body_parts_tracked_list = sorted(train.body_parts_tracked.unique())

if RUN_MODE == "tune":
    print("Running CTR-GCN random-search hyperparameter tuning...")
    device = GLOBAL_CONFIG["device"]
    stream_mode = GLOBAL_CONFIG.get("stream_mode", "one")
    param_space = GLOBAL_CONFIG["param_distributions"]
    max_trials = GLOBAL_CONFIG["max_tuning_trials"]
    num_videos = GLOBAL_CONFIG["tuning_min_videos"]
    os.makedirs("tuning_results", exist_ok=True)

    for bp_str in body_parts_tracked_list:
        train_subset_full = train[train.body_parts_tracked == bp_str]
        if len(train_subset_full) == 0:
            continue
        bp_slug = slugify_bodyparts(bp_str)
        train_subset = train_subset_full.sample(n=min(num_videos, len(train_subset_full)), random_state=42)
        ordered_joints, adjacency = get_ordered_joints_and_adjacency(filter_tracked_body_parts(json.loads(bp_str)))
        results_path = os.path.join("tuning_results", f"tuning_results_{stream_mode}_{bp_slug}.csv")
        with open(results_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trial", "window", "stride", "base_channels", "num_blocks", "dropout", "lr", "momentum", "weight_decay", "batch_size", "epochs", "temporal_kernel_size", "alpha_balance", "decision_threshold", "mean_f1", "timestamp"])

        best_f1 = -1
        best_params = None
        for trial in range(1, max_trials + 1):
            print(f"\n=== Trial {trial}/{max_trials} for body_parts_tracked {bp_slug} ===")
            window = random.choice(param_space["window"])
            stride = random.choice(param_space["stride"])
            base_channels = random.choice(param_space["base_channels"])
            num_blocks = random.choice(param_space["num_blocks"])
            dropout = random.choice(param_space["dropout"])
            lr = random.choice(param_space["learning_rate"])
            momentum = random.choice(param_space.get("momentum", [0.9]))
            weight_decay = random.choice(param_space.get("weight_decay", [1e-4]))
            batch_size = random.choice(param_space.get("batch_size", [64]))
            epochs = random.choice(param_space.get("epochs", [120]))
            temporal_kernel = random.choice(param_space.get("temporal_kernel_size", [9]))
            alpha_balance = random.choice(param_space["alpha_class_balance"])
            decision_threshold = random.choice(param_space["decision_threshold"])

            cfg = CTRGCNConfig(mode="validate", max_videos=num_videos, use_delta=True, use_bone=True, use_bone_delta=True, stream_mode=stream_mode)
            cfg.window = window
            cfg.stride = stride
            cfg.base_channels = base_channels
            cfg.num_blocks = num_blocks
            cfg.dropout = dropout
            cfg.lr = lr
            cfg.momentum = momentum
            cfg.weight_decay = weight_decay
            cfg.batch_size = batch_size
            cfg.epochs = epochs
            cfg.temporal_kernel_size = temporal_kernel
            cfg.alpha_balance = alpha_balance
            cfg.decision_threshold = decision_threshold
            cfg.max_batches = None
            cfg.max_windows = None

            batches = []
            for switch, data_df, meta_df, label_df in generate_mouse_data(train_subset, "train", generate_single=True, generate_pair=True, config=cfg):
                batches.append((switch, data_df, meta_df, label_df))

            data_dict = collect_ctr_gcn_windows(batches, ordered_joints, cfg)
            mean_f1_single = compute_validation_f1_from_windows(data_dict["windows_single"], data_dict["labels"]["single"], adjacency, cfg, device=device, seed=trial)
            mean_f1_pair = compute_validation_f1_from_windows(data_dict["windows_pair"], data_dict["labels"]["pair"], make_pair_adjacency(adjacency), cfg, device=device, seed=trial)
            mean_f1 = np.mean([v for v in [mean_f1_single, mean_f1_pair] if v is not None])

            with open(results_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([trial, window, stride, base_channels, num_blocks, dropout, lr, momentum, weight_decay, batch_size, epochs, temporal_kernel, alpha_balance, decision_threshold, mean_f1, time.time()])

            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_params = {
                    "window": window,
                    "stride": stride,
                    "base_channels": base_channels,
                    "num_blocks": num_blocks,
                    "dropout": dropout,
                    "lr": lr,
                    "alpha_balance": alpha_balance,
                    "decision_threshold": decision_threshold,
                    "stream_mode": cfg.stream_mode,
                }
                model_dir = get_stream_model_dir(cfg, bp_slug=bp_slug)
                best_model_dict_all = train_ctr_gcn_models(batches, ordered_joints, adjacency, cfg, device=device)
                for action, model in best_model_dict_all["single"].items():
                    path_single = os.path.join(model_dir, "single")
                    os.makedirs(path_single, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(path_single, f"{action}.pt"))
                for action, model in best_model_dict_all["pair"].items():
                    path_pair = os.path.join(model_dir, "pair")
                    os.makedirs(path_pair, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(path_pair, f"{action}.pt"))

        print("\n==== Hyperparameter Tuning Complete for", bp_slug, "====")
        if best_params is not None:
            pd.DataFrame([best_params]).to_csv(get_best_params_path_for_stream(cfg, bp_slug=bp_slug), index=False)
        print("Best F1:", best_f1)

if RUN_MODE == "tune_grid":
    print("Running CTR-GCN full grid search (tune_grid)...")
    device = GLOBAL_CONFIG["device"]
    stream_mode = GLOBAL_CONFIG.get("stream_mode", "one")
    param_space = GLOBAL_CONFIG["param_distributions"]
    os.makedirs("grid_results", exist_ok=True)

    for bp_str in body_parts_tracked_list:
        train_subset = train[train.body_parts_tracked == bp_str].copy()
        if len(train_subset) == 0:
            continue
        bp_slug = slugify_bodyparts(bp_str)
        ordered_joints, adjacency = get_ordered_joints_and_adjacency(filter_tracked_body_parts(json.loads(bp_str)))
        grid = list(
            itertools.product(
                param_space["window"],
                param_space["stride"],
                param_space["base_channels"],
                param_space["num_blocks"],
                param_space["dropout"],
                param_space["learning_rate"],
                param_space.get("momentum", [0.9]),
                param_space.get("weight_decay", [1e-4]),
                param_space.get("batch_size", [64]),
                param_space.get("epochs", [120]),
                param_space["alpha_class_balance"],
                param_space["decision_threshold"],
                param_space.get("temporal_kernel_size", [9]),
            )
        )
        tag = stream_mode
        grid_results_path = os.path.join("grid_results", f"grid_results_{tag}_{bp_slug}.csv")
        with open(grid_results_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["idx", "window", "stride", "base_channels", "num_blocks", "dropout", "lr", "momentum", "weight_decay", "batch_size", "epochs", "alpha_balance", "decision_threshold", "temporal_kernel_size", "mean_f1"])

        best_f1 = -1.0
        best_params = None
        for idx, combo in enumerate(grid, start=1):
            window, stride, base_channels, num_blocks, dropout, lr, momentum, weight_decay, batch_size, epochs, alpha_balance, decision_threshold, temporal_kernel = combo
            cfg = CTRGCNConfig(mode="validate", max_videos=None, use_delta=True, use_bone=True, use_bone_delta=True, stream_mode=stream_mode)
            cfg.window = window
            cfg.stride = stride
            cfg.base_channels = base_channels
            cfg.num_blocks = num_blocks
            cfg.dropout = dropout
            cfg.lr = lr
            cfg.momentum = momentum
            cfg.weight_decay = weight_decay
            cfg.batch_size = batch_size
            cfg.epochs = epochs
            cfg.temporal_kernel_size = temporal_kernel
            cfg.alpha_balance = alpha_balance
            cfg.decision_threshold = decision_threshold
            cfg.max_batches = None
            cfg.max_windows = None

            batches = []
            for switch, data_df, meta_df, label_df in generate_mouse_data(train_subset, "train", generate_single=True, generate_pair=True, config=cfg):
                batches.append((switch, data_df, meta_df, label_df))

            data_dict = collect_ctr_gcn_windows(batches, ordered_joints, cfg)
            mean_f1_single = compute_validation_f1_from_windows(data_dict["windows_single"], data_dict["labels"]["single"], adjacency, cfg, device=device, seed=idx)
            mean_f1_pair = compute_validation_f1_from_windows(data_dict["windows_pair"], data_dict["labels"]["pair"], make_pair_adjacency(adjacency), cfg, device=device, seed=idx)
            mean_f1 = np.mean([v for v in [mean_f1_single, mean_f1_pair] if v is not None])

            with open(grid_results_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([idx, window, stride, base_channels, num_blocks, dropout, lr, momentum, weight_decay, batch_size, epochs, alpha_balance, decision_threshold, temporal_kernel, mean_f1])

            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_params = {
                    "window": window,
                    "stride": stride,
                    "base_channels": base_channels,
                    "num_blocks": num_blocks,
                    "dropout": dropout,
                    "lr": lr,
                    "alpha_balance": alpha_balance,
                    "decision_threshold": decision_threshold,
                    "stream_mode": cfg.stream_mode,
                }

            model_dict_all = train_ctr_gcn_models(batches, ordered_joints, adjacency, cfg, device=device)
            model_dir = get_stream_model_dir(cfg, bp_slug=bp_slug)
            for action, model in model_dict_all["single"].items():
                path_single = os.path.join(model_dir, "single")
                os.makedirs(path_single, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(path_single, f"{action}_{idx}.pt"))
            for action, model in model_dict_all["pair"].items():
                path_pair = os.path.join(model_dir, "pair")
                os.makedirs(path_pair, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(path_pair, f"{action}_{idx}.pt"))

        print("\nGrid search completed for", bp_slug)
        if best_params is not None:
            pd.DataFrame([best_params]).to_csv(os.path.join("grid_results", f"best_params_{tag}_{bp_slug}.csv"), index=False)
            os.makedirs("tuning_results", exist_ok=True)
            pd.DataFrame([best_params]).to_csv(get_best_params_path_for_stream(cfg, bp_slug=bp_slug), index=False)

if RUN_MODE == "submit":
    submission_list: list[pd.DataFrame] = []
    submit_cfg = CTRGCNConfig(mode="submit", stream_mode=GLOBAL_CONFIG.get("stream_mode", "one"))

    for bp_str in body_parts_tracked_list:
        test_subset = test[test.body_parts_tracked == bp_str]
        if len(test_subset) == 0:
            continue
        bp_slug = slugify_bodyparts(bp_str)
        best_params = load_best_params_csv_for_config(submit_cfg, bp_slug=bp_slug)
        cfg_local = CTRGCNConfig(mode="submit", stream_mode=submit_cfg.stream_mode)
        if best_params is not None:
            for key, value in best_params.items():
                if hasattr(cfg_local, key):
                    setattr(cfg_local, key, value)

        train_rows = train[train.body_parts_tracked == bp_str]
        if len(train_rows) == 0:
            continue
        sample_row = train_rows.iloc[0]
        actions = extract_actions_from_behaviors_labeled(sample_row.behaviors_labeled)
        body_parts_tracked_str = bp_str
        ordered_joints, adjacency = get_ordered_joints_and_adjacency(json.loads(body_parts_tracked_str))
        model_dir = get_stream_model_dir(cfg_local, bp_slug=bp_slug)
        model_dict = load_ctr_gcn_models(model_dir=model_dir, actions=actions, adjacency=adjacency, config=cfg_local, device=GLOBAL_CONFIG["device"], bp_slug=bp_slug, switch_tr="single")

        submission_df_single = submit_ctr_gcn(body_parts_tracked_str, "single", model_dict, cfg_local, device=GLOBAL_CONFIG["device"], bp_slug=bp_slug)
        submission_list.append(submission_df_single)

        model_dict_pair = load_ctr_gcn_models(model_dir=model_dir, actions=actions, adjacency=adjacency, config=cfg_local, device=GLOBAL_CONFIG["device"], bp_slug=bp_slug, switch_tr="pair")
        submission_df_pair = submit_ctr_gcn(body_parts_tracked_str, "pair", model_dict_pair, cfg_local, device=GLOBAL_CONFIG["device"], bp_slug=bp_slug)
        submission_list.append(submission_df_pair)

    if submission_list:
        submission_df = pd.concat(submission_list, ignore_index=True)
        submission_df.index.name = "row_id"
        submission_df.to_csv("submission.csv", index=True)
        print(submission_df.head())

#=================================================
#    Utility functions to aid model evaluation
#=================================================
from tqdm import tqdm
import polars as pl


def create_solution_df(dataset):
    """Create the solution dataframe for validating out-of-fold predictions.

    Parameters:
    dataset: (a subset of) the train dataframe
    
    Return:
    solution: DataFrame formatted for score() and mouse_fbeta()
    """
    solution = []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
    
        lab_id = row['lab_id']
        if lab_id.startswith('MABe22'): 
            continue

        video_id = row['video_id']
        path = f"{cwd}/Data/train_annotation/{lab_id}/{video_id}.parquet"
        try:
            annot = pd.read_parquet(path)
        except FileNotFoundError:
            # MABe22 and one more training file lack annotations.
            if verbose: 
                print(f"No annotations for {path}")
            continue
    
        # Add metadata fields required by scoring logic
        annot['lab_id'] = lab_id
        annot['video_id'] = video_id
        annot['behaviors_labeled'] = row['behaviors_labeled']

        # Normalize agent/target formatting to match submission expectations
        annot['target_id'] = np.where(
            annot.target_id != annot.agent_id,
            annot['target_id'].apply(lambda s: f"mouse{s}"),
            'self'
        )
        annot['agent_id'] = annot['agent_id'].apply(lambda s: f"mouse{s}")

        solution.append(annot)
    if solution == []:
        return solution
    else:
        solution = pd.concat(solution)
        return solution

def mouse_fbeta(solution: pd.DataFrame, submission: pd.DataFrame, beta: float = 1) -> float:
    """
    Official multi-label, multi-mouse, multi-lab evaluation metric.
    """
    if len(solution) == 0 or len(submission) == 0:
        raise ValueError('Missing solution or submission data')

    expected_cols = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
    for col in expected_cols:
        if col not in solution.columns:
            raise ValueError(f'Solution missing column {col}')
        if col not in submission.columns:
            raise ValueError(f'Submission missing column {col}')

    import polars as pl

    solution = pl.DataFrame(solution)
    submission = pl.DataFrame(submission)

    # Ensure valid frame intervals
    assert (solution['start_frame'] <= solution['stop_frame']).all()
    assert (submission['start_frame'] <= submission['stop_frame']).all()

    # Only evaluate on shared videos
    solution_videos = set(solution['video_id'].unique())
    submission = submission.filter(pl.col('video_id').is_in(solution_videos))

    # Build composite keys
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

    # Score per-lab
    lab_scores = []
    for lab in solution['lab_id'].unique():
        lab_solution = solution.filter(pl.col('lab_id') == lab).clone()
        lab_videos = set(lab_solution['video_id'].unique())
        lab_submission = submission.filter(pl.col('video_id').is_in(lab_videos)).clone()
        lab_scores.append(single_lab_f1(lab_solution, lab_submission, beta=beta))

    return sum(lab_scores) / len(lab_scores)

from collections import defaultdict

class HostVisibleError(Exception):
    pass

def single_lab_f1(lab_solution: pl.DataFrame, lab_submission: pl.DataFrame, beta: float = 1) -> float:
    label_frames: defaultdict[str, set[int]] = defaultdict(set)
    prediction_frames: defaultdict[str, set[int]] = defaultdict(set)

    # Build ground truth frame sets
    for row in lab_solution.to_dicts():
        label_frames[row['label_key']].update(
            range(row['start_frame'], row['stop_frame'])
        )

    # Submission event expansion
    for video in lab_solution['video_id'].unique():
        active_labels: set[str] = set(
            json.loads(lab_solution.filter(pl.col('video_id') == video)['behaviors_labeled'].first())
        )
        predicted_mouse_pairs = defaultdict(set)

        for row in lab_submission.filter(pl.col('video_id') == video).to_dicts():

            if ','.join([str(row['agent_id']), str(row['target_id']), row['action']]) not in active_labels:
                continue

            new_frames = set(range(row['start_frame'], row['stop_frame']))
            new_frames = new_frames.difference(prediction_frames[row['prediction_key']])
            prediction_pair = ','.join([str(row['agent_id']), str(row['target_id'])])

            if predicted_mouse_pairs[prediction_pair].intersection(new_frames):
                raise HostVisibleError("Multiple predictions for same frame from one agent/target")

            prediction_frames[row['prediction_key']].update(new_frames)
            predicted_mouse_pairs[prediction_pair].update(new_frames)

    # Count TP/FP/FN
    tps = defaultdict(int)
    fns = defaultdict(int)
    fps = defaultdict(int)

    for key, pred_frames in prediction_frames.items():
        action = key.split('_')[-1]
        gt_frames = label_frames[key]
        tps[action] += len(pred_frames.intersection(gt_frames))
        fns[action] += len(gt_frames.difference(pred_frames))
        fps[action] += len(pred_frames.difference(gt_frames))

    distinct_actions = set()

    for key, gt_frames in label_frames.items():
        action = key.split('_')[-1]
        distinct_actions.add(action)
        if key not in prediction_frames:
            fns[action] += len(gt_frames)

    action_f1s = []
    for action in distinct_actions:
        TP, FN, FP = tps[action], fns[action], fps[action]
        if TP + FN + FP == 0:
            action_f1s.append(0)
        else:
            action_f1s.append((1 + beta**2) * TP / ((1 + beta**2) * TP + beta**2 * FN + FP))

    return sum(action_f1s) / len(action_f1s)

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, beta: float = 1) -> float:
    solution = solution.drop(row_id_column_name, axis='columns', errors='ignore')
    submission = submission.drop(row_id_column_name, axis='columns', errors='ignore')
    return mouse_fbeta(solution, submission, beta=beta)


#============================================
#             Dev mode code
#============================================

if RUN_MODE == "dev":
    print("\n========== DEV MODE SANITY CHECK ==========\n")
    device = torch.device("cuda" if torch.cuda.is_available() else GLOBAL_CONFIG.get("device", "cpu"))
    print(f"Using device for dev check: {device}")

    if len(body_parts_tracked_list) == 0:
        print("No body-part sets found for dev check.")
    else:
        total_sets = len(body_parts_tracked_list)
        dev_f1_scores: list[float] = []
        for bp_idx, bp_str in enumerate(body_parts_tracked_list):
            bp_slug = slugify_bodyparts(bp_str)
            body_parts_tracked = filter_tracked_body_parts(json.loads(bp_str))
            ordered_joints, adjacency = get_ordered_joints_and_adjacency(body_parts_tracked)
            print(f"\n[DEV] Body-part set {bp_idx + 1}/{total_sets}: {bp_slug}")
            print(f"[DEV] Tracking {body_parts_tracked}")

            bp_rows = train[train.body_parts_tracked == bp_str]
            unique_vids = bp_rows.video_id.unique()
            #if len(unique_vids) < 10:
             #   print(f"[DEV] Not enough videos for 5 train / 5 test (found {len(unique_vids)}) - skipping {bp_slug}.")
              #  continue
            train_vids = unique_vids[:5]
            test_vids = unique_vids[5:10]
            train_subset = bp_rows[bp_rows.video_id.isin(train_vids)]
            test_subset = bp_rows[bp_rows.video_id.isin(test_vids)]

            dev_config = CTRGCNConfig(
                mode="dev",
                max_videos=None,  # limit handled by slicing above
                max_batches=20,
                max_windows=100,
                window=64,
                stride=32,
                show_progress=True,
                stream_mode=GLOBAL_CONFIG.get("stream_mode", "one"),
            )

            # Train models on first 5 videos (single + pair)
            train_batches: list = []
            for switch, data_df, meta_df, label_df in generate_mouse_data(
                train_subset,
                "train",
                traintest_directory=traintest_directory_path,
                generate_single=True,
                generate_pair=True,
                config=dev_config,
            ):
                train_batches.append((switch, data_df, meta_df, label_df))

            if len(train_batches) == 0:
                print(f"[DEV] No training batches generated for {bp_slug} - skipping.")
                continue

            model_dict_all = train_ctr_gcn_models(train_batches, ordered_joints, adjacency, dev_config, device=device)

            # Inference on next 5 videos
            submissions_local: list[pd.DataFrame] = []
            for switch, data_df, meta_df, label_df in generate_mouse_data(
                test_subset,
                "train",
                traintest_directory=traintest_directory_path,
                generate_single=True,
                generate_pair=True,
                config=dev_config,
            ):
                actions_te = list(label_df.columns)
                mode = getattr(dev_config, "stream_mode", "one")
                data_prepared = data_df
                ordered_for_input = ordered_joints
                if switch == "pair":
                    data_prepared, ordered_for_input = flatten_pair_dataframe(data_df, ordered_joints)

                if mode == "one":
                    window_tensor, frame_ranges = prepare_ctr_gcn_input(data_prepared, ordered_for_input, dev_config)
                    if window_tensor.shape[0] == 0:
                        continue
                    X = window_tensor.to(device)
                elif mode == "two":
                    streamA_list, streamB_list, frame_ranges = prepare_ctr_gcn_input(data_prepared, ordered_for_input, dev_config)
                    if len(streamA_list) == 0:
                        continue
                    X_streamA = torch.stack(streamA_list, dim=0).to(device)
                    X_streamB = torch.stack(streamB_list, dim=0).to(device)
                else:
                    coords_list, delta_list, bone_list, bone_delta_list, frame_ranges = prepare_ctr_gcn_input(data_prepared, ordered_for_input, dev_config)
                    if len(coords_list) == 0:
                        continue
                    X_coords = torch.stack(coords_list, dim=0).to(device)
                    X_delta = torch.stack(delta_list, dim=0).to(device)
                    X_bone = torch.stack(bone_list, dim=0).to(device)
                    X_bone_delta = torch.stack(bone_delta_list, dim=0).to(device)

                frame_values = meta_df.video_frame.values
                frame_to_idx = {f: i for i, f in enumerate(frame_values)}
                actions_available = [a for a in actions_te if a in model_dict_all[switch]]
                if not actions_available:
                    continue
                n_frames = len(frame_values)
                n_actions = len(actions_available)
                sum_probs = np.zeros((n_frames, n_actions), dtype=np.float32)
                counts = np.zeros((n_frames, n_actions), dtype=np.float32)

                for action_idx, action_name in enumerate(actions_available):
                    model = model_dict_all[switch][action_name]
                    with torch.no_grad():
                        if mode == "one":
                            logits = model(X).squeeze(-1)
                        elif mode == "two":
                            logits = model(X_streamA, X_streamB).squeeze(-1)
                        else:
                            logits = model(X_coords, X_delta, X_bone, X_bone_delta).squeeze(-1)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    for w_idx, frames in enumerate(frame_ranges):
                        for t, f in enumerate(frames):
                            fi = frame_to_idx.get(f)
                            if fi is None or t >= probs.shape[1]:
                                continue
                            p = float(probs[w_idx, t])
                            sum_probs[fi, action_idx] += p
                            counts[fi, action_idx] += 1.0

                counts[counts == 0] = 1.0
                pred_array = sum_probs / counts
                pred_df = pd.DataFrame(pred_array, index=meta_df.video_frame, columns=actions_available)
                thresh = getattr(dev_config, "decision_threshold", 0.27)
                submission_part = predict_multiclass(pred_df, meta_df, threshold=thresh)
                submissions_local.append(submission_part)

            if submissions_local:
                submission_df = pd.concat(submissions_local, ignore_index=True)
            else:
                submission_df = pd.DataFrame(columns=["video_id", "agent_id", "target_id", "action", "start_frame", "stop_frame"])

            # Build solution for the dev test subset
            solution_df = create_solution_df(test_subset)
            f1_val = 0.0
            if len(submission_df) == 0:  # Prevent no actions found crashing program
                print("[DEV] No predicted events - skipping scoring.")
            else:
                f1_val = score(solution_df, submission_df, row_id_column_name="", beta=1)
                print("DEV F1 =", f1_val)

            print("\nDEV SUMMARY")
            print(f"Body parts set: {bp_slug}")
            print(f"Train videos: {len(train_vids)}, Test videos: {len(test_vids)}")
            actions_trained = len(model_dict_all['single']) + len(model_dict_all['pair'])
            print(f"Actions trained (single+pair): {actions_trained}")
            print(f"Validation F1 (mouse_fbeta-style): {f1_val:.4f}")
            dev_f1_scores.append(f1_val)

        if dev_f1_scores:
            overall_f1 = sum(dev_f1_scores) / len(dev_f1_scores)
            print(f"\n[DEV] Average holdout F1 across body-part sets: {overall_f1:.4f}")

if RUN_MODE == "validate":
    for bp_str in body_parts_tracked_list:
        train_subset = train[train.body_parts_tracked == bp_str]
        if len(train_subset) == 0:
            continue
        bp_slug = slugify_bodyparts(bp_str)
        cfg = CTRGCNConfig(mode="validate", stream_mode=GLOBAL_CONFIG.get("stream_mode", "one"))
        best_params = load_best_params_csv_for_config(cfg, bp_slug=bp_slug)
        if best_params is not None:
            print("Using tuned hyperparameters for stream_mode =", cfg.stream_mode, "body parts", bp_slug)
            for key, value in best_params.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
        else:
            print(f"No best_params file found for stream_mode {cfg.stream_mode} body parts {bp_slug} - using defaults.")

        ordered_joints, adjacency = get_ordered_joints_and_adjacency(filter_tracked_body_parts(json.loads(bp_str)))

        # Holdout dev-style evaluation for foresight (5 epochs via submit mode)
        unique_vids = train_subset.video_id.unique()
        best_holdout_f1 = 0.0
        best_holdout_models = None
        if len(unique_vids) >= 2:
            # 80/20 split
            rng = np.random.default_rng(42)
            perm = rng.permutation(unique_vids)
            split_idx = max(1, int(0.8 * len(perm)))
            train_vids = perm[:split_idx]
            val_vids = perm[split_idx:]
            if len(val_vids) == 0:  # ensure at least 1 val video
                val_vids = perm[-1:]
                train_vids = perm[:-1]
            train_part = train_subset[train_subset.video_id.isin(train_vids)]
            val_part = train_subset[train_subset.video_id.isin(val_vids)]

            eval_cfg = CTRGCNConfig(mode="submit", stream_mode=cfg.stream_mode)
            for key, value in (best_params or {}).items():
                if hasattr(eval_cfg, key):
                    setattr(eval_cfg, key, value)
            eval_cfg.max_batches = None
            eval_cfg.max_windows = None

            eval_batches = []
            for switch, data_df, meta_df, label_df in generate_mouse_data(
                train_part, "train", generate_single=True, generate_pair=True, config=eval_cfg
            ):
                eval_batches.append((switch, data_df, meta_df, label_df))

            if eval_batches:
                model_eval = train_ctr_gcn_models(eval_batches, ordered_joints, adjacency, eval_cfg, device=GLOBAL_CONFIG["device"])

                submissions_local: list[pd.DataFrame] = []
                for switch, data_df, meta_df, label_df in generate_mouse_data(
                    val_part, "train", generate_single=True, generate_pair=True, config=eval_cfg
                ):
                    actions_te = list(label_df.columns)
                    mode = getattr(eval_cfg, "stream_mode", "one")
                    if mode == "one":
                        window_tensor, frame_ranges = prepare_ctr_gcn_input(data_df, ordered_joints, eval_cfg)
                        if window_tensor.shape[0] == 0:
                            continue
                        X = window_tensor.to(GLOBAL_CONFIG["device"])
                    elif mode == "two":
                        streamA_list, streamB_list, frame_ranges = prepare_ctr_gcn_input(data_df, ordered_joints, eval_cfg)
                        if len(streamA_list) == 0:
                            continue
                        X_streamA = torch.stack(streamA_list, dim=0).to(GLOBAL_CONFIG["device"])
                        X_streamB = torch.stack(streamB_list, dim=0).to(GLOBAL_CONFIG["device"])
                    else:
                        coords_list, delta_list, bone_list, bone_delta_list, frame_ranges = prepare_ctr_gcn_input(data_df, ordered_joints, eval_cfg)
                        if len(coords_list) == 0:
                            continue
                        X_coords = torch.stack(coords_list, dim=0).to(GLOBAL_CONFIG["device"])
                        X_delta = torch.stack(delta_list, dim=0).to(GLOBAL_CONFIG["device"])
                        X_bone = torch.stack(bone_list, dim=0).to(GLOBAL_CONFIG["device"])
                        X_bone_delta = torch.stack(bone_delta_list, dim=0).to(GLOBAL_CONFIG["device"])

                    frame_values = meta_df.video_frame.values
                    frame_to_idx = {f: i for i, f in enumerate(frame_values)}
                    actions_available = [a for a in actions_te if a in model_eval[switch]]
                    if not actions_available:
                        continue
                    sum_probs = np.zeros((len(frame_values), len(actions_available)), dtype=np.float32)
                    counts = np.zeros((len(frame_values), len(actions_available)), dtype=np.float32)

                    for action_idx, action_name in enumerate(actions_available):
                        model = model_eval[switch][action_name]
                        with torch.no_grad():
                            if mode == "one":
                                logits = model(X).squeeze(-1)
                            elif mode == "two":
                                logits = model(X_streamA, X_streamB).squeeze(-1)
                            else:
                                logits = model(X_coords, X_delta, X_bone, X_bone_delta).squeeze(-1)
                        probs = torch.sigmoid(logits).cpu().numpy()
                        for w_idx, frames in enumerate(frame_ranges):
                            for t, f in enumerate(frames):
                                fi = frame_to_idx.get(f)
                                if fi is None or t >= probs.shape[1]:
                                    continue
                                p = float(probs[w_idx, t])
                                sum_probs[fi, action_idx] += p
                                counts[fi, action_idx] += 1.0

                    counts[counts == 0] = 1.0
                    pred_array = sum_probs / counts
                    pred_df = pd.DataFrame(pred_array, index=meta_df.video_frame, columns=actions_available)
                    thresh = getattr(eval_cfg, "decision_threshold", 0.27)
                    submission_part = predict_multiclass(pred_df, meta_df, threshold=thresh)
                    submissions_local.append(submission_part)

                if submissions_local:
                    submission_df = pd.concat(submissions_local, ignore_index=True)
                else:
                    submission_df = pd.DataFrame(columns=["video_id", "agent_id", "target_id", "action", "start_frame", "stop_frame"])

                solution_df = create_solution_df(val_part)
                if len(submission_df) == 0:
                    print(f"[VALIDATE] No predicted events on holdout for {bp_slug} - skipping scoring.")
                else:
                    holdout_f1 = score(solution_df, submission_df, row_id_column_name="", beta=1)
                    best_holdout_f1 = holdout_f1
                    best_holdout_models = model_eval
                    print(f"[VALIDATE] Holdout F1 for {bp_slug}: {holdout_f1:.4f}")
            else:
                print(f"[VALIDATE] No training batches for holdout split in {bp_slug}.")
        else:
            print(f"[VALIDATE] Not enough videos for holdout split (found {len(unique_vids)}) for {bp_slug}.")

        batches = []
        for switch, data_df, meta_df, label_df in generate_mouse_data(train_subset, "train", generate_single=True, generate_pair=True, config=cfg):
            batches.append((switch, data_df, meta_df, label_df))

        model_dict_all = train_ctr_gcn_models(batches, ordered_joints, adjacency, cfg, device=GLOBAL_CONFIG["device"])
        model_dir = get_stream_model_dir(cfg, bp_slug=bp_slug)
        for action, model in model_dict_all["single"].items():
            path_single = os.path.join(model_dir, "single")
            os.makedirs(path_single, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(path_single, f"{action}.pt"))
        for action, model in model_dict_all["pair"].items():
            path_pair = os.path.join(model_dir, "pair")
            os.makedirs(path_pair, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(path_pair, f"{action}.pt"))
        print(f"Saved models to {model_dir}")
        if best_holdout_models is not None:
            print(f"[VALIDATE] Best holdout F1 for {bp_slug}: {best_holdout_f1:.4f} (saved _bestdev checkpoints)")
            for action, model in best_holdout_models["single"].items():
                path_single = os.path.join(model_dir, "single")
                os.makedirs(path_single, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(path_single, f"{action}_bestdev.pt"))
            for action, model in best_holdout_models["pair"].items():
                path_pair = os.path.join(model_dir, "pair")
                os.makedirs(path_pair, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(path_pair, f"{action}_bestdev.pt"))



end_time = time.time()
elapsed = end_time - start_time
print(f"Execution time: {elapsed:.4f} seconds")
