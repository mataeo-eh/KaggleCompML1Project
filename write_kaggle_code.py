import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from pathlib import Path
import seaborn
import pyarrow.parquet as pq

def find_column_headers():
    data_dir = Path("/kaggle/input/MABe-mouse-behavior-detection") # 1. Set root directory
    parquet_files = list(data_dir.rglob("*.parquet")) # 2. Read only column names (schema) from each file
    file_columns = {}
    max_cols = 0
    for f in parquet_files:  # Construct index: "grandparent/parent/filename"
        index_str = f.parent.parent.name + "/" + f.parent.name + "/" + f.name
        cols = pq.read_schema(f).names
        file_columns[index_str] = cols
        max_cols = max(max_cols, len(cols))
    df_cols = pd.DataFrame(index=list(file_columns.keys()), columns=range(max_cols)) # 3. Create a DataFrame: rows = files, columns = enumerated column positions
    for file_stem, cols in file_columns.items(): # 4. Fill DataFrame with column names
        df_cols.loc[file_stem, :len(cols)-1] = cols
    print("Column consistency table (first 10 rows):") # 5. Quick inspection
    print(df_cols.head(10))
    print("Column consistency table (last 10 rows):")
    print(df_cols.tail(10))
    files_with_missing_cols = df_cols[df_cols.isnull().any(axis=1)] # 6. Find files with missing or extra columns
    print("\nFiles with missing columns:")
    print(files_with_missing_cols)
    unique_columns = pd.unique(df_cols.values.ravel()) # 7. List all unique column names across all files
    unique_columns = [c for c in unique_columns if c is not None]
    print(f"\nUnique column names across all files ({len(unique_columns)}):")
    print(unique_columns)
    df_t = df_cols.T  # transpose
    mask = df_t.nunique() != 1 # 8. Check which column positions have inconsistent names across files
    inconsistent_columns = df_t.loc[:, mask]
    print("\nColumn positions with inconsistent names across files:")
    print(inconsistent_columns)
    files_with_inconsistencies = inconsistent_columns[inconsistent_columns.notnull().any(axis=1)]
    print("\nFiles with inconsistencies at these column positions:") # Optional: show which files have inconsistencies in these columns
    print(files_with_inconsistencies)
import pandas as pd
from pathlib import Path
import os

def save_mouse_sequences_normalized(
    tracking_dir: str,
    annotation_dir: str,
    output_dir: str = "/kaggle/working/mouse_sequences"
):
    """
    Process all tracking+annotation files into per-video per-mouse grouped
    sequence DataFrames, saving each as a parquet file with a consistent column schema.
    Missing bodyparts are filled with NaN. Overwrites existing files and includes debug messages.

    Parameters
    ----------
    tracking_dir : str
        Directory containing tracking parquet files.
    annotation_dir : str
        Directory containing annotation parquet files.
    output_dir : str, default="/kaggle/working/mouse_sequences"
        Where to save the per-video outputs. Kaggle will keep this
        in the commit snapshot so you can make a dataset from it.
    """
    os.makedirs(output_dir, exist_ok=True)

    tracking_files = list(Path(tracking_dir).rglob("*.parquet"))
    print(f"Found {len(tracking_files)} tracking files")

    # First pass: collect all possible bodyparts to define consistent columns
    all_bodyparts = set()
    for tracking_file in tracking_files:
        try:
            df = pd.read_parquet(tracking_file)
            if "bodypart" in df.columns:
                all_bodyparts.update(df["bodypart"].dropna().unique())
        except Exception as e:
            print(f"⚠️ Could not read {tracking_file} to collect bodyparts: {e}")

    all_bodyparts = sorted(all_bodyparts)
    print(f"Detected {len(all_bodyparts)} bodyparts: {all_bodyparts}")

    # Define index columns and full final columns
    index_cols = ["video_id", "video_frame", "mouse_id", "action", "agent_id", "target_id"]
    bodypart_cols = [f"{bp}_{coord}" for bp in all_bodyparts for coord in ["x", "y"]]
    final_columns = index_cols + bodypart_cols

    # Process each tracking file
    for tracking_file in tracking_files:
        video_id = Path(tracking_file).stem
        annotation_file = Path(annotation_dir) / Path(tracking_file).parent.name / f"{video_id}.parquet"

        try:
            tracking_df = pd.read_parquet(tracking_file).sort_values("video_frame").reset_index(drop=True)
        except Exception as e:
            print(f"⚠️ Could not read {tracking_file}: {e}")
            continue

        if tracking_df.empty:
            print(f"⚠️ Tracking file empty: {tracking_file}. Skipping.")
            continue

        # Add annotation columns
        tracking_df["action"] = None
        tracking_df["agent_id"] = None
        tracking_df["target_id"] = None
        tracking_df["video_id"] = video_id

        # Apply annotations if available
        if annotation_file.exists():
            try:
                annotation_df = pd.read_parquet(annotation_file)
                for _, row in annotation_df.iterrows():
                    mask = (tracking_df.video_frame >= row.start_frame) & (tracking_df.video_frame <= row.stop_frame)
                    tracking_df.loc[mask, "action"] = row.action
                    tracking_df.loc[mask, "agent_id"] = row.agent_id
                    tracking_df.loc[mask, "target_id"] = row.target_id
            except Exception as e:
                print(f"⚠️ Could not read annotation {annotation_file}: {e}")

        # Pivot safely
        try:
            pivoted = tracking_df.pivot(
                index=index_cols,
                columns="bodypart",
                values=["x", "y"]
            )
            if pivoted.shape[1] <= 6:
                print(f"⚠️ Pivot resulted in no bodypart columns for {tracking_file}. Skipping.")
                continue

            pivoted.columns = [f"{bp}_{coord}" for coord, bp in pivoted.columns]
            pivoted = pivoted.reset_index()
        except Exception as e:
            print(f"⚠️ Pivot failed for {tracking_file}: {e}")
            continue

        # Ensure consistent columns: add missing, reorder
        for col in bodypart_cols:
            if col not in pivoted.columns:
                pivoted[col] = pd.NA
        pivoted = pivoted[final_columns]

        # Save, overwrite if exists
        out_path = Path(output_dir) / f"{video_id}.parquet"
        try:
            pivoted.to_parquet(out_path, index=False)
            print(f"✅ Saved {out_path} with shape {pivoted.shape}")
        except Exception as e:
            print(f"⚠️ Could not save {out_path}: {e}")

    print(f"\n✅ All processed files saved to {output_dir}")


# ------------------------------
# Example usage
# ------------------------------
save_mouse_sequences_normalized(
    tracking_dir="/kaggle/input/MABe-mouse-behavior-detection/train_tracking",
    annotation_dir="/kaggle/input/MABe-mouse-behavior-detection/train_annotation",
    output_dir="/kaggle/working/mouse_sequences"
)





import pandas as pd
from pathlib import Path
from tqdm import tqdm

def save_combined_mouse_sequences_uniform(
    tracking_dir: str,
    annotation_dir: str,
    annotated_output: str,
    unannotated_output: str,
    use_pyarrow: bool = True
):
    """
    Incrementally process all tracking and annotation files into a single combined
    annotated and unannotated Parquet file each, guaranteeing uniform columns across
    all videos. Includes debugging logs for empty or problematic files.
    """
    tracking_dir = Path(tracking_dir)
    annotation_dir = Path(annotation_dir)
    annotated_output = Path(annotated_output)
    unannotated_output = Path(unannotated_output)

    tracking_files = list(tracking_dir.rglob("*.parquet"))
    annotation_files = list(annotation_dir.rglob("*.parquet"))
    annotation_map = {f.name: f for f in annotation_files}

    # Remove existing combined files to start fresh
    if annotated_output.exists():
        annotated_output.unlink()
    if unannotated_output.exists():
        unannotated_output.unlink()

    # Collect all possible columns first
    all_columns = set([
        "video_id", "video_frame", "mouse_id",
        "action", "agent_id", "target_id"
    ])
    bodyparts_seen = set()

    # First pass: detect all bodyparts
    for track_file in tqdm(tracking_files, desc="Detecting bodyparts"):
        try:
            df = pd.read_parquet(track_file)
            bodyparts_seen.update(df["bodypart"].unique())
        except Exception as e:
            tqdm.write(f"Error reading {track_file}: {e}")

    # Add x/y columns for each bodypart
    for bp in bodyparts_seen:
        all_columns.add(f"{bp}_x")
        all_columns.add(f"{bp}_y")

    # Second pass: process and append
    for track_file in tqdm(tracking_files, desc="Processing videos"):
        video_id = track_file.stem
        tracking_df = pd.read_parquet(track_file).sort_values("video_frame").reset_index(drop=True)
        tracking_df["action"] = None
        tracking_df["agent_id"] = None
        tracking_df["target_id"] = None
        tracking_df["video_id"] = video_id

        # Apply annotations if available
        annot_file = annotation_map.get(track_file.name)
        is_annotated = False
        if annot_file and annot_file.exists():
            annotation_df = pd.read_parquet(annot_file)
            for _, row in annotation_df.iterrows():
                mask = (tracking_df.video_frame >= row.start_frame) & (tracking_df.video_frame <= row.stop_frame)
                tracking_df.loc[mask, "action"] = row.action
                tracking_df.loc[mask, "agent_id"] = row.agent_id
                tracking_df.loc[mask, "target_id"] = row.target_id
            is_annotated = True

        # Pivot
        try:
            pivoted = tracking_df.pivot_table(
                index=["video_id", "video_frame", "mouse_id", "action", "agent_id", "target_id"],
                columns="bodypart",
                values=["x", "y"]
            )
            if pivoted.empty:
                tqdm.write(f"Pivot resulted in empty DataFrame for {track_file}. Skipping.")
                continue
            pivoted.columns = [f"{bp}_{coord}" for coord, bp in pivoted.columns]
            pivoted = pivoted.reset_index()
        except Exception as e:
            tqdm.write(f"Error pivoting {track_file}: {e}")
            continue

        # Ensure uniform columns
        missing_cols = all_columns - set(pivoted.columns)
        for c in missing_cols:
            pivoted[c] = pd.NA
        pivoted = pivoted[list(all_columns)]  # enforce column order

        # Append to file
        output_path = annotated_output if is_annotated else unannotated_output
        if output_path.exists():
            existing_df = pd.read_parquet(output_path, engine="pyarrow")
            combined_df = pd.concat([existing_df, pivoted], ignore_index=True)
            combined_df.to_parquet(output_path, engine="pyarrow", index=False)
        else:
            pivoted.to_parquet(output_path, engine="pyarrow", index=False)

        tqdm.write(f"Processed {video_id} -> {output_path.name}, shape={pivoted.shape}")

    print(f"✅ All videos processed. Annotated: {annotated_output}, Unannotated: {unannotated_output}")
