import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from pathlib import Path
import seaborn
import pyarrow.parquet as pq
cwd = Path.cwd()
import json

'''
train = pd.read_csv(cwd / "Data" / "train.csv")
train["n_mice"] = 4 - train[["mouse1_strain", "mouse2_strain", "mouse3_strain", "mouse4_strain"]].isna().sum(axis=1)
train_without_mabe22 = train.query("~ lab_id.str.startswith('MABe22_')")
test = pd.read_csv(cwd / "Data" / "test.csv")
body_parts_tracked_list = sorted(train.body_parts_tracked.unique())

print("\n")
print("====================================================")
print("DEBUGGING CODE FOR THE BODY PARTS TRACKED LIST OF UNIQUE BODY PARTS")
print(body_parts_tracked_list)


for bp_str in body_parts_tracked_list:
    train_subset_full = train[train.body_parts_tracked == bp_str]
    print("My body parts tracked is \n",train.body_parts_tracked)
    print(f"My bpstring is {bp_str}")
    print("my trainsubset is \n",train_subset_full)
'''

"""
drop_body_parts = []

for section in range(1, len(body_parts_tracked_list)): # skip index 0 (MABe22)
    body_parts_tracked_str = body_parts_tracked_list[section]
    body_parts_tracked = json.loads(body_parts_tracked_str)
    print(f"{section}. Processing videos with {body_parts_tracked}")
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
        print("Ambrosm body parts tracked is \n",body_parts_tracked)

    # We read all training data which match the body parts tracked
    train_subset = train[train.body_parts_tracked == body_parts_tracked_str]
    print("trainsubset from Ambroms is \n",train_subset)
    """

'''

        # MORE DEBUGGING CODE
        
        # Check uniqueness
        print("mouse_id unique:", vid['mouse_id'].unique())
        print("bodypart unique:", vid['bodypart'].unique())
        counts = (
            vid.groupby(["mouse_id", "bodypart"])["video_frame"]
            .nunique()
            .reset_index(name="frames_present")
        )

        total_frames = vid["video_frame"].nunique()

        missing = counts[counts["frames_present"] < total_frames]
        missing.sort_values(by='frames_present', inplace=True, ascending=False)
        missing.to_csv('missing_stuff', index=False)
        print(f" the missing stuff is \n{missing}")
        print(f" The total frames is {total_frames}")


        # Check duplicates
        dupes = vid.duplicated(subset=["video_frame","mouse_id","bodypart"], keep=False)
        print("Duplicate rows:", dupes.sum())

        # Check if x/y are numeric
        print(vid[['x','y']].dtypes)
        
            # Identify missing body parts
            missing_mask = pvid.isna().any(axis=0)   # True for columns that contain any NaN
            missing_cols = pvid.columns[missing_mask]

            print("Missing/misaligned columns (mouse_id, bodypart, coord):")
            for col in missing_cols:
                mouse_id, bodypart, coord = col  # because MultiIndex: (mouse_id, bodypart, x/y)
                print(f" - mouse id = {mouse_id}: missing \nBodypart = {bodypart} \ncoord=({coord})")

'''
from pprint import pprint

train = pd.read_csv(cwd / "Data" / "train.csv")
train["n_mice"] = 4 - train[["mouse1_strain", "mouse2_strain", "mouse3_strain", "mouse4_strain"]].isna().sum(axis=1)
train_without_mabe22 = train.query("~ lab_id.str.startswith('MABe22_')")
test = pd.read_csv(cwd / "Data" / "test.csv")
body_parts_tracked_list = sorted(train.body_parts_tracked.unique())

print("\nBody parts tracked and file counts:\n")
for idx, bp in enumerate(body_parts_tracked_list):
    count = train[train.body_parts_tracked == bp].shape[0]
    print(f"{idx:2d} | {count:4d} files | {bp}")

body_parts_tracked_list = sorted(train_without_mabe22.body_parts_tracked.unique())

print("\nBody parts tracked and file counts excluding mabe22:\n")
for idx, bp in enumerate(body_parts_tracked_list):
    count = train[train.body_parts_tracked == bp].shape[0]
    print(f"{idx:2d} | {count:4d} files | {bp}")