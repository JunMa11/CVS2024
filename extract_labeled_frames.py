import glob
import os
from statistics import mode
import json
import numpy as np
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm

frames_path = '/home/jma/Documents/CVS/train/frames'
labels_path = '/home/jma/Documents/CVS/train/labels'

def is_annotated(fname):
    frame_id = int(os.path.splitext(os.path.basename(fname))[0].split('_')[-1]) - 1
    return frame_id % 150 == 0

def raw_labels(row, category):
    labels = [
        row[f"{category}_rater1"].iloc[0],
        row[f"{category}_rater2"].iloc[0],
        row[f"{category}_rater3"].iloc[0],
    ]
    return labels

def majority_vote(row, category):
    labels = [
        row[f"{category}_rater1"].iloc[0],
        row[f"{category}_rater2"].iloc[0],
        row[f"{category}_rater3"].iloc[0],
    ]
    return mode(labels)

def confidence_multiplexed_majority_vote(row, category,confidences):
    labels = np.array([
        row[f"{category}_rater1"].iloc[0],
        row[f"{category}_rater2"].iloc[0],
        row[f"{category}_rater3"].iloc[0],
    ])
    labeler_confidences = np.array(confidences)
    confidence_aware_label = 0.5+1/3.0*np.dot(labels-0.5,labeler_confidences)
    return confidence_aware_label


search_pattern = os.path.join(frames_path, "**", "*.jpg")
frames = sorted(glob.glob(search_pattern, recursive=True))
image_paths = [file for file in frames if is_annotated(file)]

#%%
label_info = OrderedDict()
label_info['video_name'] = []
label_info['frame_name'] = []
label_info['c1'] = []
label_info['c2'] = []
label_info['c3'] = []
label_info['ca_c1'] = []
label_info['ca_c2'] = []
label_info['ca_c3'] = []

for img_path in tqdm(image_paths):
    video_name = img_path.split("/")[-2]
    frame_id = int(img_path.split("/")[-1].replace(".jpg", "").split('_')[-1]) - 1
    label_path = os.path.join(labels_path, f"{video_name}/frame.csv")
    video_label_path = os.path.join(labels_path, f"{video_name}/video.csv")

    # Load c1,c2,c3 label
    label_df = pd.read_csv(label_path)
    video_label_df = pd.read_csv(video_label_path)
    confidences = [float(video_label_df[f'confidence_rater{i+1}'].iloc[0]) for i in range(3)]
    label_df = label_df[label_df["frame_id"] == frame_id]
    c1, c2, c3 = (
        majority_vote(label_df, "c1"),
        majority_vote(label_df, "c2"),
        majority_vote(label_df, "c3"),
    )
    ca_c1 = confidence_multiplexed_majority_vote(label_df, "c1", confidences)
    ca_c2 = confidence_multiplexed_majority_vote(label_df, "c2", confidences)
    ca_c3 = confidence_multiplexed_majority_vote(label_df, "c3", confidences)
    # label = torch.as_tensor([c1, c2, c3], dtype=torch.float32)

    metadata = {}
    metadata["raw_labels"] = {}
    for key in ["c1", "c2", "c3"]:
        metadata["raw_labels"][key] = raw_labels(label_df, key)

    # Add label information to label_info
    label_info['video_name'].append(video_name)
    label_info['frame_name'].append(img_path)
    label_info['c1'].append(c1)
    label_info['c2'].append(c2)
    label_info['c3'].append(c3)
    label_info['ca_c1'].append(ca_c1)
    label_info['ca_c2'].append(ca_c2)
    label_info['ca_c3'].append(ca_c3)
    metadata["confidence_aware_labels"] = {}
    ca_labels = [ca_c1,ca_c2,ca_c3]
    for i,key in enumerate(["c1", "c2", "c3"]):
        metadata["confidence_aware_labels"][key] = ca_labels[i]

label_df = pd.DataFrame(label_info)
label_df.to_csv("labeled_frames.csv", index=False)

#%% save metadata as a json file
with open('labeled_frames_metadata.json', 'w') as f:
    json.dump(metadata, f)

