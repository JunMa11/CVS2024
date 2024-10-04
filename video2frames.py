# Import necessary libraries
# from PIL import Image
# import matplotlib.pyplot as plt
# from IPython.display import clear_output
# import numpy as np
# import pandas as pd
# salloc -c 40 -t 6:0:0 --mem 300G --partition=gpu_bwanggroup

import os
import subprocess
import multiprocessing as mp
from tqdm import tqdm

video_path = 'data-CVS/videos'
frames_path="data-CVS/train2/frames"
os.makedirs(frames_path, exist_ok=True)

# convert video to frame
video_files = sorted([f for f in os.listdir(video_path) if f.endswith('.mp4')])
print('video num:', len(video_files))
# continue with your input
input('Press Enter to continue...')

# for video_file in video_files:
def convert_video_to_frames(video_file):
    base_name = os.path.splitext(os.path.basename(video_file))[0]
    output_folder = os.path.join(frames_path, base_name)
    os.makedirs(output_folder, exist_ok=True)
    ffmpeg_command = [
        'ffmpeg', '-i', os.path.join(video_path, video_file),
        '-vf', 'fps=30', '-q:v', '2',
        os.path.join(output_folder, f'{base_name}_%04d.jpg')
    ]
    subprocess.run(ffmpeg_command)

# labels_path = '/home/jma/Documents/CVS/train/labels'
# os.makedirs(labels_path, exist_ok=True)

if __name__ == '__main__':
    with mp.Pool(40) as p:
        r = list(tqdm(p.imap(convert_video_to_frames, video_files), total=len(video_files)))

