"""Create titles for training."""

import pandas as pd
import re
import os

METADATA = '../resources/USvideos.csv'
THUMBNAILS_DIR = '../resources/thumbnails/'
TITLES = '../resources/titles.txt'

def preprocess_text(text):
    """
    Preprocess text.

    - Make text lower case
    - Remove non-alphanumeric characters
    """
    tokens = re.split('\W+', text.lower())
    return ' '.join(tokens)


if __name__ == "__main__":
    # Import CSV and grab video ids and titles
    yt_df = pd.read_csv(METADATA)
    yt_df = yt_df[['video_id', 'title']]

    video_ids = list(yt_df.video_id.values)
    titles = list(yt_df.title.values)

    # Store video ids and titles in dictionary
    yt = dict()
    for video_id, title in zip(video_ids, titles):
        yt[video_id] = title

    # Get video ids of downloaded thumbnails
    dled = [f.strip('.jpg') for f in os.listdir(THUMBNAILS_DIR)]

    # Write preprocessed titles to file with video id
    with open(TITLES, 'w') as f:
        for thumb in dled:
            if thumb in yt.keys():
                title = preprocess_text(yt[thumb])
                f.write(thumb + '\t' + title +'\n')

