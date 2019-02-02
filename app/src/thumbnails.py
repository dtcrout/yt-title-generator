"""Download YouTube video thumbnails from metadata."""

import os
import pandas as pd
import requests
import shutil

METADATA = '../resources/USvideos.csv'
THUMBNAILS_DIR = '../resources/thumbnails/'

if __name__ == "__main__":
    # Import CSV and grab video ids and thumbnail links
    yt_df = pd.read_csv(METADATA)
    yt_df = yt_df[['video_id', 'thumbnail_link']]

    video_ids = list(yt_df.video_id.values)
    thumbnail_links = list(yt_df.thumbnail_link.values)

    # Download thumbnails
    for video_id, thumbnail_link in zip(video_ids, thumbnail_links):
        img_path = THUMBNAILS_DIR + video_id + '.jpg'

        if not os.path.exists(img_path):
            r = requests.get(thumbnail_link, stream=True)

            if r.status_code == 200:
                print('Downloading', video_id, '...')
                with open(img_path, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
