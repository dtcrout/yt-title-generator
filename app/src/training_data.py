"""Methods used to create training data.

This file contains methods to generate training data required for the
captioning model. Given the Kaggle YT dataset, we want to get video thumbnails
and their titles.

get_thumbnails(): Downloads video thumbnails
get_titles(): Creates list of titles of downloaded thumbnails
generate_image_features(): Creates image thumbnails using VGG16
"""

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import os
import pandas as pd
import pickle
import re
import requests
import shutil

# metadata_files = ["../resources/USvideos.csv", "../resources/GBvideos.csv"]
metadata_files = ["../resources/USvideos.csv"]
thumbnails_dir = "../resources/thumbnails/"
titles_path = "../resources/titles.txt"
features_path = "../resources/features.pkl"


def preprocess_text(text):
    """
    Preprocess text by:
        - Making text lower case
        - Removing non-alphanumeric characters
        - Remove a's

    Args
    ----
    text: str

    Returns
    -------
    The original text with all lower case characters
    and no non-alphanumeric characters.
    """
    tokens = re.split("\W+", text.lower())
    clean_tokens = [t for t in tokens if t not in ["a"]]
    return " ".join(clean_tokens)


def load_data(metadata_files):
    """
    Import metadata CSV files and grab video ids and thumbnail links.

    Args
    ----
    metadata_files: list
        List of filepaths to metadata CSV files.

    Returns
    -------
    yt_dict: dict
        Dictionary of titles and thumbnail links with video_id as key.
    """
    yt_dict = dict()

    for metadata in metadata_files:
        yt_df = pd.read_csv(metadata, error_bad_lines=False)
        yt_df = yt_df[["video_id", "title", "thumbnail_link"]]

        video_ids = list(yt_df.video_id.values)
        titles = list(yt_df.title.values)
        thumbnails = list(yt_df.thumbnail_link.values)

        for video_id, title, thumbnail in zip(video_ids, titles, thumbnails):
            yt_dict[video_id] = {"title": title, "thumbnail": thumbnail}

    return yt_dict


def get_thumbnails(yt_data, thumbnails_dir):
    """
    Download video thumbnails.

    Args
    ----
    yt_data: dict
        Dictionary of titles and thumbnail links with video_id as key.
    thumbnails_dir: str
        Path to save thumbnails.
    """
    for video_id, values in yt_data.items():
        thumbnail_link = values["thumbnail"]
        img_path = thumbnails_dir + video_id + ".jpg"

        if not os.path.exists(img_path):
            print("Downloading", video_id, "...")
            try:
                r = requests.get(thumbnail_link, stream=True, timeout=10)

                if r.status_code == 200:
                    with open(img_path, "wb") as f:
                        shutil.copyfileobj(r.raw, f)
                else:
                    print(r.status_code, "Error retrieving", video_id)
            except Exception as e:
                print("Error:", e)
        else:
            print(video_id, "already exists!")


def get_titles(yt_data, thumbnails_dir, titles_path):
    """
    Create list of video titles.

    Args
    ----
    yt_data: dict
        Dictionary of titles and thumbnail links with video_id as key.
    thumbnails_dir: str
        Path of save thumbnails.
    titles_path: str
        Path to save title list.
    """
    # Get video ids of downloaded thumbnails
    dled = [f.strip(".jpg") for f in os.listdir(thumbnails_dir)]

    # Write preprocessed titles to file with video id
    with open(titles_path, "w") as f:
        for video_id in dled:
            if video_id in yt_data.keys():
                title = preprocess_text(yt_data[video_id]["title"])
                f.write(video_id + "\t" + title + "\n")


def generate_image_features(thumbnails_dir, shape=(224, 224, 3), verbose=False):
    """
    Create CNN features for thumbnails using VGG16.

    Args
    ----
    thumbnails_dir: str
        Path to thumbnails.
    shape: tuple
        Shape of features.

    Returns
    -------
    features: dict
        CNN features for model.
    """
    # Load model
    model = VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    if verbose:
        print(model.summary())

    # Initialize features dictionary
    features = dict()

    for f in os.listdir(thumbnails_dir):
        name = thumbnails_dir + f

        image = load_img(name, target_size=(shape[0], shape[1]))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        feature = model.predict(image, verbose=0)

        video_id = f.split(".")[0]
        features[video_id] = feature

        print(video_id, "completed...")

    return features


if __name__ == "__main__":
    print("Loading data...")
    yt_data = load_data(metadata_files)

    print("Downloading thumbnails...")
    get_thumbnails(yt_data, thumbnails_dir)

    print("Creating titles...")
    get_titles(yt_data, thumbnails_dir, titles_path)

    print("Generating image features...")
    image_features = generate_image_features(thumbnails_dir, verbose=True)
    pickle.dump(image_features, open(features_path, "wb"))
