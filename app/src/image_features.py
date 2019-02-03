"""Generate CNN features for the model."""

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import os
import pickle

THUMBNAILS_DIR = '../resources/thumbnails/'
RESOURCES = '../resources/'

def generate_features(path, shape=(224, 224, 3), verbose=False):
    """
    Create CNN features for thumbnails using VGG16.

    Input
    -----
    path: str
        Path to thumbnails.
    shape: tuple
        Shape of features.

    Returns
    -------
    features: dict
        CNN features for model.
    """
    # Load model
    in_layer = Input(shape=shape)
    model = VGG16(include_top=False, input_tensor=in_layer)

    if verbose:
        print(model.summary())

    # Initialize features dictionary
    features = dict()

    for f in os.listdir(path):
        name = path + f

        image = load_img(name, target_size=(shape[0], shape[1]))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0],
                               image.shape[1],
                               image.shape[2]))
        image = preprocess_input(image)

        feature = model.predict(image, verbose=0)

        video_id = f.split('.')[0]
        features[video_id] = feature

        print(video_id, 'completed...')


if __name__ == "__main__":
    features = generate_features(THUMBNAILS_DIR, verbose=True)

    pickle.dump(features, open(RESOURCES, 'wb'))
