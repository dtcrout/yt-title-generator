"""Training a caption model."""

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import pad_sequences
from keras.utils import to_categorical
from model import model
import pickle

TITLES = '../resources/titles.txt'

def load_titles(path):
    titles = {}
    with open(path, 'r') as f:
        for line in f:
            video_id = line.split('\t')[0]
            title = line.strip('\n').split('\t')[1]

            # Add start and end tags to title
            pp_title = 'startseq ' + title + ' endseq'

            titles[video_id] = pp_title

    return titles


def load_image_features(path, dataset):
    features = pickle.load(open(path, 'rb'))
    return {k: features[k] for k in dataset}


def create_tokenizer(titles):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(titles)
    return tokenizer


def create_sequences(tokenizer, title, image, max_length):
    X_img = []
    X_seq = []
    Y = []

    seq = tokenizer.texts_to_sequences([title])[0]

    for i in range(1, len(seq)):
        in_seq = seq[:i]
        out_seq = seq[i]

        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

        X_img.append(image)
        X_seq.append(in_seq)
        Y.append(out_seq)

    return [X_img, X_seq, Y]


if __name__ == "__main__":
    # Load titles
    titles = load_titles(TITLES)

    # Take the first 100 titles
    dataset = titles[:100]


