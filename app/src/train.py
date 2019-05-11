"""Train a caption model.

This script trains the captioning model.

Code is adapted from:
https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/
"""

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from model import caption_model
from numpy import array
import pickle
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords

TITLES = "../resources/titles.txt"
FEATURES = "../resources/features.pkl"
MODEL = "../resources/model.h5"
TOKENIZER = "../resources/tokenizer.pkl"

stopwords = stopwords.words("english")


def clean_titles(title):
    title = title.split()
    title = [word.lower() for word in title]
    title = [word for word in title if len(word) > 1]
    title = [word for word in title if word.isalpha()]
    title = [word for word in title if word not in stopwords]
    title = " ".join(title)
    return title


def load_titles(path):
    titles = dict()
    with open(path, "r") as f:
        for line in f:
            tokens = line.strip("\n").split("\t")
            video_id, title = tokens[0], tokens[1]
            title = "startseq " + title + " endseq"
            titles[video_id] = title

    return titles


def load_image_features(path, dataset):
    features = pickle.load(open(path, "rb"))
    return {k: features[k] for k in dataset}


def to_lines(titles):
    all_titles = []
    for key in titles.keys():
        for t in titles[key].split():
            all_titles.append(t)
    return all_titles


def create_tokenizer(titles):
    all_titles = []
    for k in titles.keys():
        all_titles.append(titles[k])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_titles)
    return tokenizer


def max_length(titles):
    lengths = []
    for key in titles.keys():
        for t in titles[key].split():
            lengths.append(len(t))
    return max(lengths)


def make_title_set(titles, dataset):
    title_set = dict()
    for video_id in dataset:
        title_set[video_id] = titles[video_id]
    return title_set


def create_sequences(tokenizer, max_length, titles, images):
    X_img = []
    X_seq = []
    Y = []

    for key, title in titles.items():
        # title = clean_titles(title)
        seq = tokenizer.texts_to_sequences([title])[0]

        for i in range(1, len(seq)):
            in_seq = seq[:i]
            out_seq = seq[i]

            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

            X_img.append(images[key][0])
            X_seq.append(in_seq)
            Y.append(out_seq)

    return array(X_img), array(X_seq), array(Y)


if __name__ == "__main__":
    # Load titles
    titles = load_titles(TITLES)
    dataset = list(titles.keys())

    train_dataset, test_dataset = train_test_split(
        dataset, train_size=0.7, test_size=0.3
    )

    # Make image feature sets
    train_features = load_image_features(FEATURES, train_dataset)  # dict()
    test_features = load_image_features(FEATURES, test_dataset)  # dict()

    # Make title sets
    train_titles = make_title_set(titles, train_dataset)  # dict()
    test_titles = make_title_set(titles, test_dataset)  # dict()

    # Prepare tokenizer
    tokenizer = create_tokenizer(train_titles)
    vocab_size = len(tokenizer.word_index) + 1
    print("Vocab size: ", vocab_size)
    max_length = max_length(train_titles)
    print("Title length:", max_length)

    pickle.dump(tokenizer, open(TOKENIZER, "wb"))

    X1_train, X2_train, Ytrain = create_sequences(
        tokenizer, max_length, train_titles, train_features
    )
    X1_test, X2_test, Ytest = create_sequences(
        tokenizer, max_length, test_titles, test_features
    )

    model = caption_model(vocab_size, max_length)

    filepath = MODEL
    checkpoint = ModelCheckpoint(
        filepath, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
    )

    model.fit(
        [X1_train, X2_train],
        Ytrain,
        epochs=50,
        verbose=2,
        callbacks=[checkpoint],
        validation_data=([X1_test, X2_test], Ytest),
    )
