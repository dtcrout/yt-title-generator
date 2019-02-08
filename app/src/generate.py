"""Generate thumbnail titles."""

from main import *
from keras.models import load_model
from numpy import argmax
from numpy import random


FEATURES = '../resources/features.pkl'
TITLES = '../resources/titles.txt'
MODEL = './model.h5'

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_title(model, tokenizer, photo, max_length):
    in_text = 'startseq'

    vocab = len(tokenizer.word_index) + 1

    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]

        sequence = pad_sequences([sequence], maxlen=max_length)

        yhat = model.predict([photo, sequence], verbose=0)

        yhat = random.choice(list(range(vocab)), 1, p=yhat[0])

        # yhat = argmax(yhat)

        word = word_for_id(yhat, tokenizer)

        if word is None:
            break

        in_text += ' ' + word

        if word == 'endseq':
            break

    return in_text


if __name__ == "__main__":
    features = pickle.load(open(FEATURES, 'rb'))

    tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

    model = load_model(MODEL)

    with open('generated.txt', 'w') as f:
        for k, v in features.items():
            title = generate_title(model, tokenizer, v, max_length=21)

            line = k + '\t' + title

            print(line)

            f.write(line + '\n')
