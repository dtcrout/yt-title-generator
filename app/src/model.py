"""Caption generation model."""

from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import LSTM
from keras.layers.merge import add
from keras.models import Model


def caption_model(vocab_size, max_length):
    """Captioning model architecture."""
    # Feature extractor
    inputs1 = Input(shape=(4096,))
    # fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation="relu")(inputs1)

    # Sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation="relu")(decoder1)
    outputs = Dense(vocab_size, activation="softmax")(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    print(model.summary())

    return model
