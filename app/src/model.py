"""Caption generation model."""

from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Model
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layenrs.merge import concatenate
from keras.layers.pooling import GlobalMaxPooling2D


def model(vocab_size, max_length, verbose=False):
    # Feature extractor
    inputs1 = Input(shape=(7, 7, 512))
    fe1 = GlobalMaxPooling2D()(inputs1)
    fe2 = Dense(128, activation='relu')(fe1)
    fe3 = RepeatVector(max_length)(fe2)

    # Embedding
    inputs2 = Input(shape=(max_length,))
    emb2 = Embedding(vocab_size, 50, mask_zero=True)(inputs2)
    emb3 = LSTM(256, return_sequences=True)(emb2)
    emb4 = TimeDistributed(Dense(128, activation='relu'))(emb3)

    # Merge inputs
    merged = concatenate([fe3, emb4])

    # Language model
    lm2 = LSTM(500)(merged)
    lm3 = Dense(500, activation='relu')(lm2)
    outputs = Dense(vocab_size, activation='softmax')(lm3)

    # Model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    if verbose:
        print(model.summary())

    return model
