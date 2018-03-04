# -*- coding: utf-8 -*-
"""
Backend for the bot
"""
from collections import OrderedDict
import logging
import numpy as np
import pickle
import tensorflow as tf
from keras.layers import Input, LSTM, Embedding, Concatenate, Dense
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence

# Enable logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

LOGGER = logging.getLogger(__name__)

MAX_SENTENCE_LENGTH = 30
EMBEDDING_DIM = 300
HIDDEN_SIZE = 300

class Singleton(type):
    """
    Define an Instance operation that lets clients access its unique
    instance.
    """
    def __init__(cls, name, bases, attrs, **kwargs):
        super().__init__(name, bases, attrs)
        cls._instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance

class ConversationBot(metaclass=Singleton):
    """
    Bot that can reply to messages with a wrapped conversational model
    """
    def __init__(self):
        self._index_word = pickle.load(open("data/index_word.data", "rb"))
        self._word_index = OrderedDict([(w, i) for i, w in enumerate(self._index_word)])
        self._model = self._load_model()
        self._graph = tf.get_default_graph()

    def reply(self, message):
        """
        Replies to the message with a prediction of the model
        Parameters
        ----------
        message : string
                  Message which the bot has to reply to
        Returns
        -------
        reply : string 
                Reply text
        """
        message = text_to_word_sequence(message)
        message = [self._word_index[word] if word in self._word_index
                   else self._word_index['$_UNK_$'] for word in message]
        part_answer = [self._word_index['$_START_$']]
        answer = []
        x_predict = pad_sequences([message], maxlen=MAX_SENTENCE_LENGTH)
        x_ans = pad_sequences([part_answer], maxlen=MAX_SENTENCE_LENGTH, padding='post')
        with self._graph.as_default():
            while len(part_answer) < MAX_SENTENCE_LENGTH:
                w_ix = np.argmax(self._model.predict([x_predict, x_ans]))
                word = self._index_word[w_ix]
                if word == "$_EOS_$":
                    break
                else:
                    part_answer.append(w_ix)
                    x_ans = pad_sequences([part_answer],
                                          maxlen=MAX_SENTENCE_LENGTH, padding='post')
                    if word != "$_UNK_$":
                        answer.append(word)
        if len(answer) == 0:
            return "?" # avoid empty messages
        else:
            return ' '.join(answer)

    def _load_model(self):
        in_dialog = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
        in_answer = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
        d_encoder = LSTM(HIDDEN_SIZE)
        a_encoder = LSTM(HIDDEN_SIZE)
        embedding_layer = Embedding(output_dim=EMBEDDING_DIM, input_dim=len(self._index_word),
                                    mask_zero=True, input_length=MAX_SENTENCE_LENGTH)
        dialog_encoding = d_encoder(embedding_layer(in_dialog))
        answer_encoding = a_encoder(embedding_layer(in_answer))
        concat_layer = Concatenate(axis=1)([dialog_encoding, answer_encoding])
        out = Dense(len(self._index_word), activation="softmax")(concat_layer)
        model = Model(inputs=[in_dialog, in_answer], outputs=[out])
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam')
        model.load_weights("data/weights/convo_weights.hdf5")
        return model
