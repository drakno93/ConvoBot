# -*- coding: utf-8 -*-
"""
Trains a neural conversation model
"""
from argparse import ArgumentParser
from collections import OrderedDict
from os import chdir, replace
from os.path import dirname, abspath, isfile
import logging
import numpy as np
import pickle
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Embedding, LSTM, Dense, Concatenate
from keras.models import Model
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

chdir(dirname(dirname(abspath(__file__)))) # Set working directory to project root

# Enable logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

LOGGER = logging.getLogger(__name__)

MAX_SENTENCE_LENGTH = 30
MAX_WORDS = 8000
EMBEDDING_DIM = 300
HIDDEN_SIZE = 300

def load_glove(words, glove_file):
    """
    Prepares the weights for the model's embedding layer
    Parameters
    ----------
    words : list
            List of words in the vocabulary
    glove_file : string
            Path to the GloVe embeddings file
    Returns
    -------
    embedding_matrix : np.array 
            Matrix with the word embeddings of words in the vocabulary
    """
    embeddings = dict()
    with open(glove_file, encoding='utf8') as glove_data:
        for line in glove_data:
            _split = line.split()
            embeddings[_split[0]] = np.asarray(_split[1:], dtype='float32')

    embedding_matrix = np.zeros((len(words), EMBEDDING_DIM))
    for i, word in enumerate(words):
        emb = embeddings.get(word)
        if emb is not None:
            embedding_matrix[i] = emb
    del embeddings
    return embedding_matrix

def prepare_data(data):
    """
    Transforms data in numerical sequences
    Parameters
    ----------
    data : file
           Text file with the training set
    Returns
    -------
    tuple : (list, list, collections.OrderedDict) 
            Tuple containing :\n
            dialogs : List of strings with the dialogs;\n
            answers : List of strings with the answers;\n
            word_index : Vocabulary that maps words to indices
    """
    dialogs = []
    answers = []
    for line in data:
        split = line.split("\t")
        dialogs.append(split[0])
        answers.append(split[1])

    # To word sequences
    dialogs = [text_to_word_sequence(sent) for sent in dialogs]
    answers = [(['$_START_$'] + text_to_word_sequence(sent) + ['$_EOS_$']) for sent in answers]

    # Apply UNK to sents
    index_word = pickle.load(open("data/index_word.data", "rb"))
    word_index = OrderedDict([(w, i) for i, w in enumerate(index_word)])
    for i, sent in enumerate(dialogs):
        for j, word in enumerate(sent):
            if word not in word_index:
                dialogs[i][j] = "$_UNK_$"
    for i, sent in enumerate(answers):
        for j, word in enumerate(sent):
            if word not in word_index:
                answers[i][j] = "$_UNK_$"
    return (dialogs, answers, word_index)

def get_batch(dialogs, answers, word_index, batch_size):
    """
    Batch generator for training
    Parameters
    ----------
    dialogs : list
            List of strings with the dialogs
    answers : list
            List of strings with the answers
    word_index : collections.OrderedDict
            Vocabulary that maps words to indices
    batch_size : int
            Number of lines of dialog processed in a batch.
            Note: this does not equal to the Keras training batch size
    Yields
    -------
    tuple : (list, np.array) 
            Tuple containing:\n
            - List with:\n
            d_train: NumPy matrix (np.array) with the encoding of the dialogs in the batch;\n
            a_train: NumPy matrix (np.array) with the encoding of the partial answers in the batch;\n
            - y_train: np.array\n\t
            NumPy matrix with the one-hot encodings of the next word to predict  
    """
    start = 0
    while True:
		# Convert batches to numerical sequence
        d_batch = [[word_index[word] for word in sent]
                   for sent in dialogs[start:start+batch_size]]
        a_batch = [[word_index[word] for word in sent]
                   for sent in answers[start:start+batch_size]]

        d_train = []
        a_train = []
        targets = []
        for i, dlg in enumerate(d_batch):
            for k in range(len(a_batch[i])):
                d_train.append(dlg)
                a_train.append(a_batch[i][:k+1])
                targets.append(a_batch[i][k+1])
                if targets[-1] == word_index['$_EOS_$']:
                    break

		# Pad for variable length sequences
        d_train = pad_sequences(d_train, maxlen=MAX_SENTENCE_LENGTH)
        a_train = pad_sequences(a_train, maxlen=MAX_SENTENCE_LENGTH, padding='post')
		# One hot for the labels
        y_train = np.zeros((len(targets), len(word_index)), dtype=np.bool)
        for i, word in enumerate(targets):
            y_train[i, word] = 1

        start += batch_size
        if start >= len(dialogs):
            start = 0 # Reset generator
        yield ([d_train, a_train], y_train)

def build_model(nwords, embedding_matrix):
    """
    Returns the compiled model
    Parameters
    ----------
    nwords : int
            Number of words in the vocabulary used for the training
    embedding_matrix : np.array
            Matrix with the word embeddings (one for each row) 
    Returns
    -------
    model : keras.models.Model 
            Compiled Keras conversation model
    """
    in_dialog = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
    in_answer = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
    d_encoder = LSTM(HIDDEN_SIZE)
    a_encoder = LSTM(HIDDEN_SIZE)
    embedding_layer = Embedding(output_dim=EMBEDDING_DIM, input_dim=nwords, mask_zero=True,
                                weights=[embedding_matrix] if embedding_matrix is not None else None,
                                input_length=MAX_SENTENCE_LENGTH)
    dialog_encoding = d_encoder(embedding_layer(in_dialog))
    answer_encoding = a_encoder(embedding_layer(in_answer))
    concat_layer = Concatenate(axis=1)([dialog_encoding, answer_encoding])
    out = Dense(nwords, activation="softmax")(concat_layer)
    model = Model(inputs=[in_dialog, in_answer], outputs=[out])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam')
    return model

def _check_files(files):
    missing = [_file for _file in files if _file is not None and not isfile(_file)]
    return len(missing) == 0, missing

def main():
    """
    Launches training
    """
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--glove",
                       help="300-dimensional GloVe embeddings file")
    group.add_argument("--weights",
                       help="Starting weights for continuing training")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    args = vars(parser.parse_args())
    passed, missing = _check_files(["./data/train.tsv", args['glove'], args['weights']])
    if not passed:
        for _file in missing:
            LOGGER.error("Cannot find file %s", abspath(_file))
        return

    with open("./data/train.tsv", "rt") as data:
        print("Processing data...")
        dialogs, answers, word_index = prepare_data(data)

    embedding_matrix = None
    if args['glove'] is not None:
        embedding_matrix = load_glove(list(word_index.keys()), args['glove'])

    print("Dictionary size: {}".format(len(word_index) - 4))
    print("Training samples: {}".format(len(dialogs)))

    # Build model
    model = build_model(len(word_index), embedding_matrix)
    if args['weights'] is not None:
        model.load_weights(args["weights"]) # Load weights to resume training

    # Train, 32 lines at a time
    model_checkpoint = ModelCheckpoint("data/weights/convo_weights_ep{epoch:02d}.hdf5",
                                       monitor='loss',
                                       save_weights_only=True,
                                       period=1,
                                       save_best_only=True
                                      )
    model.fit_generator(get_batch(dialogs, answers, word_index, 32),
                        steps_per_epoch=round(len(dialogs) / 32),
                        epochs=args.get('epochs'),
                        callbacks=[model_checkpoint])

    # Automatically rename last weight file
    replace("data/weights/convo_weights_ep{0:02d}.hdf5".format(args['epochs']),
           "data/weights/convo_weights.hdf5")

if __name__ == '__main__':
    main()
