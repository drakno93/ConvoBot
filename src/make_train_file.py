# -*- coding: utf-8 -*-
"""Makes the dataset for the conversational model and the corresponding index to word file"""
from os import chdir
from os.path import dirname, abspath, isfile, join
from argparse import ArgumentParser
import logging
import re
import pickle
from keras.preprocessing.text import text_to_word_sequence

chdir(dirname(dirname(abspath(__file__)))) # Set working directory to project root

# Enable logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

LOGGER = logging.getLogger(__name__)

MAX_CONVERSATIONS = 35000
MAX_WORDS = 8000

def make_index_word(sequences):
    """
    Makes a index-word from the input sequences
    """
    index_word = []
    # Add special tokens
    index_word.append('$_EMPTY_$')
    index_word.append('$_START_$')
    index_word.append('$_EOS_$')
    index_word.append('$_UNK_$')

    # Make word counts
    word_counts = {}
    for sent in sequences:
        for word in sent:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1

    # Limit to MAX_WORDS words
    for word, _ in sorted(word_counts.items(), key=lambda i: i[1], reverse=True):
        index_word.append(word)
        if len(index_word) == MAX_WORDS + 4:
            break
    return index_word

def _check_files(files):
    missing = [_file for _file in files if not isfile(_file)]
    return len(missing) == 0, missing

def main():
    """
    Launches script
    """
    parser = ArgumentParser()
    parser.add_argument("--corpus", required=True,
                        help="Cornell Movie-Dialogs corpus folder")
    parser.add_argument("--glove", required=True,
                        help="300-dimensional GloVe embeddings file")
    args = vars(parser.parse_args())
    corpus_dir = args.get("corpus")
    glove_file = args.get("glove")
    passed, missing = _check_files([join(corpus_dir, "movie_lines.txt"),
                                    join(corpus_dir, "movie_conversations.txt"),
                                    glove_file])
    if not passed:
        for _file in missing:
            LOGGER.error("Cannot find %s", abspath(_file))
        return

    with open(join(corpus_dir, "movie_lines.txt"), encoding="cp1252") as m_lines_file:
        lines_dict = {}
        for line in m_lines_file:
            _split = line.split(" +++$+++ ")
            lid = int(re.match(r'L(\d+)', _split[0]).group(1))
            lines_dict[lid] = _split[-1].rstrip('\n')

    with open(join(corpus_dir, "movie_conversations.txt")) as m_conv_file:
        conversations = []
        for line in m_conv_file:
            _list = line.split(" +++$+++ ")[-1]
            match = re.findall(r'\d+', _list)
            if match is not None:
                conversations.append([lines_dict[int(l)] for l in match])

    # Make vocabulary
    print("Building index to word file...")
    sequences = [text_to_word_sequence(txt) for txt in lines_dict.values()]
    del lines_dict
    index_word = make_index_word(sequences)
    pickle.dump(index_word, open("data/index_word.data", "wb"))

    conversations = conversations[:MAX_CONVERSATIONS]
    # Create training set
    print("Creating training set...")
    with open("data/train.tsv", "wt") as train_file:
        for conv in conversations:
            prev = []
            for j in range(len(conv) - 1):
                prev.append(conv[j])
                dialog = ' '.join(prev)
                prev = prev[-2:]
                answer = conv[j+1]
                train_file.write("{}\t{}\n".format(dialog, answer))

    print("Done.")

if __name__ == '__main__':
    main()
            