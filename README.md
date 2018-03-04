# ConvoBot

**ConvoBot** is a telegram bot that uses a rather simple *neural conversational model* to chat with the users.

## Installation

To run the scripts you will need [Tensorflow], [Keras] and [python-telegram-bot] in your environment. Using the GPU version of Tensorflow is highly recommended, since training time will be drastically reduced to a few hours for the default number of epochs. Moreover, you will need 300-dimensional [GloVe embeddings] and the [Cornell Movie-Dialogs Corpus]. The code is tested with Python 3.6.

[Tensorflow]: https://www.tensorflow.org/install/
[Keras]: https://keras.io/#installation
[python-telegram-bot]: https://python-telegram-bot.org/
[GloVe embeddings]: https://nlp.stanford.edu/projects/glove/
[Cornell Movie-Dialogs Corpus]: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

## Usage

- Make the dataset by running the command:
    ```bash
    python make_train_file.py --corpus <corpus folder> --glove <embeddings file>
    ```
    It is also possible to specify the number of training epochs with the option `--epochs` or previously existing weights (instead of the GloVe embeddings) with the option `--weights`.
- Train the bot by running the command:
    ```bash
    python train_bot.py --glove <embeddings file>
    ```
- To run the bot you need the Telegram API key, follow the instructions to [create a new Telegram bot] and start the bot:
    ```bash
    python start_bot.py <key>
    ```
[create a new Telegram bot]: https://core.telegram.org/bots#3-how-do-i-create-a-bot