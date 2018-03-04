# -*- coding: utf-8 -*-
"""
Bot implementation
"""
from argparse import ArgumentParser
import logging
from os import chdir
from os.path import dirname, abspath, isfile
chdir(dirname(dirname(abspath(__file__)))) # Set working directory to project root

# Enable logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

LOGGER = logging.getLogger(__name__)

def _check_files():
    files = [
        "./data/index_word.data",
        "./data/weights/convo_weights.hdf5"
    ]
    missing = [_file for _file in files if not isfile(_file)]
    return len(missing) == 0, missing

def main():
    """
    Runs the bot
    """
    from conversationbot.backend import ConversationBot
    from conversationbot.handlers.conversation import conversation
    from telegram.ext import Updater
    from telegram.error import InvalidToken, Unauthorized
    
    parser = ArgumentParser()
    parser.add_argument("key", help="Telegram API key")
    telegram_api_key = vars(parser.parse_args()).get("key")
    if telegram_api_key is None:
        return
    passed, missing = _check_files()
    if not passed:
        for _file in missing:
            LOGGER.error("Cannot find %s", abspath(_file))
        return
    try:
        updater = Updater(telegram_api_key)
    except (InvalidToken, Unauthorized):
        LOGGER.error("Invalid Telegram API token")
        return

    # Register handlers
    def error_handler(bot, update, error):
        pass
    
    updater.dispatcher.add_handler(conversation())
    updater.dispatcher.add_error_handler(error_handler)
    
    # Load conversation bot
    ConversationBot()

    LOGGER.info("Bot ready")
    updater.start_polling(timeout=20)
    updater.idle()

if __name__ == '__main__':
    main()
