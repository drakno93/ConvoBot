# -*- coding: utf-8 -*-
"""Conversation mode"""
from telegram import ChatAction, ParseMode
from telegram.ext import CommandHandler, ConversationHandler, MessageHandler, Filters
from ..backend import ConversationBot

CM_WAIT_QUESTION = range(1)

def start(bot, update):
    """Entry point. Handler for /start command"""
    bot.send_message(chat_id=update.message.chat_id,
                     text="_Conversation started_",
                     parse_mode=ParseMode.MARKDOWN)
    return CM_WAIT_QUESTION

def cm_answer(bot, update, chat_data):
    """Answers to the user"""
    bot.send_chat_action(chat_id=update.message.chat_id,
                         action=ChatAction.TYPING)
    conv_bot = ConversationBot()
    if 'dialog' not in chat_data:
        chat_data['dialog'] = [update.message.text]
    else:
        chat_data['dialog'].append(update.message.text)
    reply = conv_bot.reply(" ".join(chat_data['dialog']))
    bot.send_message(chat_id=update.message.chat_id,
                     text=reply.capitalize())
    # Update dialog
    chat_data['dialog'].append(reply)
    chat_data['dialog'] = chat_data['dialog'][-2:]
    return CM_WAIT_QUESTION

def conversation():
    """Returns Conversation mode Conversation Handler"""
    handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            CM_WAIT_QUESTION:[MessageHandler(Filters.text, cm_answer, pass_chat_data=True)]
            },
        fallbacks=[])
    return handler
