from typing import List, Any
import numpy as np
import cv2
from tensorflow import keras
from keras.models import Sequential
from keras import optimizers
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, LSTM, BatchNormalization
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from keras import backend as K
from keras.constraints import maxnorm
import tensorflow as tf
import idx2numpy
import telebot
import random
import os
from telebot import types
emnist_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]





def letters_extract(image_file: str, out_size=28) -> List[Any]:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:

                y_pos = size_max//2 - h//2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                x_pos = size_max//2 - w//2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    letters.sort(key=lambda x: x[0], reverse=False)

    return letters


model = keras.models.load_model('versia03.h5')

def emnist_predict_img(model, img):
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr/255.0
    img_arr[0] = np.rot90(img_arr[0], 3)
    img_arr[0] = np.fliplr(img_arr[0])
    img_arr = img_arr.reshape((1, 28, 28, 1))

    predict = model.predict([img_arr])
    result = np.argmax(predict, axis=1)
    return chr(emnist_labels[result[0]])


def img_to_str(model: Any, image_file: str):
    letters = letters_extract(image_file)
    s_out = ""
    for i in range(len(letters)):
        dn = letters[i+1][0] - letters[i][0] - letters[i][1] if i < len(letters) - 1 else 0
        s_out += emnist_predict_img(model, letters[i][2])
        if (dn > letters[i][1]/4):
            s_out += ' '
    return s_out


model = keras.models.load_model('versia03.h5')

#tg part
token = "5212739885:AAEJy-9HV63ftgVRxhWoDK_SztXYpEThFcY"


bot = telebot.TeleBot(token=token)

@bot.message_handler(commands=["start"])
def start(message, res=False):
    text = message.text
    user = message.chat.id
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1 = types.KeyboardButton("/info")
    item2 = types.KeyboardButton("/format")
    item3 = types.KeyboardButton("/contacts")
    markup.add(item1)
    markup.add(item2)
    markup.add(item3)

    bot.send_message(message.chat.id, "Добро пожаловать, для начала работы отправьте файл, который нужно отсканировать.\nДля получения справок нажмите /info", reply_markup=markup)

@bot.message_handler(commands=["contacts"])
def contacts(message, res=False):
    text = message.text
    user = message.chat.id
    bot.send_message(user, "Telegram: @Sokolov_koma\nVK: sokolovkoma\nGmail: sokolov.koma@gmail.com\nБуду рад любой обратной связи")
@bot.message_handler(commands=["format"])
def format(message):
    text = message.text
    user = message.chat.id
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1 = types.KeyboardButton("txt")
    item2 = types.KeyboardButton("docx")
    item3 = types.KeyboardButton("xlsx")
    markup.add(item1)
    markup.add(item2)
    markup.add(item3)
    bot.send_message(message.chat.id, "Выберите нужный вам формат среди кнопок", reply_markup=markup)

@bot.message_handler(commands=["info"])
def info(message, res=False):
    text = message.text
    user = message.chat.id
    bot.send_message(user, "Данный бот создан для перевода текста на картинках разных форматов в текстовый формат \nПоддерживаемые форсаты для импорта: png, jpg, jpeg\nЧтобы получить результат не в качестве сообщения, а файла - воспользуйтесь командой /format\nКонтакты для обратной связи можно получить по команде /contacts")

@bot.message_handler(content_types=["text"])
def tekst(message):
    if message.text == 'txt':
        bot.send_message(message.chat.id, 'Теперь отправь мне файл, и я конвертирую его в нужном тебе формате')
    if message.text == 'docx':
        bot.send_message(message.chat.id, 'Теперь отправь мне файл, и я конвертирую его в нужном тебе формате')
    if message.text == 'xlsx':
        bot.send_message(message.chat.id, 'Теперь отправь мне файл, и я конвертирую его в нужном тебе формате')



@bot.message_handler(content_types=['document'])
def echo(message):
    # message - входящее сообщение
    # message.text - это его текст

    text = message.text
    user = message.chat.id
    bot.send_message(user, 'Выполняется...')
    #
    chat_id = message.chat.id

    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    src = '/Users/user/Dropbox/Мой Mac (MacBook Air — User)/Desktop/zzzxxxccc.jpg'
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)
    bot.send_message(user, img_to_str(model, '/Users/user/Dropbox/Мой Mac (MacBook Air — User)/Desktop/zzzxxxccc.jpg'))
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '/Users/user/Dropbox/Мой Mac (MacBook Air — User)/Desktop/zzzxxxccc.jpg')
    os.remove(path)




bot.polling(none_stop=True)
