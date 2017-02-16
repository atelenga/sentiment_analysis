#!/usr/bin/python
# -*- coding: utf8 -*-

from senticnet.senticnet import Senticnet
from nltk.classify import NaiveBayesClassifier
import pymorphy2
import codecs

def word_feats(words):
    return dict([(word, True) for word in words])

sn = Senticnet('ru')
morph = pymorphy2.MorphAnalyzer()

# заполняем масссив тэгами SenticNet
positive_vocab = ['#интерес', '#радость', '#сюрприз', '#восхищение']
negative_vocab = ['#попугать', '#гнев', '#печаль', '#отвращение']

# добавляем слова из WordNet
with codecs.open('dict/positive.txt', encoding='utf-8') as file_object:
    for line in file_object:
        line = line.rstrip('\n\r')
        positive_vocab.append(morph.parse(line)[0].normal_form)
with codecs.open('dict/negative.txt', encoding='utf-8') as file_object:
    for line in file_object:
        line = line.rstrip('\n\r')
        negative_vocab.append(morph.parse(line)[0].normal_form)

# наполнаяем множества позитивных и негативных слов и обучаем классификатор
positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
train_set = negative_features + positive_features
classifier = NaiveBayesClassifier.train(train_set)

# Основная работа
sentence = 'Тест'
sentence = sentence.lower()
words = sentence.split(' ')

neg = 0
pos = 0
mood = 0

for word in words:
    word = morph.parse(word.decode('utf8'))[0].normal_form
    try: # пробуем получить теги из SenticNet
        moodtags = sn.moodtags(word.encode('utf8'))
        for item in moodtags:
            mood += 1
            if item in negative_vocab:
                neg += 1
            elif item in positive_vocab:
                pos += 1
    except: # проверяем, не входит ли слово в словарь
        if word in negative_vocab:
            neg += 1
            mood += 1
        elif word in positive_vocab:
            pos += 1
            mood += 1

# если ничего не помогает, пытаемся классифицировать слово через наивный Баесовский классификатор
if mood == 0:
    for word in words:
        classResult = classifier.classify(word_feats(word))
        if classResult == 'neg':
            mood += 1
            neg += 1
        if classResult == 'pos':
            pos += 1
            mood += 1

print('Positive: ' + str(float(pos) / mood))
print('Negative: ' + str(float(neg) / mood))
print('Result: ' + str((float(pos)-float(neg)) / mood))
