#!/usr/bin/python
# -*- coding: utf8 -*-

# import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
# from nltk.corpus import names
import codecs
import pymorphy2

'''
def word_feats(words):
    return dict([(word, True) for word in words])
'''


def word_feats(words):
    return dict.fromkeys(words, True)


positive_vocab = []
negative_vocab = []
morph = pymorphy2.MorphAnalyzer()
with codecs.open('dict/positive.txt', encoding='utf-8') as file_object:
    for line in file_object:
        line = line.rstrip('\n\r')
        positive_vocab.append(morph.parse(line)[0].normal_form)
with codecs.open('dict/negative.txt', encoding='utf-8') as file_object:
    for line in file_object:
        line = line.rstrip('\n\r')
        negative_vocab.append(morph.parse(line)[0].normal_form)

positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
# neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]

# train_set = negative_features + positive_features + neutral_features
train_set = negative_features + positive_features

classifier = NaiveBayesClassifier.train(train_set)

# Predict
neg = 0
pos = 0
sentence = u"Маша хорошая девочка"
sentence = sentence.lower()
words = sentence.split(' ')

for word in words:
    classResult = classifier.classify(word_feats(morph.parse(word)[0].normal_form))
    if classResult == 'neg':
        neg += 1
    if classResult == 'pos':
        pos += 1

print('Positive: ' + str(float(pos) / len(words)))
print('Negative: ' + str(float(neg) / len(words)))
print('Result: ' + str((float(pos)-float(neg)) / len(words)))
