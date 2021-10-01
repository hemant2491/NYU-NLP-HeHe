import json
import collections
import argparse
import random

from util import *

random.seed(42)

def extract_unigram_features(ex):
    """Return unigrams in the hypothesis and the premise.
    Parameters:
        ex : dict
            Keys are gold_label (int, optional), sentence1 (list), and sentence2 (list)
    Returns:
        A dictionary of BoW featurs of x.
    Example:
        "I love it", "I hate it" --> {"I":2, "it":2, "hate":1, "love":1}
    """
    # BEGIN_YOUR_CODE

    # bag of words map
    bow = {}
    sentence_count = 0
    word_count = 0

    sentences = [ex["sentence1"], ex["sentence2"]]
    for sentence in sentences:
        sentence_count += 1
        try:
            for word in sentence:
                word_count += 1
                if word in bow:
                    bow[word] += 1
                else:
                    bow[word] = 1
        except Exception as e:
            print(e)

    # print("Sentence count: ")
    # print(sentence_count)
    # print("word count: ")
    # print(word_count)
     
    # print(bow)
    return bow
    #pass
    # END_YOUR_CODE

def find_ngrams(input_list, n):
    nkeys = []
    tuples = zip(*[input_list[i:] for i in range(n)])
    for tuple in tuples:
        nkeys.append(''.join(tuple))
        # if tuple not in nkeys:
        #     nkeys.append(''.join(tuple))
    # print("keys: ")
    # print(nkeys)
    return nkeys


def extract_custom_features(ex):
    """Design your own features.
    """
    # BEGIN_YOUR_CODE
    # bag of words map
    bow = {}
    sentence_count = 0
    word_count = 0

    sentences = [ex["sentence1"], ex["sentence2"]]
    for sentence in sentences:
        sentence_count += 1
        try:
            for word in sentence:
                word_count += 1
                if word in bow:
                    bow[word] += 1
                else:
                    bow[word] = 1
        except Exception as e:
            print(e)
    n=2

    # sentences = [ex["sentence1"], ex["sentence2"]]
    ngrams1 = find_ngrams(ex["sentence1"], n)
    ngrams2 = find_ngrams(ex["sentence2"], n)
    ngrams = ngrams1 + ngrams2

    for ngram in ngrams:
        if ngram in bow:
            bow[ngram] += 1
        else:
            bow[ngram] = 1


    return bow
    # pass
    # END_YOUR_CODE

def learn_predictor(train_data, valid_data, feature_extractor, learning_rate, num_epochs):
    """Running SGD on training examples using the logistic loss.
    You may want to evaluate the error on training and dev example after each epoch.
    Take a look at the functions predict and evaluate_predictor in util.py,
    which will be useful for your implementation.
    Parameters:
        train_data : [{gold_label: {0,1}, sentence1: [str], sentence2: [str]}]
        valid_data : same as train_data
        feature_extractor : function
            data (dict) --> feature vector (dict)
        learning_rate : float
        num_epochs : int
    Returns:
        weights : dict
            feature name (str) : weight (float)
    """
    # BEGIN_YOUR_CODE
    try:
        # dictionary with weights of all words
        weights = {}
        n=2

        # initialize weights of total_data = train_data + valid_data
        # total_data = {'train_data': train_data, 'valid_data': valid_data}
        # for key, data_set in total_data.items():
        # for key, data_set in {'train_data': train_data, 'valid_data': valid_data}.items():
        for data_set in [train_data, valid_data]:
            for dict_i in data_set:
                # sentences = [dict_i["sentence1"], dict_i["sentence2"]]
                # for sentence in sentences:
                for sentence in [dict_i["sentence1"], dict_i["sentence2"]]:
                    ngrams = find_ngrams(sentence, n)
                    for word in sentence:
                        if word not in weights:
                            weights[word] = 0
                    for ngram in ngrams:
                        if ngram not in weights:
                            weights[ngram] = 0

        # print("weights length")
        # print(len(weights))
        # epoch iterations
        for i in range(num_epochs):
            for dict_l in train_data:
                feature = feature_extractor(dict_l)
                # print("feature")
                # print(feature)
                pred = predict(weights, feature)
                for k, v in feature.items():
                    if k in weights:
                        weights[k] = weights[k] - learning_rate * (pred - dict_l["gold_label"]) * v
    except Exception as e:
        print(e)
        raise

    return weights
    # pass
    # END_YOUR_CODE
