import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow  # tensorflow==1.5 =, not 2.0
import random

# ssl issue fix
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('punkt')      # this command needs to run first and then you can comment that out

# Loading Json Data
import json
import pickle

with open('intents.json') as file:
    data = json.load(file)

# Data Extraction from Json
try:
    with open('tmp', 'rb') as f:
        words, labels, training, output = pickle.load(f)
except:

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    # print(f'words: {words} \nlabels: {labels} \ndocs_x: {docs_x} \ndocs_y: {docs_y}')

    # Words stemming
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    # distinct sorted words
    words = sorted(list(set(words)))

    # sorted labels
    labels = sorted(labels)

    # Bag of words
    training = []
    output = []

    out_empty = [0 for i in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    # training data and expected output label
    training = numpy.array(training)
    output = numpy.array(output)

    with open('tmp', 'wb') as f:
        pickle.dump((words, labels, training, output), f)

# Model fit: using feedforward neural network with two hidden layers
# reset previous settings just in case
tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])  # input layer
net = tflearn.fully_connected(net, 8)  # 8 neurons to hidden layer
net = tflearn.fully_connected(net, 8)  # second hidden layer
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")  # output layer
net = tflearn.regression(net)

model = tflearn.DNN(net)


# Model Training and to be saved or reloaded
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)  #n_epoch is adjustable for accuracy
    model.save("model.tflearn")

# Predictions based on model
def bag_of_words(s, words):
    bag = [0 for i in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]  # predicted probabilities
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        print(f'results: {results} \nresults_index: {results_index} \ntag: {tag} \nprob: {results[results_index] * 100}')

        if results[results_index] >= 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))  # if response are multiple, just randomly select one
        else:
            print("Sorry... I don't quite understand since I am still a baby. Could you try another expression")


chat()
