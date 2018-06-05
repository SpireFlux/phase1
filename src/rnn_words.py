"""
Towards Data Science
https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537


Original Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments

Modified by: Michael Mathes


"""
from __future__ import print_function

import collections
import datetime
import os
import random
import string
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


def get_model_save_path(name):
    """
    # Create the save path with the following pattern
    # /hp-lstm/models/ISODATE/model.ckpt

    Args:
        name (str): model name

    Returns:
        save_path (string): specific formatted model save path
    """
    t_now = datetime.datetime.now().isoformat("_", timespec="seconds").replace(":", "-")

    model_name = name + "model.ckpt"

    model_path = os.path.join("models", t_now, model_name)
    base_dir = os.path.dirname(os.getcwd())
    save_path = os.path.join(base_dir, model_path)
    return save_path


def build_dataset(words):
    """
    Creates the word <-> integer mapping
    Args:
        words (String):

    Returns:
        dictionary (dict):
        reverse_dictionary (dict):

    """
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary


def clean_data(content, include_newlines, include_punct):
    """
    Read the input file content and convert into a numpy vector.

    Args:
        content (str): str content
        include_newlines (boolean): Include newlines?
        include_punct (Boolean): Include punctuation?

    Returns:
        training_data (ndarray): a large ndarray | vector with the file content
    """
    content_list = []

    # Include Newlines?
    if include_newlines:

        # split int lines
        for line in content.split("\n"):

            for word in line.split(" "):

                # handle double space case
                if word == '':
                    continue

                # remove all punctuation
                translator = str.maketrans('', '', string.punctuation)
                new_word = word.translate(translator)

                # remove quotes
                last_word = new_word.replace("\"", "")
                last_word = last_word.replace("”", "")
                last_word = last_word.replace("“", "")

                content_list.append(last_word)

            content_list.append("\n")

    # remove newlines
    else:
        # split int lines

        for line in content.split("\n"):
            if line == '':
                continue

            for word in line.split(" "):
                # handle double space case
                if word == '':
                    continue

                # remove all punctuation
                translator = str.maketrans('', '', string.punctuation)
                new_word = word.translate(translator)

                # remove quotes
                last_word = new_word.replace("\"", "")
                last_word = last_word.replace("”", "")
                last_word = last_word.replace("“", "")

                content_list.append(last_word)

    training_data = np.array(content_list)
    training_data = np.reshape(training_data, [-1, ])
    return training_data


def setup_data(training_file_path, include_newlines, include_punct):
    """
        returns dictionary and reverse dictionary and training data

    Args:
        training_file_path (str):
        include_newlines (boolean): Include newlines?
        include_punct (Boolean): Include punctuation?

    Returns:
        training_data ():
        dictionary ():
        reverse_dictionary():
    """
    # Text file containing words for training
    with open(training_file_path) as f:
        content = f.read()

    # Clean the training data:
    # The two booleans adjust 'include newlines' or include punctuation
    print("HP-LSTM: Loaded training data...")
    training_data = clean_data(content, include_newlines, include_punct)

    dictionary, reverse_dictionary = build_dataset(training_data)

    return dictionary, reverse_dictionary, training_data


def RNN(x, weights, biases, n_input, n_hidden):
    """
    Args:
        x        (tf.placeholder float):
        weights  ():
        biases   ():
        n_input  (int): width of the buffer
        n_hidden (int): units in the rnn cell

    Returns:
        prediction (tensor): Return the y_hat value

    """
    with tf.name_scope(name="Prediction"):
        # reshape to [1, n_input]
        x = tf.reshape(x, [-1, n_input])

        # Generate a n_input-element sequence of inputs
        x = tf.split(x, n_input, 1)

        # 2-layer LSTM, each layer has n_hidden units.
        rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)])

        # generate prediction
        outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

        # there are n_input outputs but we only want the last one
        y_hat = tf.matmul(outputs[-1], weights) + biases

        # print("y_hat")
        # print(type(y_hat))
        # print(y_hat)
        return y_hat


def train_model(dictionary, reverse_dictionary, training_data, save_path, tensorboard_name):
    """

    Args:
        dictionary():
        reverse_dictionary():
        training_data():
        save_path():
        tensorboard_name ():

    Returns:
        None
    """
    start_time = time.time()

    # Parameters
    learning_rate = 0.003
    training_iters = 10_000
    display_step = 1000

    # Word buffer
    n_input = 9

    # number of units in RNN cell
    n_hidden = 512

    # Length of the vocab size
    vocab_size = len(dictionary)

    print("vocab size: {}".format(vocab_size))

    # This might work?
    tf.summary.scalar("hidden units", n_hidden)
    tf.summary.scalar("vocab_size", vocab_size)
    tf.summary.scalar("order", n_input)

    # tf Graph input
    x = tf.placeholder("float", [None, n_input, 1], name="x")
    y = tf.placeholder("float", [None, vocab_size], name="y")

    # RNN output node weights and biases
    # random normal creates and populates random values for these variables in various shapes
    weights = tf.Variable(tf.random_normal([n_hidden, vocab_size]), name="w")
    biases = tf.Variable(tf.random_normal([vocab_size]), name="b")

    tf.summary.histogram("weights", weights)
    tf.summary.histogram("biases", biases)

    pred = RNN(x, weights, biases, n_input, n_hidden)

    # Loss and optimizer
    with tf.name_scope("xent"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        tf.summary.scalar("xent", cost)

    with tf.name_scope("train"):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    # Model evaluation
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    summ = tf.summary.merge_all()

    # Target log path
    logs_path = os.path.join('/tmp/tensorflow/rnn_words', tensorboard_name)
    writer = tf.summary.FileWriter(logs_path)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # init the saver model and location
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as session:
        session.run(init)
        step = 0
        offset = random.randint(0, n_input+1)
        end_offset = n_input + 1
        acc_total = 0
        loss_total = 0

        writer.add_graph(session.graph)

        print("HP-LSTM: Starting Training")
        print("Training Data Length: {}".format(len(training_data)))

        display = True

        while step < training_iters:

            if (step+1) % display_step == 0:
                display = True

            if offset > (len(training_data)-end_offset):
                offset = random.randint(0, n_input+1)

            # setup X
            symbols_in_keys = [[dictionary[str(training_data[i])]] for i in range(offset, offset+n_input)]

            # if display:
            #     print("X: {}".format(symbols_in_keys))

            symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
            symbols_in = [training_data[i] for i in range(offset, offset + n_input)]

            if display:
                print(symbols_in)

            # Setup Y
            symbols_out_onehot = np.zeros([vocab_size], dtype=float)
            symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
            symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

            # if display:
            #     print(symbols_out_onehot)

            ###################################################################
            # RUN THE MODEL
            ###################################################################
            # s, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], summ \
            #                                         feed_dict={x: symbols_in_keys, y: symbols_out_onehot})

            # print(pred)
            # print(type(pred))

            _, acc, loss, onehot_pred, s = session.run([optimizer, accuracy, cost, pred, summ],  feed_dict={x: symbols_in_keys, y: symbols_out_onehot})




            # print(s)
            # print(type(s))
            writer.add_summary(s, step)

            # update loss
            loss_total += loss
            acc_total += acc

            tf.summary.scalar("accuracy", accuracy)
            tf.summary.scalar("cost", cost)



            # writer.add_summary(summary, step)

            ###################################################################
            # Display some stats every 1000 iterations
            ###################################################################
            # if (step+1) % display_step == 0:

            if display:
                print("Iter= " + str(step+1) + ", Average Loss= " + \
                      "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                      "{:.2f}%".format(100*acc_total/display_step))
                acc_total = 0
                loss_total = 0
                symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
                symbols_out = training_data[offset + n_input]
                symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
                print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))

                display = False

            # update step by 1
            step += 1
            offset += (n_input+1)

            if display:
                print("step: {}, Offset: {}".format(step, offset))
                print()


        #######################################################################
        # Save the model for later
        #######################################################################
        print("Saving Model")
        saver.save(session, save_path)
        print(len(dictionary))

    ########################################################################
    # Session Ended
    ########################################################################
    print("HP-LSTM: Optimization Finished!")
    print("Elapsed time: ", time.time() - start_time)
    print(learning_rate, n_input, n_hidden, training_iters)



    # make predictions!
    while True:
        prompt = "%s words: " % n_input
        sentence = input(prompt)
        sentence = sentence.strip()
        words = sentence.split(' ')
        if len(words) != n_input:
            continue
        try:
            symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
            for i in range(32):
                keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                onehot_pred = session.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(sentence)
        except:
            print("Word not in dictionary")











if __name__ == '__main__':
    print("HP-LSTM: Starting...")

    # Setup Training file parameters
    training_file = sys.argv[1]
    tensorboard_name = os.path.basename(training_file)

    include_newlines = True
    include_punct = False

    # init the save path
    filename = os.path.basename(training_file)
    save_path = get_model_save_path(filename)

    # setup all of the data
    print("HP-LSTM: Training File: {}".format(training_file))
    dictionary, r_dictionary, training_data = setup_data(training_file,
                                                         include_newlines,
                                                         include_punct)

    # # setup tensorflow and run the model
    train_model(dictionary, r_dictionary, training_data, save_path, tensorboard_name)
