import numpy as np
import tensorflow as tf


def main(n_input, dictionary, reverse_dictionary, pred, x):

    # load session from saved file
    with tf.Session as sess:
        saver.restore(sess, "/")



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
    print("HP-LSTM: Starting Predictions!")


    # n_input
    # dictionary
    # reverse_dictionary
    # session
    # x
    # pred
