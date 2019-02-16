from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np
import timeit
from collections import OrderedDict
from pprint import pformat
import tensorflow as tf


def run(algorithm, x_train, y_train, x_test, y_test):
    print('Running...')
    start = timeit.default_timer()
    np.random.seed(0)
    predicted_y_test = algorithm(x_train, y_train, x_test)
    np.random.seed()
    stop = timeit.default_timer()
    run_time = stop - start

    correct_predict = (y_test
                       == predicted_y_test).astype(np.int32).sum()
    incorrect_predict = len(y_test) - correct_predict
    accuracy = float(correct_predict) / len(y_test)

    print('Correct Predict: {}/{} total \tAccuracy: {:5f} \tTime: {:2f}'.format(correct_predict,
            len(y_test), accuracy, run_time))
    return (correct_predict, accuracy, run_time)


def knn(x_train, y_train, x_test):
    """
    x_train: 60000 x 784 matrix: each row is a flattened image of an MNIST digit
    y_train: 60000 vector: label for x_train
    x_test: 1000 x 784 testing images
    return: predicted y_test which is a 1000 vector
    """

    # init graph
    g = tf.Graph()
    with g.as_default() as g:
      # set placeholders to be used in sess
      x_train_ph = tf.placeholder(dtype=tf.float32,shape=(55000, 784))
      y_train_ph = tf.placeholder(dtype=tf.float32,shape=(55000))
      x_test_ph = tf.placeholder(dtype=tf.float32,shape=(784))

      # init y_test to hold predicted label for the 1000 x_test testing images
      y_test = np.zeros(1000)

      # set k-value
      k = 3

      # compute sum of elements between x_test and x_train (total distance)
      d = tf.reduce_sum(tf.abs(tf.subtract(x_train_ph, x_test_ph)), axis=1)

      # get the k-nearest-neighbors indexs
      k_n_n = tf.nn.top_k(tf.negative(d), k=k, sorted=False)[1]

      # get the labels of k-nearest neighbors, append to n_n
      n_n = []
      for i in range(k):
        n_n.append(y_train_ph[k_n_n[i]])

      # output k_nearest neighbors, to be executed by sess
      output = n_n

    # start tensorflow session
    with tf.Session(graph=g) as sess:
      # init global variables
      sess.run(tf.global_variables_initializer())

      # for each test image, append predicted label to y_test
      for i in range(len(y_test)):
        # sess.run returns the k-nearest labels, np.bincount() counts the frequency of each label, np.argmax() selects most frequent label
        y_test[i] = np.argmax(np.bincount(sess.run(output, feed_dict={x_train_ph: x_train, x_test_ph: x_test[i, :], y_train_ph:y_train})))

    return y_test


mnist = read_data_sets('data', one_hot=False)
result = [OrderedDict(first_name='Insert your First name here',
          last_name='Insert your Last name here')]

(x_train, y_train) = (mnist.train._images, mnist.train._labels)
(x_valid, y_valid) = (mnist.test._images, mnist.test.labels)

# Only testing first 1000 samples of test set
(x_valid, y_valid) = (x_valid[:1000], y_valid[:1000])

print("Dimension of dataset: ")
print("Train:", x_train.shape, y_train.shape, "\nTest:", x_valid.shape, y_valid.shape)

(correct_predict, accuracy, run_time) = run(knn, x_train, y_train, x_valid, y_valid)
result = OrderedDict(correct_predict=correct_predict,
                     accuracy=accuracy,
                     run_time=run_time)

with open('result.txt', 'w') as f:
    f.writelines(pformat(result, indent=4))

print(pformat(result, indent=4))
