import os
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pprint import pprint
import tensorflow as tf

train_features = np.load('train_features.npy')
train_labels = np.load('train_labels.npy')

valid_features = np.load('validate_features.npy')
valid_labels = np.load('validate_labels.npy')

test_features = np.load('test_features.npy')
test_labels = np.load('test_labels.npy')

features = tf.placeholder(dtype = tf.float32)
labels = tf.placeholder(dtype = tf.float32)

weights = tf.Variable(tf.truncated_normal(shape = [784, 10]))
bias = tf.Variable(tf.zeros(shape = [10]))

logits = tf.matmul(features, weights) + bias

prediction = tf.nn.softmax(logits)

cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)

loss = tf.reduce_mean(cross_entropy)

train_feed_dict = {features: train_features, labels: train_labels}
valid_feed_dict = {features: valid_features, labels: valid_labels}
test_feed_dict = {features: test_features, labels: test_labels}

is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))

accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

    # TODO: Find the best parameters for each configuration

epochs = 1
batch_size = 50
learning_rate = 0.01

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# The accuracy measured against the validation set
validation_accuracy = 0.0

# Measurements use for graphing loss and accuracy
log_batch_step = 50
batches = []
loss_batch = []
train_acc_batch = []
valid_acc_batch = []

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    batch_count = int(math.ceil(len(train_features)/batch_size))

    for epoch_i in range(epochs):

        # Progress bar
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')

        # The training cycle
        for batch_i in batches_pbar:
            # Get a batch of training features and labels
            batch_start = batch_i*batch_size
            batch_features = train_features[batch_start:batch_start + batch_size]
            batch_labels = train_labels[batch_start:batch_start + batch_size]

            # Run optimizer and get loss
            _, l = session.run(
                [optimizer, loss],
                feed_dict={features: batch_features, labels: batch_labels})

            # Log every 50 batches
            if not batch_i % log_batch_step:
                # Calculate Training and Validation accuracy
                training_accuracy = session.run(accuracy, feed_dict=train_feed_dict)
                validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

                # Log batches
                previous_batch = batches[-1] if batches else 0
                batches.append(log_batch_step + previous_batch)
                loss_batch.append(l)
                train_acc_batch.append(training_accuracy)
                valid_acc_batch.append(validation_accuracy)

        # Check accuracy against Validation data
        validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

loss_plot = plt.subplot(211)
loss_plot.set_title('Loss')
loss_plot.plot(batches, loss_batch, 'g')
loss_plot.set_xlim([batches[0], batches[-1]])
acc_plot = plt.subplot(212)
acc_plot.set_title('Accuracy')
acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
acc_plot.plot(batches, valid_acc_batch, 'x', label='Validation Accuracy')
acc_plot.set_ylim([0, 1.0])
acc_plot.set_xlim([batches[0], batches[-1]])
acc_plot.legend(loc=1)
plt.tight_layout()
plt.show()

print('Validation accuracy at {}'.format(validation_accuracy))
