import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import math
import os
import scipy.io
import scipy.io as sio
# Use PrettyTensor to simplify Neural Network construction.
import prettytensor as pt

# Generate the One-Hot encoded class-labels from an array of integers.
# For example, if class_number=2 and num_classes=3 then
# the one-hot encoded label is the float array: [0. 0. 1.]
def one_hot_encoded(class_numbers, num_classes=None):
    if num_classes is None:
        num_classes = np.max(class_numbers) - 1

    return np.eye(num_classes, dtype=float)[class_numbers]


#   input data 
traindata = sio.loadmat('traindata.mat')
trainX = traindata['trainX']
print("training data size", trainX.shape)
train_data = np.reshape(trainX,(trainX.shape[0],64,64,3))
print(trainX.shape)
print("training image data size", trainX.shape)
trainY = traindata['trainY']
train_classes = np.reshape(np.array(trainY), (len(train_data),))
train_classes = np.array(train_classes, dtype=np.int64)
train_labels_total = one_hot_encoded(class_numbers=train_classes, num_classes=2)
print("classes labels size ", train_classes.shape)

# testing data
testdata = sio.loadmat('testdata.mat')
testX = testdata['testX']
print("testing data size", testX.shape)
test_data = np.reshape(testX,(testX.shape[0],64,64,3))

# training data
images_train = train_data[1:22000]
cls_train = train_classes[1:22000]
labels_train = train_labels_total[1:22000]

#validation data
images_test = train_data[22000:25000]
cls_test = train_classes[22000:25000]
labels_test = train_labels_total[22000:25000]

#blanck 
blank_test_labels = np.zeros(shape=labels_test.shape, dtype=np.int)

img_size = 64
# 3 channels: Red, Green, Blue.
num_channels = 3
img_size_flat = img_size * img_size * num_channels
num_classes = 2

# placeholder variable, as tensorflow only inputs tensorflow graph so to 
# change our input vector into tensorflow graph these placeholder are used 
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


# function to create main network 2 convolution layer with 2 fully connected layer
# uses pretytensor library(based on tensorflow), it use batch normalization in first layer
def main_network(images, training):
    x_pretty = pt.wrap(images)
    if training:
        phase = pt.Phase.train
    else:
        phase = pt.Phase.infer

    with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
        y_pred, loss = x_pretty.\
            conv2d(kernel=5, depth=64, name='layer_conv1', batch_normalize=True).\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=5, depth=64, name='layer_conv2').\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=256, name='layer_fc1').\
            fully_connected(size=128, name='layer_fc2').\
            softmax_classifier(class_count=num_classes, labels=y_true)

    return y_pred, loss

# Wrap the neural network in the scope named 'network'.
# Create new variables during training, and re-use during testing.
# create tensorflow graph
def create_network(training):
    with tf.variable_scope('network', reuse=not training):
        images = x
        y_pred, loss = main_network(images=images, training=training)

    return y_pred, loss

# A TensorFlow variable that keeps track of the number of optimization iterations performed so far
global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)

# Create the neural network to be used for training.
# we only need the loss-function during training.
_, loss = create_network(training=True)

# Create an optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)

# the neural network for the test-phase
# testing we only need y_pred
y_pred, _ = create_network(training=False)

# calculate the predicted class number as an integer
y_pred_cls = tf.argmax(y_pred, dimension=1)

# to calculate accuracy
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# tensorflow saver and session object
saver = tf.train.Saver()
session = tf.Session()

# creating checkpoint and storing it, so we dont have to do optimization every time
save_dir = 'checkpoints_main1/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = save_dir + 'cifar10_cnn'

# Use TensorFlow to find the latest checkpoint - if any.
# Try and load the data in the checkpoint.
# If the above failed for some reason, simply
# initialize all the variables for the TensorFlow graph.    
try:
    print("Trying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
    saver.restore(session, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except:
    print("Failed to restore checkpoint. Initializing variables instead.")
    session.run(tf.initialize_all_variables())

# function to get random batch of training images
train_batch_size = 150
def random_batch():
    num_images = len(images_train)
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)
    x_batch = images_train[idx, :, :, :]
    y_batch = labels_train[idx, :]
    return x_batch, y_batch

# This function performs a number of optimization iterations 
# so as to gradually improve the variables of the network layers
def optimize(num_iterations):
    start_time = time.time()

    for i in range(num_iterations):
        x_batch, y_true_batch = random_batch()

        # A dict for placeholder variables 
        # in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        
        i_global, _ = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)

        # Print status to screen every 100 iterations
        if (i_global % 100 == 0) or (i == num_iterations - 1):
            batch_acc = session.run(accuracy,
                                    feed_dict=feed_dict_train)

            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))

        # Save a checkpoint to disk every 1000 iterations.
        if (i_global % 1000 == 0) or (i == num_iterations - 1):
            saver.save(session,
                       save_path=save_path,
                       global_step=global_step)

            print("Saved checkpoint.")

    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# This function calculates the predicted classes of images
batch_size = 200
def predict_cls(images, labels, cls_true):
    num_images = len(images)
    cls_pred = np.zeros(shape=num_images, dtype=np.int)
    i = 0

    while i < num_images:
        j = min(i + batch_size, num_images)
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    correct = (cls_true == cls_pred)
    return correct, cls_pred

# This function calculates the predicted classes of images
def predict_cls_test():
    return predict_cls(images = images_test,
                       labels = labels_test,
                       cls_true = cls_test)

# functions to print accuracy
# return classification and number of correct classification
def classification_accuracy(correct):
    return correct.mean(), correct.sum()

def print_test_accuracy():
    correct, cls_pred = predict_cls_test()
    acc, num_correct = classification_accuracy(correct)
    num_images = len(correct)
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

print_test_accuracy()
if False:
    optimize(num_iterations=200)
    print_test_accuracy()
    optimize(num_iterations=200)
    print_test_accuracy()
    optimize(num_iterations=200)
    print_test_accuracy()


# function to predict class of test data
batch_size = 200
def predict_test_cls(images, labels):
    num_images = len(images)
    cls_pred = np.zeros(shape=num_images, dtype=np.int)
    i = 0

    while i < num_images:
        j = min(i + batch_size, num_images)
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}

        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    return cls_pred

# function to wirte output to a file
def print_into_file(labels):
    num_images = len(labels)
    output_file = open("output.txt", "w")
    i = 0
    while i < num_images:
        output_file.write(str(labels[i]))
        output_file.write('\n')
        i += 1

if False:
    predicted_cls = predict_test_cls(images = test_data, labels = blank_test_labels)
    print_into_file(predicted_cls)

if False:
    session.close()
