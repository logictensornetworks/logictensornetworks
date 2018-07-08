from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import logictensornetworks as ltn
import numpy as np
from logictensornetworks import Forall,Exists, Implies, And, Or, Not
from datetime import datetime
import matplotlib.pyplot as plt
from termcolor import colored, cprint

# prepare data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def overlay_pictures(pics,labels):
    permutation = np.arange(len(labels))
    np.random.shuffle(permutation)
    pics_p = pics[permutation]
    labels_p = labels[permutation]
    ppics = np.zeros_like(pics)
    llabels = np.zeros_like(np.concatenate([labels,labels_p], axis=1))
    for i in range(len(pics)):
        w = np.random.beta(2.0, 2.0)
        if np.argmax(labels[i]) <= np.argmax(labels_p[i]):
            ppics[i] = pics[i] - w*pics_p[i]
            llabels[i] = np.concatenate([labels[i], labels_p[i]], axis=0)
        else:
            ppics[i] = pics_p[i] - w*pics[i]
            llabels[i] = np.concatenate([labels_p[i], labels[i]], axis=0)
    return ppics,llabels

ltn.LAYERS = 3
ltn.set_universal_aggreg("mean")
ltn.set_tnorm("prod")
logs_path = 'semantic_mnist_log/' + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
embedding_size = 10
train_batch_size = 50
test_batch_size = 30

# begin MNIST conv net

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, [None, 784],name="x")
y_ = tf.placeholder(tf.float32, shape=[None, embedding_size*2],name="y_")

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32,name="keep_prob")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, embedding_size*2])
b_fc2 = bias_variable([embedding_size*2])

y_conv_before_softmax = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv = tf.concat([
            tf.nn.softmax(y_conv_before_softmax[:,:embedding_size]),
            tf.nn.softmax(y_conv_before_softmax[:,embedding_size:])],axis=1)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=tf.reshape(y_,(-1,embedding_size)),
        logits=tf.reshape(y_conv_before_softmax[:train_batch_size],(-1,embedding_size))))
correct_prediction = tf.equal(
    tf.argmax(tf.reshape(y_conv,(-1,embedding_size)), axis=1),
    tf.argmax(tf.reshape(y_,(-1,embedding_size)), axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# end of mnist

# begin of ltn

dd = ltn.variable("double_digit",y_conv)

def get_nth_element(n):
    def result(p):
        return p[:,n]
    return result

IS1 = {n:ltn.predicate("is_equal_to_"+str(n),embedding_size*2,
                       pred_definition=get_nth_element(n)) for n in range(10)}

IS2 = {n:ltn.predicate("is_equal_to_"+str(n),embedding_size*2,
                       pred_definition=get_nth_element(10+n)) for n in range(10)}


examples_of_1 = {n:ltn.variable("examples_of_1_"+str(n),
                    tf.gather(y_conv[:train_batch_size],
                              tf.where(tf.equal(y_[:train_batch_size,tf.constant(n)],1))[:,0]))
                 for n in range(10)}

examples_of_2 = {n:ltn.variable("examples_of_2_"+str(n),
                    tf.gather(y_conv[:train_batch_size],
                              tf.where(tf.equal(y_[:train_batch_size,tf.constant(10+n)],1))[:,0]))
                 for n in range(10)}

examples_of_12 = {(m,n):ltn.variable("examples_of_12_"+str(m)+"_"+str(n),
                    tf.gather(y_conv[:train_batch_size],
                              tf.where(
                                  tf.logical_and(
                                      tf.equal(y_[:train_batch_size,tf.constant(10+n)],1),
                                      tf.equal(y_[:train_batch_size,tf.constant(10+n)],1)))[:,0]))
                 for m in range(10) for n in range(10)}

constraints = tf.concat(
    [Forall(examples_of_1[n],IS1[n](examples_of_1[n]))
               for n in range(10)]+ \
    [Forall(examples_of_2[n],IS2[n](examples_of_2[n]))
               for n in range(10)] + \
    [Exists(dd,IS1[n](dd)) for n in range(10)] + \
    [Exists(dd,IS2[n](dd)) for n in range(10)] + \
    [Forall(dd,Not(And(IS1[n](dd),IS2[m](dd))))
               for n in range(10) for m in range(n)],0)

# [Forall(dd, And(Not(And(IS1[n](dd), IS1[m](dd))),
#                 Not(And(IS2[n](dd), IS2[m](dd)))))
#  for m in range(10) for n in range(m) if m != n] + \
# [Forall(dd, And(Or(*[IS1[n](dd) for n in range(10)]),
#                 Or(*[IS2[n](dd) for n in range(10)])))] + \
    # [Forall(examples_of_1[n],Not(IS1[m](examples_of_1[n])))
#  for m in range(10)
#  for n in range(10) if n != m] + \
# [Forall(examples_of_2[n], Not(IS2[m](examples_of_2[n])))
#  for m in range(10)
#  for n in range(10) if n != m] + \


constr_min = tf.reduce_min(constraints,axis=0)
constr_hmean = 1/tf.reduce_mean(1/constraints)
constr_mean = tf.reduce_mean(constraints,axis=0)
constr_sqmean = tf.reduce_mean(tf.square(constraints),axis=0)
constr_var = tf.reduce_mean(tf.square(constraints-constr_mean))
constraints_loss = (1.0-constr_mean)

optimize_only_conv_net = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
optimize_conv_net_and_ltn = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy+constraints_loss)
optimize_only_ltn = tf.train.AdamOptimizer(1e-3).minimize(constraints_loss)

init = tf.global_variables_initializer()

def get_feed_dict(train_size,test_size):
    global x,y_keep_prob
    train_x, train_y = mnist.train.next_batch(train_size)
    train_x,train_y = overlay_pictures(train_x,train_y)
    test_x = mnist.test.images[np.random.randint(len(mnist.test.images),size=test_size)]
    test_x,_ = overlay_pictures(test_x,mnist.test.labels[:test_size])
    feed_dict = {}
    feed_dict[x] = np.concatenate([train_x,test_x],0)
    feed_dict[y_] = train_y
    feed_dict[keep_prob] = 1.0
    return feed_dict

with tf.Session() as sess:
    sess.run(init)
    for i in range(20000):
        if i%100 == 0:
            train_feed_dict = get_feed_dict(train_batch_size,test_batch_size)
            test_images,test_labels = overlay_pictures(mnist.test.images,mnist.test.labels)
            train_feed_dict[keep_prob] = 1.0
            sc = sess.run(constraints,feed_dict=train_feed_dict)
            for s in sc:
                cprint(s,"red")
            print("accuracy at %d = %.5e"%
                  (i,sess.run(accuracy,
                              feed_dict={x:test_images,
                                         y_:test_labels,
                                        keep_prob:1.0})))


        if i < 0:
            train_feed_dict[keep_prob] = .5
            sess.run(optimize_only_conv_net,feed_dict=train_feed_dict)
        else:
            train_feed_dict[keep_prob] = 1.0
            sess.run(optimize_only_ltn,feed_dict=train_feed_dict)

