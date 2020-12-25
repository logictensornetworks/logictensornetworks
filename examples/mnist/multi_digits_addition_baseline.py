import argparse
import tensorflow as tf
from tensorflow.keras import layers
import baselines, data, commons
import argparse

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--csv-path',type=str,default="MDadd_baseline.csv")
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--n-examples-train',type=int,default=15000)
    parser.add_argument('--n-examples-test',type=int,default=2500)
    parser.add_argument('--batch-size',type=int,default=32)
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args

args = parse_args()
n_examples_train = args['n_examples_train']
n_examples_test = args['n_examples_test']
batch_size = args['batch_size']
EPOCHS = args['epochs']
csv_path = args['csv_path']

""" DATASET """

ds_train, ds_test = data.get_mnist_op_dataset(
        count_train=n_examples_train,
        count_test=n_examples_test,
        buffer_size=10000,
        batch_size=batch_size,
        n_operands=4,
        op=lambda args: 10*args[0]+args[1]+10*args[2]+args[3])

""" MODEL AND LOSS """

multi_digits_addition = baselines.MultiDigits(199,hidden_dense_sizes=(128,))
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

""" TRAINING """

optimizer = tf.keras.optimizers.Adam(0.001)
metrics_dict = {
    'train_loss': tf.keras.metrics.Mean(name="train_loss"),
    'train_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy"),
    'test_loss': tf.keras.metrics.Mean(name="test_loss"),
    'test_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")    
}

@tf.function
def train_step(images_x1,images_x2,images_y1,images_y2,labels_z,**kwargs):
    # loss
    with tf.GradientTape() as tape:
        logits = multi_digits_addition([images_x1,images_x2,images_y1,images_y2])
        loss = loss_fn(labels_z,logits)
    gradients = tape.gradient(loss, multi_digits_addition.trainable_variables)
    optimizer.apply_gradients(zip(gradients, multi_digits_addition.trainable_variables))
    metrics_dict['train_loss'](loss)
    metrics_dict['train_accuracy'](labels_z, logits)

@tf.function
def test_step(images_x1,images_x2,images_y1,images_y2,labels_z,**kwargs):
    logits = multi_digits_addition([images_x1,images_x2,images_y1,images_y2])
    loss = loss_fn(labels_z,logits)
    metrics_dict['test_loss'](loss)
    metrics_dict['test_accuracy'](labels_z, logits)

commons.train(
    EPOCHS,
    metrics_dict,
    ds_train,
    ds_test,
    train_step,
    test_step,
    csv_path=csv_path
)