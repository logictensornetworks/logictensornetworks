import argparse
import tensorflow as tf
from tensorflow.keras import layers
import baselines, data, commons

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--csv-path',type=str,default="SDadd_baseline.csv")
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--n-examples-train',type=int,default=30000)
    parser.add_argument('--n-examples-test',type=int,default=5000)
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
        n_operands=2,
        op=lambda args: args[0]+args[1])

""" MODEL AND LOSS """

single_digits_addition = baselines.MultiDigits(19)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

""" TRAINING"""

optimizer = tf.keras.optimizers.Adam(0.001)
metrics_dict = {
    'train_loss': tf.keras.metrics.Mean(name="train_loss"),
    'train_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy"),
    'test_loss': tf.keras.metrics.Mean(name="test_loss"),
    'test_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")    
}

@tf.function
def train_step(images_x, images_y, labels_z):
    with tf.GradientTape() as tape:
        logits = single_digits_addition([images_x,images_y])
        loss = loss_fn(labels_z, logits)
    gradients = tape.gradient(loss, single_digits_addition.trainable_variables)
    optimizer.apply_gradients(zip(gradients, single_digits_addition.trainable_variables))
    
    metrics_dict['train_loss'](loss)
    metrics_dict['train_accuracy'](labels_z, logits)

@tf.function
def test_step(images_x, images_y, labels_z):
    logits = single_digits_addition([images_x,images_y])
    loss = loss_fn(labels_z, logits)
    
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