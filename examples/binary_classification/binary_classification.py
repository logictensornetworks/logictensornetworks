import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import ltn
import argparse

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--csv-path',type=str,default=None)
    parser.add_argument('--epochs',type=int,default=1000)
    parser.add_argument('--batch-size',type=int,default=64)
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args

args = parse_args()
batch_size = args['batch_size']
EPOCHS = args['epochs']
csv_path = args['csv_path']

# # Data
# Sample data from [0,1]^2.
# The groundtruth positive is data close to the center (.5,.5) (given a threshold)
# All the other data is considered as negative examples
nr_samples = 100
data = np.random.uniform([0,0],[1,1],(nr_samples,2))
labels = np.sum(np.square(data-[.5,.5]),axis=1)<.09
# 50 examples for training; 50 examples for testing
ds_train = tf.data.Dataset.from_tensor_slices((data[:50],labels[:50])).batch(batch_size)
ds_test = tf.data.Dataset.from_tensor_slices((data[50:],labels[50:])).batch(batch_size)

# # LTN

A = ltn.Predicate.MLP([2],hidden_layer_sizes=(16,16))

# # Axioms
# 
# ```
# forall x_A: A(x_A)
# forall x_not_A: ~A(x_not_A)
# ```

Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=2),semantics="forall")
Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(p=2),semantics="exists")

formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=2))

@tf.function
def axioms(data, labels):
    x_A = ltn.Variable("x_A",data[labels])
    x_not_A = ltn.Variable("x_not_A",data[tf.logical_not(labels)])
    axioms = [
        Forall(x_A, A(x_A)),
        Forall(x_not_A, Not(A(x_not_A)))
    ]
    sat_level = formula_aggregator(axioms).tensor
    return sat_level


# Initialize all layers and the static graph.

for _data, _labels in ds_test:
    print("Initial sat level %.5f"%axioms(_data, _labels))
    break

# # Training
# 
# Define the metrics

metrics_dict = {
    'train_sat': tf.keras.metrics.Mean(name='train_sat'),
    'test_sat': tf.keras.metrics.Mean(name='test_sat'),
    'train_accuracy': tf.keras.metrics.BinaryAccuracy(name="train_accuracy",threshold=0.5),
    'test_accuracy': tf.keras.metrics.BinaryAccuracy(name="test_accuracy",threshold=0.5)
}

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
@tf.function
def train_step(data, labels):
    # sat and update
    with tf.GradientTape() as tape:
        sat = axioms(data, labels)
        loss = 1.-sat
    gradients = tape.gradient(loss, A.trainable_variables)
    optimizer.apply_gradients(zip(gradients, A.trainable_variables))
    metrics_dict['train_sat'](sat)
    # accuracy
    predictions = A.model(data)
    metrics_dict['train_accuracy'](labels,predictions)

@tf.function
def test_step(data, labels):
    # sat and update
    sat = axioms(data, labels)
    metrics_dict['test_sat'](sat)
    # accuracy
    predictions = A.model(data)
    metrics_dict['test_accuracy'](labels,predictions)

import commons

commons.train(
    EPOCHS,
    metrics_dict,
    ds_train,
    ds_test,
    train_step,
    test_step,
    csv_path=csv_path,
    track_metrics=20
)
