import logging; logging.basicConfig(level=logging.INFO)
import tensorflow as tf
import pandas as pd
import ltn
import argparse

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--csv-path',type=str,default=None)
    parser.add_argument('--epochs',type=int,default=500)
    parser.add_argument('--batch-size',type=int,default=64)
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args

args = parse_args()
batch_size = args['batch_size']
EPOCHS = args['epochs']
csv_path = args['csv_path']


# # Data
# 
# Load the real estate dataset
df = pd.read_csv("real-estate.csv")
df = df.sample(frac=1) #shuffle

x = df[['X1 transaction date', 'X2 house age',
       'X3 distance to the nearest MRT station',
       'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']]
y = df[['Y house price of unit area']]
ds_train = tf.data.Dataset.from_tensor_slices((x[:330],y[:330])).batch(batch_size)
ds_test = tf.data.Dataset.from_tensor_slices((x[330:],y[330:])).batch(batch_size)


# # LTN
#
# Regressor (trainable)
f = ltn.Function.MLP(input_shapes=[6],output_shape=[1],hidden_layer_sizes=(8,8))
# Equality Predicate
eq = ltn.Predicate.Lambda(
    #lambda args: tf.exp(-0.05*tf.sqrt(tf.reduce_sum(tf.square(args[0]-args[1]),axis=1)))        
    lambda args: 1/(1+0.5*tf.sqrt(tf.reduce_sum(tf.square(args[0]-args[1]),axis=1)))
)

# Operators and axioms
Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=2),semantics="forall")
Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(p=2),semantics="exists")
@tf.function
def axioms(x_data, y_data):
    x = ltn.Variable("x", x_data)
    y = ltn.Variable("y", y_data)
    return Forall(ltn.diag(x,y), eq([f(x),y]))

# Initialize all layers and the static graph
for x, y in ds_test:
    print("Initial sat level %.5f"%axioms(x,y))
    break

# # Training
# 
# Define the metrics. While training, we measure:
# 1. The level of satisfiability of the Knowledge Base of the training data.
# 1. The level of satisfiability of the Knowledge Base of the test data.
# 3. The training accuracy.
# 4. The test accuracy.
metrics_dict = {
    'train_sat': tf.keras.metrics.Mean(name='train_sat'),
    'test_sat': tf.keras.metrics.Mean(name='test_sat'),
    'train_accuracy': tf.keras.metrics.RootMeanSquaredError(name="train_accuracy"),
    'test_accuracy': tf.keras.metrics.RootMeanSquaredError(name="test_accuracy")
}


# Define the training and test step
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
@tf.function
def train_step(x, y):
    # sat and update
    with tf.GradientTape() as tape:
        sat = axioms(x, y)
        loss = 1.-sat
    gradients = tape.gradient(loss, f.trainable_variables)
    optimizer.apply_gradients(zip(gradients, f.trainable_variables))
    sat = axioms(x, y)
    metrics_dict['train_sat'](sat)
    # accuracy
    metrics_dict['train_accuracy'](y,f.model(x))
    
@tf.function
def test_step(x, y):
    # sat
    sat = axioms(x, y)
    metrics_dict['test_sat'](sat)
    # accuracy
    metrics_dict['test_accuracy'](y,f.model(x))

# Train

import commons

commons.train(
    EPOCHS,
    metrics_dict,
    ds_train,
    ds_test,
    train_step,
    test_step,
    csv_path=csv_path,
    track_metrics=50
)