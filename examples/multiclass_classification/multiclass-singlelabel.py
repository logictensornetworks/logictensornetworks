import logging; logging.basicConfig(level=logging.INFO)
import tensorflow as tf
import pandas as pd
import ltn
import argparse

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--csv-path',type=str,default="crabs_results.csv")
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
# Load the iris dataset: 50 samples from each of three species of iris flowers (setosa, virginica, versicolor), measured with four features.

df_train = pd.read_csv("iris_training.csv")
df_test = pd.read_csv("iris_test.csv")

labels_train = df_train.pop("species")
labels_test = df_test.pop("species")
ds_train = tf.data.Dataset.from_tensor_slices((df_train,labels_train)).batch(batch_size)
ds_test = tf.data.Dataset.from_tensor_slices((df_test,labels_test)).batch(batch_size)


# # LTN
# 
# Predicate with softmax `P(x,class)`

class MLP(tf.keras.Model):
    """Model that returns logits."""
    def __init__(self, n_classes, hidden_layer_sizes=(16,16,8)):
        super(MLP, self).__init__()
        self.denses = [tf.keras.layers.Dense(s, activation="elu") for s in hidden_layer_sizes]
        self.dense_class = tf.keras.layers.Dense(n_classes)
        self.dropout = tf.keras.layers.Dropout(0.2)
        
    def call(self, inputs, training=False):
        x = inputs[0]
        for dense in self.denses:
            x = dense(x)
            x = self.dropout(x, training=training)
        return self.dense_class(x)

logits_model = MLP(3)
p = ltn.Predicate.FromLogits(logits_model, activation_function="softmax", with_class_indexing=True)


# Constants to index/iterate on the classes
class_A = ltn.Constant(0, trainable=False)
class_B = ltn.Constant(1, trainable=False)
class_C = ltn.Constant(2, trainable=False)


# Operators and axioms
Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=2),semantics="forall")

formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=2))

@tf.function
def axioms(features, labels, training=True):
    x_A = ltn.Variable("x_A",features[labels==0])
    x_B = ltn.Variable("x_B",features[labels==1])
    x_C = ltn.Variable("x_C",features[labels==2])
    axioms = [
        Forall(x_A,p([x_A,class_A],training=training)),
        Forall(x_B,p([x_B,class_B],training=training)),
        Forall(x_C,p([x_C,class_C],training=training))
    ]
    sat_level = formula_aggregator(axioms).tensor
    return sat_level

# Initialize all layers and the static graph
for features, labels in ds_train:
    print("Initial sat level %.5f"%axioms(features,labels,training=False))
    break


# # Training
# 
# Define the metrics. While training, we measure:
# 1. The level of satisfiability of the Knowledge Base of the training data.
# 1. The level of satisfiability of the Knowledge Base of the test data.
# 3. The training accuracy.
# 4. The test accuracy.

metrics_dict = {
    'train_sat_kb': tf.keras.metrics.Mean(name='train_sat_kb'),
    'test_sat_kb': tf.keras.metrics.Mean(name='test_sat_kb'),
    'train_accuracy': tf.keras.metrics.CategoricalAccuracy(name="train_accuracy"),
    'test_accuracy': tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")
}


# Define the training and test step
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step(features, labels):
    # sat and update
    with tf.GradientTape() as tape:
        sat = axioms(features, labels, training=True)
        loss = 1.-sat
    gradients = tape.gradient(loss, p.trainable_variables)
    optimizer.apply_gradients(zip(gradients, p.trainable_variables))
    sat = axioms(features, labels) # compute sat without dropout
    metrics_dict['train_sat_kb'](sat)
    # accuracy
    predictions = p.logits_model([features])
    metrics_dict['train_accuracy'](tf.one_hot(labels,3),predictions)
    
@tf.function
def test_step(features, labels):
    # sat
    sat = axioms(features, labels, training=False)
    metrics_dict['test_sat_kb'](sat)
    # accuracy
    predictions = p.logits_model([features])
    metrics_dict['test_accuracy'](tf.one_hot(labels,3),predictions)


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
    track_metrics=20
)
