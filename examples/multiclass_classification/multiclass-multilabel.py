import logging; logging.basicConfig(level=logging.INFO)
import tensorflow as tf
import ltn
import pandas as pd
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
# Crabs dataset from: http://www.stats.ox.ac.uk/pub/PRNN/
# 
# The crabs data frame has 200 rows and 8 columns, describing 5 morphological measurements on 50 crabs each of two colour forms and both sexes, of the species Leptograpsus variegatus collected at Fremantle, W. Australia.
# 
# - Multi-class: Male, Female, Blue, Orange.
# - Multi-label: Only Male-Female and Blue-Orange are mutually exclusive.
# 
df = pd.read_csv("crabs.dat",sep=" ", skipinitialspace=True)
df = df.sample(frac=1) #shuffle

features = df[['FL','RW','CL','CW','BD']]
labels_sex = df['sex']
labels_color = df['sp']

ds_train = tf.data.Dataset.from_tensor_slices((features[:160],labels_sex[:160],labels_color[:160])).batch(batch_size)
ds_test = tf.data.Dataset.from_tensor_slices((features[160:],labels_sex[160:],labels_color[160:])).batch(batch_size)


# # LTN
# 
# ### Predicate
# 
# | index | class | 
# | --- | --- |
# | 0 | Male |
# | 1 | Female |
# | 2 | Blue |
# | 3 | Orange |
# 
# Let's note that, since the classes are not mutually exclusive, the last layer of the model will be a `sigmoid` and not a `softmax`.

class MLP(tf.keras.Model):
    """Model that returns logits."""
    def __init__(self, n_classes, hidden_layer_sizes=(16,16,8)):
        super(MLP, self).__init__()
        self.denses = [tf.keras.layers.Dense(s, activation="elu") for s in hidden_layer_sizes]
        self.dense_class = tf.keras.layers.Dense(n_classes)
        
    def call(self, inputs):
        x = inputs[0]
        for dense in self.denses:
            x = dense(x)
        return self.dense_class(x)
logits_model = MLP(4)
p = ltn.Predicate.FromLogits(logits_model, activation_function="sigmoid", with_class_indexing=True)

# Constants to index the classes
class_male = ltn.Constant(0, trainable=False)
class_female = ltn.Constant(1, trainable=False)
class_blue = ltn.Constant(2, trainable=False)
class_orange = ltn.Constant(3, trainable=False)


# ### Axioms
# 
# ```
# forall x_blue: C(x_blue,blue)
# forall x_orange: C(x_orange,orange)
# forall x_male: C(x_male,male)
# forall x_female: C(x_female,female)
# forall x: ~(C(x,male) & C(x,female))
# forall x: ~(C(x,blue) & C(x,orange))
# ```

Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=2),semantics="forall")

formula_aggregator = ltn.fuzzy_ops.Aggreg_pMeanError(p=2)

formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=2))

@tf.function
def axioms(features,labels_sex,labels_color):
    x = ltn.Variable("x",features)
    x_blue = ltn.Variable("x_blue",features[labels_color=="B"])
    x_orange = ltn.Variable("x_orange",features[labels_color=="O"])
    x_male = ltn.Variable("x_blue",features[labels_sex=="M"])
    x_female = ltn.Variable("x_blue",features[labels_sex=="F"])
    axioms = [
        Forall(x_blue, p([x_blue,class_blue])),
        Forall(x_orange, p([x_orange,class_orange])),
        Forall(x_male, p([x_male,class_male])),
        Forall(x_female, p([x_female,class_female])),
        Forall(x,Not(And(p([x,class_blue]),p([x,class_orange])))),
        Forall(x,Not(And(p([x,class_male]),p([x,class_female]))))
    ]
    sat_level = formula_aggregator(axioms).tensor
    return sat_level


# Initialize all layers and the static graph.

for features, labels_sex, labels_color in ds_train:
    print("Initial sat level %.5f"%axioms(features,labels_sex,labels_color))
    break


# # Training
# 
# Define the metrics
# While training, we measure:
# 1. The level of satisfiability of the Knowledge Base of the training data.
# 2. The level of satisfiability of the Knowledge Base of the test data.
# 3. The training accuracy.
# 4. The test accuracy.
# 5. The level of satisfiability of a formula phi_1 we expect to have a high truth value. 
#       forall x (p(x,blue)->~p(x,orange))
# 6. The level of satisfiability of a formula phi_1 we expect to have a low truth value. 
#       forall x (p(x,blue)->p(x,orange))
# 7. The level of satisfiability of a formula phi_1 we expect to have a neither high neither low truth value. 
#       forall x (p(x,blue)->p(x,male))

metrics_dict = {
    'train_sat_kb': tf.keras.metrics.Mean(name='train_sat_kb'),
    'test_sat_kb': tf.keras.metrics.Mean(name='test_sat_kb'),
    'train_accuracy': tf.keras.metrics.Mean(name="train_accuracy"),
    'test_accuracy': tf.keras.metrics.Mean(name="test_accuracy"),
    'test_sat_phi1': tf.keras.metrics.Mean(name='test_sat_phi1'),
    'test_sat_phi2': tf.keras.metrics.Mean(name='test_sat_phi2'),
    'test_sat_phi3': tf.keras.metrics.Mean(name='test_sat_phi3')
}

@tf.function()
def sat_phi1(features):
    x = ltn.Variable("x",features)
    phi1 = Forall(x, Implies(p([x,class_blue]),Not(p([x,class_orange]))),p=5)
    return phi1.tensor

@tf.function()
def sat_phi2(features):
    x = ltn.Variable("x",features)
    phi2 = Forall(x, Implies(p([x,class_blue]),p([x,class_orange])),p=5)
    return phi2.tensor

@tf.function()
def sat_phi3(features):
    x = ltn.Variable("x",features)
    phi3 = Forall(x, Implies(p([x,class_blue]),p([x,class_male])),p=5)
    return phi3.tensor

def multilabel_hamming_loss(y_true, y_pred, threshold=0.5,from_logits=False):
    if from_logits:
        y_pred = tf.math.sigmoid(y_pred)
    y_pred = y_pred > threshold
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)
    nonzero = tf.cast(tf.math.count_nonzero(y_true-y_pred,axis=-1),tf.float32)
    return nonzero/y_true.get_shape()[-1]


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
@tf.function
def train_step(features, labels_sex, labels_color):
    # sat and update
    with tf.GradientTape() as tape:
        sat = axioms(features, labels_sex, labels_color)
        loss = 1.-sat
    gradients = tape.gradient(loss, p.trainable_variables)
    optimizer.apply_gradients(zip(gradients, p.trainable_variables))
    metrics_dict['train_sat_kb'](sat)
    # accuracy
    predictions = logits_model([features])
    labels_male = (labels_sex == "M")
    labels_female = (labels_sex == "F")
    labels_blue = (labels_color == "B")
    labels_orange = (labels_color == "O")
    onehot = tf.stack([labels_male,labels_female,labels_blue,labels_orange],axis=-1)
    metrics_dict['train_accuracy'](1-multilabel_hamming_loss(onehot,predictions,from_logits=True))
    
@tf.function
def test_step(features, labels_sex, labels_color):
    # sat
    sat_kb = axioms(features, labels_sex, labels_color)
    metrics_dict['test_sat_kb'](sat_kb)
    metrics_dict['test_sat_phi1'](sat_phi1(features))
    metrics_dict['test_sat_phi2'](sat_phi2(features))
    metrics_dict['test_sat_phi3'](sat_phi3(features))
    # accuracy
    predictions = logits_model([features])
    labels_male = (labels_sex == "M")
    labels_female = (labels_sex == "F")
    labels_blue = (labels_color == "B")
    labels_orange = (labels_color == "O")
    onehot = tf.stack([labels_male,labels_female,labels_blue,labels_orange],axis=-1)
    metrics_dict['test_accuracy'](1-multilabel_hamming_loss(onehot,predictions,from_logits=True))

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
