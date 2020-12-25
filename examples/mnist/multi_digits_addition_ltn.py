import argparse
import tensorflow as tf
from tensorflow.keras import layers
import logictensornetworks as ltn
import baselines, data, commons
import argparse

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--csv-path',type=str,default="MDadd_ltn.csv")
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--n-examples-train',type=int,default=15000)
    parser.add_argument('--n-examples-test',type=int,default=2500)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--p',type=str,choices=["1","2","schedule"], default="schedule")
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args

args = parse_args()
n_examples_train = args['n_examples_train']
n_examples_test = args['n_examples_test']
batch_size = args['batch_size']
EPOCHS = args['epochs']
csv_path = args['csv_path']
p_type = args["p"]

""" DATASET """

ds_train, ds_test = data.get_mnist_op_dataset(
        count_train=n_examples_train,
        count_test=n_examples_test,
        buffer_size=10000,
        batch_size=batch_size,
        n_operands=4,
        op=lambda args: 10*args[0]+args[1]+10*args[2]+args[3])

""" LTN MODEL AND LOSS """
### Predicates
logits_model = baselines.SingleDigit()
Digit = ltn.Predicate(ltn.utils.LogitsToPredicateModel(logits_model))
### Variables
d1 = ltn.variable("digits1", range(10))
d2 = ltn.variable("digits2", range(10))
d3 = ltn.variable("digits3", range(10))
d4 = ltn.variable("digits4", range(10))
### Operators
Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(),semantics="forall")
Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(),semantics="exists")
### Axioms
@tf.function
def axioms(images_x1,images_x2,images_y1,images_y2,labels_z,p_schedule):
    images_x1 = ltn.variable("x1", images_x1)
    images_x2 = ltn.variable("x2", images_x2)
    images_y1 = ltn.variable("y1", images_y1)
    images_y2 = ltn.variable("y2", images_y2)
    labels_z = ltn.variable("z", labels_z)
    return Forall(
            ltn.diag(images_x1,images_x2,images_y1,images_y2,labels_z),
            Exists(
                (d1,d2,d3,d4),
                And(
                    And(Digit([images_x1,d1]),Digit([images_x2,d2])),
                    And(Digit([images_y1,d3]),Digit([images_y2,d4]))
                ),
                mask_vars=[d1,d2,d3,d4,labels_z],
                mask_fn=lambda vars: tf.equal(10*vars[0]+vars[1]+10*vars[2]+vars[3],vars[4]),
                p=p_schedule
            ),
            p=1
        )
### Initialize layers and weights
x1, x2, y1, y2, z = next(ds_train.as_numpy_iterator())
axioms(x1, x2, y1, y2, z, 2)

""" TRAINING """

optimizer = tf.keras.optimizers.Adam(0.001)
metrics_dict = {
    'train_loss': tf.keras.metrics.Mean(name="train_loss"),
    'train_accuracy': tf.keras.metrics.Mean(name="train_accuracy"),
    'test_loss': tf.keras.metrics.Mean(name="test_loss"),
    'test_accuracy': tf.keras.metrics.Mean(name="test_accuracy")    
}

@tf.function
def train_step(images_x1,images_x2,images_y1,images_y2,labels_z,**kwargs):
    # loss
    with tf.GradientTape() as tape:
        loss = 1.- axioms(images_x1,images_x2,images_y1,images_y2,labels_z,**kwargs)
    gradients = tape.gradient(loss, logits_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, logits_model.trainable_variables))
    metrics_dict['train_loss'](loss)
    # accuracy
    predictions_x1 = tf.argmax(logits_model(images_x1),axis=-1)
    predictions_x2 = tf.argmax(logits_model(images_x2),axis=-1)
    predictions_y1 = tf.argmax(logits_model(images_y1),axis=-1)
    predictions_y2 = tf.argmax(logits_model(images_y2),axis=-1)
    predictions_z = 10*predictions_x1+predictions_x2+10*predictions_y1+predictions_y2
    match = tf.equal(predictions_z,tf.cast(labels_z,predictions_z.dtype))
    metrics_dict['train_accuracy'](tf.reduce_mean(tf.cast(match,tf.float32)))
    
@tf.function
def test_step(images_x1,images_x2,images_y1,images_y2,labels_z,**kwargs):
    # loss
    loss = 1.- axioms(images_x1,images_x2,images_y1,images_y2,labels_z,**kwargs)
    metrics_dict['test_loss'](loss)
    # accuracy
    predictions_x1 = tf.argmax(logits_model(images_x1),axis=-1)
    predictions_x2 = tf.argmax(logits_model(images_x2),axis=-1)
    predictions_y1 = tf.argmax(logits_model(images_y1),axis=-1)
    predictions_y2 = tf.argmax(logits_model(images_y2),axis=-1)
    predictions_z = 10*predictions_x1+predictions_x2+10*predictions_y1+predictions_y2
    match = tf.equal(predictions_z,tf.cast(labels_z,predictions_z.dtype))
    metrics_dict['test_accuracy'](tf.reduce_mean(tf.cast(match,tf.float32)))

from collections import defaultdict

scheduled_parameters = defaultdict(lambda: {})
if p_type=="schedule":
    for epoch in range(0,6):
        scheduled_parameters[epoch] = {"p_schedule":tf.constant(1.)}
    for epoch in range(6,12):
        scheduled_parameters[epoch] = {"p_schedule":tf.constant(2.)}
    for epoch in range(12,18):
        scheduled_parameters[epoch] = {"p_schedule":tf.constant(4.)}
    for epoch in range(18,EPOCHS):
        scheduled_parameters[epoch] = {"p_schedule":tf.constant(6.)}
if p_type=="1":
    for epoch in range(EPOCHS):
        scheduled_parameters[epoch] = {"p_schedule":tf.constant(1.)}
if p_type=="2":
    for epoch in range(EPOCHS):
        scheduled_parameters[epoch] = {"p_schedule":tf.constant(2.)}

commons.train(
    EPOCHS,
    metrics_dict,
    ds_train,
    ds_test,
    train_step,
    test_step,
    csv_path=csv_path,
    scheduled_parameters=scheduled_parameters
)