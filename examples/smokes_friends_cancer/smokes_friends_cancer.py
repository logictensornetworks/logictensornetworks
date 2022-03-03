import tensorflow as tf
import numpy as np
import ltn
import argparse

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--csv-path',type=str,default="sfc_results.csv")
    parser.add_argument('--epochs',type=int,default=1000)
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args

args = parse_args()
EPOCHS = args['epochs']
csv_path = args['csv_path']

# Language

embedding_size = 10

g1 = {l:ltn.Constant(np.random.uniform(low=0.0,high=1.0,size=embedding_size),trainable=True) for l in 'abcdefgh'}
g2 = {l:ltn.Constant(np.random.uniform(low=0.0,high=1.0,size=embedding_size),trainable=True) for l in 'ijklmn'}
g = {**g1,**g2}

Smokes = ltn.Predicate.MLP([embedding_size],hidden_layer_sizes=(16,16))
Friends = ltn.Predicate.MLP([embedding_size,embedding_size],hidden_layer_sizes=(16,16))
Cancer = ltn.Predicate.MLP([embedding_size],hidden_layer_sizes=(16,16))

friends = [('a','b'),('a','e'),('a','f'),('a','g'),('b','c'),('c','d'),('e','f'),('g','h'),
           ('i','j'),('j','m'),('k','l'),('m','n')]
smokes = ['a','e','f','g','j','n']
cancer = ['a','e']

Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=2),semantics="forall")
Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(p=6),semantics="exists")

formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError())

# defining the theory
@tf.function
def axioms(p_exists):
    """
    NOTE: we update the embeddings at each step
        -> we should re-compute the variables.
    """
    p = ltn.Variable.from_constants("p",list(g.values()))
    q = ltn.Variable.from_constants("q",list(g.values()))
    axioms = []
    # Friends: knowledge incomplete in that
    #     Friend(x,y) with x<y may be known
    #     but Friend(y,x) may not be known

    axioms.append(formula_aggregator(
            [Friends([g[x],g[y]]) for (x,y) in friends]))
    axioms.append(formula_aggregator(
            [Not(Friends([g[x],g[y]])) for x in g1 for y in g1 if (x,y) not in friends and x<y ]+\
            [Not(Friends([g[x],g[y]])) for x in g2 for y in g2 if (x,y) not in friends and x<y ]))
    # Smokes: knowledge complete
    axioms.append(formula_aggregator(
            [Smokes(g[x]) for x in smokes]))
    axioms.append(formula_aggregator(
            [Not(Smokes(g[x])) for x in g if x not in smokes]))
    # Cancer: knowledge complete in g1 only
    axioms.append(formula_aggregator(
            [Cancer(g[x]) for x in cancer]))
    axioms.append(formula_aggregator(
            [Not(Cancer(g[x])) for x in g1 if x not in cancer]))
    # friendship is anti-reflexive
    axioms.append(Forall(p,Not(Friends([p,p])),p=5))
    # friendship is symmetric
    axioms.append(Forall((p,q),Implies(Friends([p,q]),Friends([q,p])),p=5))
    # everyone has a friend
    axioms.append(Forall(p,Exists(q,Friends([p,q]),p=p_exists)))
    # smoking propagates among friends
    axioms.append(Forall((p,q),Implies(And(Friends([p,q]),Smokes(p)),Smokes(q))))
    # smoking causes cancer + not smoking causes not cancer
    axioms.append(Forall(p,Implies(Smokes(p),Cancer(p))))
    axioms.append(Forall(p,Implies(Not(Smokes(p)),Not(Cancer(p)))))
    # computing sat_level
    sat_level = formula_aggregator(axioms).tensor
    return sat_level

# Initialize all layers and the static graph.
print("Initial sat level %.5f"%axioms(p_exists=tf.constant(6.)))

# # Training
# 
# Define the metrics

metrics_dict = {
    'train_sat': tf.keras.metrics.Mean(name='train_sat'),
    'test_phi1': tf.keras.metrics.Mean(name='test_phi1'),
    'test_phi2': tf.keras.metrics.Mean(name='test_phi2')
}

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
trainable_variables = \
        Smokes.trainable_variables \
        + Friends.trainable_variables \
        + Cancer.trainable_variables \
        + ltn.as_tensors(list(g.values()))

@tf.function
def train_step(p_exists):
    # sat and update
    with tf.GradientTape() as tape:
        sat = axioms(p_exists)
        loss = 1.-sat
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    metrics_dict['train_sat'](sat)

@tf.function
def sat_phi1():
    p = ltn.Variable.from_constants("p",list(g.values()))
    q = ltn.Variable.from_constants("q",list(g.values()))
    phi1 = Forall(p,Implies(Cancer(p),Smokes(p)),p=5)
    return phi1.tensor
@tf.function
def sat_phi2():
    p = ltn.Variable.from_constants("p",list(g.values()))
    q = ltn.Variable.from_constants("q",list(g.values()))
    phi2 = Forall((p,q), Implies(Or(Cancer(p),Cancer(q)),Friends([p,q])),p=5)
    return phi2.tensor

@tf.function
def test_step():
    # sat
    metrics_dict['test_phi1'](sat_phi1())
    metrics_dict['test_phi2'](sat_phi2())

track_metrics=20
template = "Epoch {}"
for metrics_label in metrics_dict.keys():
    template += ", %s: {:.4f}" % metrics_label
if csv_path is not None:
    csv_file = open(csv_path,"w+")
    headers = ",".join(["Epoch"]+list(metrics_dict.keys()))
    csv_template = ",".join(["{}" for _ in range(len(metrics_dict)+1)])
    csv_file.write(headers+"\n")

for epoch in range(EPOCHS):
    for metrics in metrics_dict.values():
        metrics.reset_states()

    if 0 <= epoch < 200:
        p_exists = tf.constant(1.)
    else:
        p_exists = tf.constant(6.)

    train_step(p_exists=p_exists)
    test_step()

    metrics_results = [metrics.result() for metrics in metrics_dict.values()]
    if epoch%track_metrics == 0:
        print(template.format(epoch,*metrics_results))
    if csv_path is not None:
        csv_file.write(csv_template.format(epoch,*metrics_results)+"\n")
        csv_file.flush()
if csv_path is not None:
    csv_file.close()