# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger()
logger.basicConfig = logging.basicConfig(level=logging.DEBUG)
import logictensornetworks_wrapper as ltnw
import tensorflow as tf
import numpy as np
import logictensornetworks as ltn
import matplotlib.pyplot as plt
from logictensornetworks import Not,And,Implies,Forall,Exists,Equiv
import pandas as pd
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:,.2f}'.format

def plt_heatmap(df):
    # display the result of a nxm pandas dataframe in a heatmap
    plt.pcolor(df)
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
    plt.colorbar()

ltn.LAYERS = 4
ltn.BIAS_factor = -1e-8
ltn.set_universal_aggreg("mean") # The truth value of forall x p(x) is
                                 # interpretable as the percentage of
                                 # element in the range of x that satisties p

embedding_size = 10 # each constant is interperted in a vector of this size

# create on constant for each individual a,b,... i,j, ...
for l in 'abcdefghijklmn':
    ltnw.constant(l, min_value=[0.] * embedding_size, max_value=[1.] * embedding_size)


# create variables that ranges on all the individuals, and the individuals of group 1 and group 2.

ltnw.variable("p",tf.concat(list(ltnw.CONSTANTS.values()),axis=0))
ltnw.variable("q",ltnw.VARIABLES["p"])
ltnw.variable("p1",tf.concat([ltnw.CONSTANTS[l] for l in "abcdefgh"],axis=0))
ltnw.variable("q1",ltnw.VARIABLES["p1"])
ltnw.variable("p2",tf.concat([ltnw.CONSTANTS[l] for l in "ijklmn"],axis=0))
ltnw.variable("q2",ltnw.VARIABLES["p2"])


# declare the predicates
ltnw.predicate('Friends', embedding_size * 2)
ltnw.predicate('Smokes', embedding_size)
ltnw.predicate('Cancer', embedding_size)


# add the assertional knowledge in our posses

ltnw.axiom("Friends(a,b)")
ltnw.axiom("~Friends(a,c)")
ltnw.axiom("~Friends(a,d)")
ltnw.axiom("Friends(a,e)")
ltnw.axiom("Friends(a,f)")
ltnw.axiom("Friends(a,g)")
ltnw.axiom("~Friends(a,h)")
ltnw.axiom("Friends(b,c)")
ltnw.axiom("~Friends(b,d)")
ltnw.axiom("~Friends(b,e)")
ltnw.axiom("~Friends(b,f)")
ltnw.axiom("~Friends(b,g)")
ltnw.axiom("~Friends(b,h)")
ltnw.axiom("Friends(c,d)")
ltnw.axiom("~Friends(c,e)")
ltnw.axiom("~Friends(c,f)")
ltnw.axiom("~Friends(c,g)")
ltnw.axiom("~Friends(c,h)")
ltnw.axiom("~Friends(d,e)")
ltnw.axiom("~Friends(d,f)")
ltnw.axiom("~Friends(d,g)")
ltnw.axiom("~Friends(d,h)")
ltnw.axiom("Friends(e,f)")
ltnw.axiom("~Friends(e,g)")
ltnw.axiom("~Friends(e,h)")
ltnw.axiom("~Friends(f,g)")
ltnw.axiom("~Friends(f,h)")
ltnw.axiom("Friends(g,h)")
ltnw.axiom("Friends(i,j)")
ltnw.axiom("~Friends(i,k)")
ltnw.axiom("~Friends(i,l)")
ltnw.axiom("~Friends(i,m)")
ltnw.axiom("~Friends(i,n)")
ltnw.axiom("~Friends(j,k)")
ltnw.axiom("~Friends(j,l)")
ltnw.axiom("Friends(j,m)")
ltnw.axiom("~Friends(j,n)")
ltnw.axiom("Friends(k,l)")
ltnw.axiom("~Friends(k,m)")
ltnw.axiom("~Friends(k,n)")
ltnw.axiom("~Friends(l,m)")
ltnw.axiom("~Friends(l,n)")
ltnw.axiom("Friends(m,n)")
ltnw.axiom("Smokes(a)")
ltnw.axiom("~Smokes(b)")
ltnw.axiom("~Smokes(c)")
ltnw.axiom("~Smokes(d)")
ltnw.axiom("Smokes(e)")
ltnw.axiom("Smokes(f)")
ltnw.axiom("Smokes(g)")
ltnw.axiom("~Smokes(h)")
ltnw.axiom("~Smokes(i)")
ltnw.axiom("Smokes(j)")
ltnw.axiom("~Smokes(k)")
ltnw.axiom("~Smokes(l)")
ltnw.axiom("~Smokes(m)")
ltnw.axiom("Smokes(n)")
ltnw.axiom("Cancer(a)")
ltnw.axiom("~Cancer(b)")
ltnw.axiom("~Cancer(c)")
ltnw.axiom("~Cancer(d)")
ltnw.axiom("Cancer(e)")
ltnw.axiom("~Cancer(f)")
ltnw.axiom("~Cancer(g)")
ltnw.axiom("~Cancer(h)")

# add general constraints on the friendship relation
ltnw.axiom("forall p: ~Friends(p,p)")
ltnw.axiom("forall p,q:Friends(p,q) % Friends(q,p)")

# Add constraints stating that the satisfaction level of the
# formula forall x smokes(x) -> cancer(x) in group 1 and group 2
# should be the same. Since in group 1 everything is known, this
# formula has the effect to transfer the knowledge about group 1
# to group 2.

ltnw.axiom("forall p1:Cancer(p1) % forall p2:Cancer(p2)")
ltnw.axiom("forall p1:(Smokes(p1) -> Cancer(p1)) % forall p2:(Smokes(p2) -> Cancer(p2))")
ltnw.axiom("forall p1:(Cancer(p1) -> Smokes(p1)) % forall p2:(Cancer(p2) -> Smokes(p2))")

# initialize knowledge base

ltnw.initialize_knowledgebase(
    optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01,decay=.9),
    formula_aggregator=lambda *x: 1./tf.reduce_mean(1./tf.concat(x,axis=0)))

# Train the KB
ltnw.train(max_epochs=10000)


# query the KB and display the results

df_smokes_cancer = pd.DataFrame(np.concatenate([ltnw.ask("Smokes(p)"),
                                                ltnw.ask("Cancer(p)")],axis=1),
                   columns=["Smokes","Cancer"],
                   index=list('abcdefghijklmn'))

df_friends_ah = pd.DataFrame(np.squeeze(ltnw.ask("Friends(p1,q1)")),
                   index=list('abcdefgh'),
                   columns=list('abcdefgh'))
df_friends_in = pd.DataFrame(np.squeeze(ltnw.ask("Friends(p2,q2)")),
                   index=list('ijklmn'),
                   columns=list('ijklmn'))
print(df_smokes_cancer)
print(df_friends_ah)
print(df_friends_in)
plt.figure(figsize=(15,4))
plt.subplot(131)
plt_heatmap(df_smokes_cancer)
plt.subplot(132)
plt_heatmap(df_friends_ah)
plt.subplot(133)
plt_heatmap(df_friends_in)
plt.show()
