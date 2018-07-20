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

ltnw.formula("Friends(a,b)")
ltnw.formula("~Friends(a,c)")
ltnw.formula("~Friends(a,d)")
ltnw.formula("Friends(a,e)")
ltnw.formula("Friends(a,f)")
ltnw.formula("Friends(a,g)")
ltnw.formula("~Friends(a,h)")
ltnw.formula("Friends(b,c)")
ltnw.formula("~Friends(b,d)")
ltnw.formula("~Friends(b,e)")
ltnw.formula("~Friends(b,f)")
ltnw.formula("~Friends(b,g)")
ltnw.formula("~Friends(b,h)")
ltnw.formula("Friends(c,d)")
ltnw.formula("~Friends(c,e)")
ltnw.formula("~Friends(c,f)")
ltnw.formula("~Friends(c,g)")
ltnw.formula("~Friends(c,h)")
ltnw.formula("~Friends(d,e)")
ltnw.formula("~Friends(d,f)")
ltnw.formula("~Friends(d,g)")
ltnw.formula("~Friends(d,h)")
ltnw.formula("Friends(e,f)")
ltnw.formula("~Friends(e,g)")
ltnw.formula("~Friends(e,h)")
ltnw.formula("~Friends(f,g)")
ltnw.formula("~Friends(f,h)")
ltnw.formula("Friends(g,h)")
ltnw.formula("Friends(i,j)")
ltnw.formula("~Friends(i,k)")
ltnw.formula("~Friends(i,l)")
ltnw.formula("~Friends(i,m)")
ltnw.formula("~Friends(i,n)")
ltnw.formula("~Friends(j,k)")
ltnw.formula("~Friends(j,l)")
ltnw.formula("Friends(j,m)")
ltnw.formula("~Friends(j,n)")
ltnw.formula("Friends(k,l)")
ltnw.formula("~Friends(k,m)")
ltnw.formula("~Friends(k,n)")
ltnw.formula("~Friends(l,m)")
ltnw.formula("~Friends(l,n)")
ltnw.formula("Friends(m,n)")
ltnw.formula("Smokes(a)")
ltnw.formula("~Smokes(b)")
ltnw.formula("~Smokes(c)")
ltnw.formula("~Smokes(d)")
ltnw.formula("Smokes(e)")
ltnw.formula("Smokes(f)")
ltnw.formula("Smokes(g)")
ltnw.formula("~Smokes(h)")
ltnw.formula("~Smokes(i)")
ltnw.formula("Smokes(j)")
ltnw.formula("~Smokes(k)")
ltnw.formula("~Smokes(l)")
ltnw.formula("~Smokes(m)")
ltnw.formula("Smokes(n)")
ltnw.formula("Cancer(a)")
ltnw.formula("~Cancer(b)")
ltnw.formula("~Cancer(c)")
ltnw.formula("~Cancer(d)")
ltnw.formula("Cancer(e)")
ltnw.formula("~Cancer(f)")
ltnw.formula("~Cancer(g)")
ltnw.formula("~Cancer(h)")

# add general constraints on the friendship relation
ltnw.formula("forall p: ~Friends(p,p)")
ltnw.formula("forall p,q:Friends(p,q) % Friends(q,p)")

# Add constraints stating that the satisfaction level of the
# formula forall x smokes(x) -> cancer(x) in group 1 and group 2
# should be the same. Since in group 1 everything is known, this
# formula has the effect to transfer the knowledge about group 1
# to group 2.

ltnw.formula("forall p1:Cancer(p1) % forall p2:Cancer(p2)")
ltnw.formula("forall p1:(Smokes(p1) -> Cancer(p1)) % forall p2:(Smokes(p2) -> Cancer(p2))")
ltnw.formula("forall p1:(Cancer(p1) -> Smokes(p1)) % forall p2:(Cancer(p2) -> Smokes(p2))")

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
