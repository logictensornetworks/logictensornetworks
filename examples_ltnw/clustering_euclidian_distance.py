# -*- coding: utf-8 -*-
import logging; logging.basicConfig(level=logging.INFO)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import logictensornetworks_wrapper as ltnw
ltnw.ltn.set_universal_aggreg("min")
ltnw.ltn.set_existential_aggregator("max")
ltnw.ltn.set_tnorm("prod")
ltnw.ltn.LAYERS = 4
import logictensornetworks_library as ltnl

# generate data
nr_of_clusters = 2
nr_of_points_x_cluster = 50
clusters=[]
for i in range(nr_of_clusters):
    mean = np.random.uniform([-1,-1],[1,1],2).astype(np.float32)
    cov = np.array([[.001,0],[0,.001]])
    clusters.append(np.random.multivariate_normal(mean=mean,cov=cov,size=nr_of_points_x_cluster).astype(np.float32) )
data  = np.concatenate(clusters)

# define the language
ltnw.variable("?x",data)
ltnw.variable("?y",data)
ltnw.predicate("close",2,ltnl.equal_euclidian)
[ltnw.predicate("C_"+str(i),2) for i in range(nr_of_clusters)]

## define the theory
print("defining the theory T")
ltnw.formula("forall ?x: %s" % "|".join(["C_%s(?x)" % i for i in range(nr_of_clusters)]))
for i in range(nr_of_clusters):
    ltnw.formula("exists ?x: C_%s(?x)" % i)
    ltnw.formula("forall ?x,?y: (C_%s(?x) & close(?x,?y)) -> C_%s(?y)" % (i,i))
    ltnw.formula("forall ?x,?y: (C_%s(?x) & ~close(?x,?y)) -> (%s)" % (i,"|".join(["C_%s(?y)" % j for j in range(nr_of_clusters) if i!=j])))
    
    for j in range(i+1,nr_of_clusters):
        ltnw.formula("forall ?x: ~(C_%s(?x) & C_%s(?x))" % (i,j))
print("\n".join(sorted(ltnw.FORMULAS.keys())))

## initialize and optimize
ltnw.initialize_knowledgebase(optimizer=tf.train.RMSPropOptimizer(learning_rate=0.1,decay=.9),
                              initial_sat_level_threshold=.0)
ltnw.train(max_epochs=1000)

## visualize results
nr_of_clusters=len(clusters)
prC = [ltnw.ask("C_%s(?x)" % i) for i in range(nr_of_clusters)]
n = 2
m = (nr_of_clusters + 1) // n + 1

fig = plt.figure(figsize=(3 * 3, m * 3))

fig.add_subplot(m, n, 1)
plt.title("groundtruth")
for c in clusters:
    plt.scatter(c[:, 0], c[:, 1])
data=np.concatenate(clusters)
x0 = data[:, 0]
x1 = data[:, 1]
for i in range(nr_of_clusters):
    fig.add_subplot(m, n, i + 2)
    plt.title("C" + str(i) + "(x)")
    plt.scatter(x0, x1, c=prC[i].T[0])
    plt.scatter(x0[:2], x1[:2], s=0, c=[0, 1])
    plt.colorbar()
plt.show()
