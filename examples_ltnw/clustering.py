import logging
logger = logging.getLogger()
logger.basicConfig = logging.basicConfig(level=logging.DEBUG)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import logictensornetworks_wrapper as ltnw
ltnw.ltn.set_universal_aggreg("min")
ltnw.ltn.set_existential_aggregator("max")
ltnw.ltn.set_tnorm("prod")
ltnw.ltn.LAYERS = 4

def show_result(clusters):
    nr_of_clusters=len(clusters)
    prC = [ltnw.ask("C_%s(x)" % i) for i in range(nr_of_clusters)]
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

# generating data

nr_of_clusters = 4
nr_of_points_x_cluster = 50

clusters=[]
for i in range(nr_of_clusters):
    mean = np.random.uniform([-1,-1],[1,1],2).astype(np.float32)
    cov = np.array([[.01,0],[0,.01]])
    clusters.append(np.random.multivariate_normal(mean=mean,cov=cov,size=nr_of_points_x_cluster).astype(np.float32) )

data  = np.concatenate(clusters)
close_data = np.array([np.concatenate([data[i],data[j]])
                             for i in range(len(data))
                             for j in range(i,len(data))
                             if np.sum(np.square(data[i]-data[j])) < np.square(.5)])

close_data = close_data[np.random.random_integers(0,len(data),1000)]
distant_data = np.array([np.concatenate([data[i],data[j]])
                             for i in range(len(data))
                             for j in range(len(data))
                             if np.sum(np.square(data[i]-data[j])) > np.square(1.)])

# defining the language
ltnw.variable("x",data)
ltnw.variable("y",data)
ltnw.variable("close_x_y",close_data)
ltnw.variable("distant_x_y",distant_data)
[ltnw.predicate("C_"+str(i),2) for i in range(nr_of_clusters)]

ltnw.function("first",2,fun_definition=lambda d:d[:,:2])
ltnw.function("second",2,fun_definition=lambda d:d[:,2:])

print("defining the theory T")
ltnw.formula("forall x: %s" % "|".join(["C_%s(x)" % i for i in range(nr_of_clusters)]))
for i in range(nr_of_clusters):
    ltnw.formula("exists x: C_%s(x)" % i)
    ltnw.formula("forall close_x_y: C_%s(first(close_x_y)) %% C_%s(second(close_x_y))" % (i,i))
    ltnw.formula("forall distant_x_y: C_%s(first(distant_x_y)) %% C_%s(second(distant_x_y))" % (i,i))
    for j in range(i+1,nr_of_clusters):
        ltnw.formula("forall x: ~(C_%s(x) & C_%s(x))" % (i,j))
print("%s" % "\n".join(ltnw.FORMULAS.keys()))

ltnw.initialize_knowledgebase(optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01,decay=.9),
                              initial_sat_level_threshold=.5)

ltnw.train(max_iterations=100)

show_result(clusters)
