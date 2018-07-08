import logictensornetworks as ltn
ltn.set_universal_aggreg("min")
ltn.set_existential_aggregator("max")
ltn.set_tnorm("prod")
ltn.LAYERS = 4

from logictensornetworks import And,Not,Or,Forall,Exists,Implies,Equiv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def show_result():
    x0 = data[:, 0]
    x1 = data[:, 1]
    prC = sess.run([C[i](x) for i in clst_ids])
    n = 2
    m = (nr_of_clusters + 1) // n + 1

    fig = plt.figure(figsize=(3 * 3, m * 3))

    fig.add_subplot(m, n, 1)
    plt.title("groundtruth")
    for \
            i in clst_ids:
        plt.scatter(cluster[i][:, 0], cluster[i][:, 1])
    for i in clst_ids:
        fig.add_subplot(m, n, i + 2)
        plt.title("C" + str(i) + "(x)")
        plt.scatter(x0, x1, c=prC[i].T[0])
        plt.scatter(x0[:2], x1[:2], s=0, c=[0, 1])
        plt.colorbar()
    plt.show()



# loading data

nr_of_clusters = 4
nr_of_points_x_cluster = 50
clst_ids = range(nr_of_clusters)

mean = np.random.uniform([-1,-1],[1,1],(nr_of_clusters,2)).astype(np.float32)

cov = np.array([[[.01,0],[0,.01]]]*nr_of_clusters)

cluster = {}
for i in clst_ids:
    cluster[i] = np.random.multivariate_normal(mean=mean[i],cov=cov[i],size=nr_of_points_x_cluster).astype(np.float32)

data  = np.concatenate([cluster[i] for i in clst_ids])
closed_data = np.array([np.concatenate([data[i],data[j]])
                             for i in range(len(data))
                             for j in range(i,len(data))
                             if np.sum(np.square(data[i]-data[j])) < np.square(.5)])

closed_data = closed_data[np.random.random_integers(0,len(data),1000)]
distant_data = np.array([np.concatenate([data[i],data[j]])
                             for i in range(len(data))
                             for j in range(len(data))
                             if np.sum(np.square(data[i]-data[j])) > np.square(1.)])

# defining the language

x = ltn.variable("x",data)
y = ltn.variable("y",data)
closed_x_y = ltn.variable("closed_x_y",closed_data)
distant_x_y = ltn.variable("distant_x_y",distant_data)


C = {i:ltn.predicate("C_"+str(i),x) for i in clst_ids}

first = ltn.function("first",closed_x_y,fun_definition=lambda d:d[:,:2])
second = ltn.function("second",closed_x_y,fun_definition=lambda d:d[:,2:])

print("defining the theory T")
T = tf.reduce_mean(tf.concat(
        [Forall(x,Or(*[C[i](x) for i in clst_ids]))] +
        [Exists(x,C[i](x)) for i in clst_ids] +
        [Forall(closed_x_y,Equiv(C[i](first(closed_x_y)),C[i](second(closed_x_y)))) for i in clst_ids] +
        [Forall(distant_x_y, Not(And(C[i](first(distant_x_y)),(C[i](second(distant_x_y)))))) for i in clst_ids] +
        [Forall(x,Not(And(C[i](x),C[j](x)))) for i in clst_ids for j in clst_ids if i != j]
    ,axis=0))

# setting the learnign parameters

# Lambda = 1e-3
# reg_term = Lambda*tf.reduce_sum(tf.stack([tf.norm(p) for i in clst_ids for p in C[i].pars]))
opt = tf.train.RMSPropOptimizer(learning_rate=0.01,decay=.9).minimize(-T)
init = tf.global_variables_initializer()

# Start Learning by optimizing the satisfiability of T:

sess = tf.Session()
sess.run(init)
sat_level = sess.run(tf.squeeze(T))
show_result()
while sat_level < .5:
    print("reinitializing parameters")
    sess.run(init)
    sess.run(opt)
    sat_level = sess.run(tf.squeeze(T))
for i in range(10000):
    sess.run(opt)
    if i % 100 == 0:
        sat_level = sess.run(tf.squeeze(T))
        print(i,"---->",sat_level)
        if sat_level > .999:
            break

show_result()

