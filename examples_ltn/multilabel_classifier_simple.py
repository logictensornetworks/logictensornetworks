import logictensornetworks as ltn
from logictensornetworks import And,Not,Equiv,Forall,Implies
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# loading data

data = np.random.uniform([-1,-1],[1,1],(500,2),).astype(np.float32)

# defining the language

x = ltn.variable("x",data)
y = ltn.variable("y",data)

a = ltn.constant("a",[0.5,0.5])
b = ltn.constant("x",[-0.5,-0.5])

A = ltn.predicate("A",2)
B = ltn.predicate("B",2)
T = And(A(a),B(b),Not(A(b)),Forall(x,Implies(A(x),B(x))))

# start a tensorflow session

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(-T)

# optimize the satisfiability of T

sess.run(init)
sat_level = sess.run(T)
while sat_level == 0.0:
    print("reinitializing parameters")
    sess.run(init)
    sat_level = sess.run(T)
for i in range(10000):
    if i % 100 == 0:
        print(i,"sat level -----> ",sess.run(T))
    sess.run(opt)

# show results:

Ax = sess.run(tf.squeeze(A(x)))
Aa = sess.run(A(a))
Bx = sess.run(tf.squeeze(B(x)))
Bb = sess.run(B(b))
x0,x1 = sess.run(tf.transpose(x))
a0,a1 = sess.run(tf.expand_dims(a,1))
b0,b1 = sess.run(tf.expand_dims(b,1))

fig = plt.figure(figsize=(13, 5))
fig.add_subplot(1,2,1)
plt.title("A(x)")
plt.xlabel("x[0]")
plt.ylabel("x[1]")
plt.scatter(np.concatenate([x0,a0]),
            np.concatenate([x1,a1]),
            c=np.concatenate([Ax,Aa]))
plt.annotate("a",(a0,a1))
plt.colorbar()

fig.add_subplot(1,2,2)
plt.title("B(x)")
plt.xlabel("x[0]")
plt.ylabel("x[1]")
plt.scatter(np.concatenate([x0,b0]),
            np.concatenate([x1,b1]),
            c=np.concatenate([Bx,Bb]))
plt.annotate("b",(b0,b1))
plt.colorbar()
plt.show()

