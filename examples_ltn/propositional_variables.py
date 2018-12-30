import logictensornetworks as ltn
from logictensornetworks import Implies,And,Not,Forall,Exists
import tensorflow as tf
import numpy as np
a = ltn.proposition("a",value=.2)
b = ltn.proposition("b")
c = ltn.proposition("c")
w1 = ltn.proposition("w1",value=.3)
w2 = ltn.proposition("w2",value=.9)

x = ltn.variable("x",np.array([[1,2],[3,4],[5,6]]).astype(np.float32))
P = ltn.predicate("P",2)

formula = And(Implies(And(Forall(x,P(x)),a,b),Not(c)),c)
w1_formula1 = Implies(w1,Forall(x,P(x)))
w2_formula2 = Implies(w2,Exists(x,P(x)))

sat = tf.train.GradientDescentOptimizer(0.01).minimize(-tf.concat([formula,w1_formula1,w2_formula2],axis=0))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        sess.run(sat)
        if i % 10 == 0:
            print(sess.run(formula))
    print(sess.run([a,b,c]))
    print(sess.run(And(a,P(x))))