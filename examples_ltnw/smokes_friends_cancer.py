# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger()
logger.basicConfig = logging.basicConfig(level=logging.DEBUG)

import tensorflow as tf
import logictensornetworks as ltn
import logictensornetworks_wrapper as ltnw

ltn.LAYERS = 4
ltn.BIAS_factor = 1e-3

size = 10

g1='abcdefgh'
g2='ijklmn'
g=g1+g2
for l in 'abcdefghijklmn':
    ltnw.constant(l,min_value=[0.]*size,max_value=[1.]*size)

friends = [('a','b'),('a','e'),('a','f'),('a','g'),('b','c'),('c','d'),('e','f'),('g','h'),
           ('i','j'),('j','m'),('k','l'),('m','n')]
smokes = ['a','e','f','g','j','n']
cancer = ['a','e']

p = ltnw.variable("p",tf.concat(list(ltnw.CONSTANTS.values()),axis=0))
q = ltnw.variable("q",tf.concat(list(ltnw.CONSTANTS.values()),axis=0))

Friends = ltnw.predicate('Friends',size*2)
Smokes = ltnw.predicate('Smokes',size)
Cancer = ltnw.predicate('Cancer',size)

[ltnw.formula("Friends(%s,%s)" %(x,y)) for (x,y) in friends]
[ltnw.formula("~Friends(%s,%s)" %(x,y)) for x in g1 for y in g1 if (x,y) not in friends and x < y]
[ltnw.formula("~Friends(%s,%s)" %(x,y)) for x in g2 for y in g2 if (x,y) not in friends and x < y]

[ltnw.formula("Smokes(%s)" % x) for x in smokes]
[ltnw.formula("~Smokes(%s)" % x) for x in g if x not in smokes]

[ltnw.formula("Cancer(%s)" % x) for x in cancer]
[ltnw.formula("~Cancer(%s)" % x) for x in g if x not in cancer]


ltnw.formula("forall p: ~Friends(p,p)")
ltnw.formula("forall p,q:Friends(p,q) -> Friends(q,p)")
ltnw.formula("forall p: exists q: Friends(p,q)")
ltnw.formula("forall p,q:Friends(p,q) -> (Smokes(p)->Smokes(q))")
ltnw.formula("forall p: Smokes(p) -> Cancer(p)")
    
ltnw.SESSION=tf.Session() 
loss = tf.reduce_mean(tf.concat(list(ltnw.FORMULAS.values()),axis=0))
opt = tf.train.RMSPropOptimizer(learning_rate=.01,decay=.9)
optimize = opt.minimize(-(loss+ltn.BIAS))

ltnw.SESSION.run(tf.global_variables_initializer())
for i in range(10000):
        ltnw.SESSION.run(optimize)
        if i % 100 == 0:
            print(i,"=====>",ltnw.SESSION.run(loss),ltnw.SESSION.run(ltn.BIAS))

for x in g:
    print(x," = ",ltnw.ask(x).squeeze())
    print("Cancer("+x+"): %.2f" % ltnw.ask("Cancer(%s)" % x).squeeze())
    print("Smokes("+x+"): %.2f" % ltnw.ask("Smokes(%s)" % x).squeeze())

    for y in g:
        print("Friends("+x+","+y+"): %.2f" % ltnw.ask("Friends(%s,%s)" % (x,y)).squeeze())

for formula in ["forall p: ~Friends(p,p)",
                "forall p,q: Friends(p,q) -> Friends(q,p)",
                "forall p: exists q: Friends(p,q)",
                "forall p,q: Friends(p,q) -> (Smokes(p)->Smokes(q))"]:
    print(formula,": %.2f" % ltnw.ask(formula).squeeze())
