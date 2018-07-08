import tensorflow as tf
import logictensornetworks as ltn
from logictensornetworks import Not,And,Implies,Forall,Exists

ltn.LAYERS = 4
ltn.BIAS_factor = 1e-3

size = 10
g1 = {l:ltn.constant(l,min_value=[0.]*size,max_value=[1.]*size) for l in 'abcdefgh'}
g2 = {l:ltn.constant(l,min_value=[0.]*size,max_value=[1.]*size) for l in 'ijklmn'}
g = {**g1,**g2}


friends = [('a','b'),('a','e'),('a','f'),('a','g'),('b','c'),('c','d'),('e','f'),('g','h'),
           ('i','j'),('j','m'),('k','l'),('m','n')]
smokes = ['a','e','f','g','j','n']
cancer = ['a','e']


p = ltn.variable("p",tf.concat(list(g.values()),axis=0))
q = ltn.variable("q",tf.concat(list(g.values()),axis=0))


Friends = ltn.predicate('Friends',size*2)
Smokes = ltn.predicate('Smokes',size)
Cancer = ltn.predicate('Cancer',size)


facts = [Friends(g[x],g[y]) for (x,y) in friends]+\
        [Not(Friends(g[x],g[y])) for x in g1 for y in g1
                                 if (x,y) not in friends and x < y]+\
        [Not(Friends(g[x],g[y])) for x in g2 for y in g2
                                 if (x, y) not in friends and x < y] +\
        [Smokes(g[x]) for x in smokes]+\
        [Not(Smokes(g[x])) for x in g if x not in smokes]+\
        [Cancer(g[x]) for x in cancer]+\
        [Not(Cancer(g[x])) for x in g1 if x not in cancer] +\
        [Forall(p,Not(Friends(p,p))),
         Forall((p,q),Implies(Friends(p,q),Friends(q,p))),
         Forall(p,Exists(q,Friends(p,q))),
         Forall((p,q),Implies(Friends(p,q),Implies(Smokes(p),Smokes(q)))),
         Forall(p,Implies(Smokes(p),Cancer(p)))]


loss = tf.reduce_mean(tf.concat(facts,axis=0))
opt = tf.train.RMSPropOptimizer(learning_rate=.01,decay=.9)
optimize = opt.minimize(-(loss+ltn.BIAS))
with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
                sess.run(optimize)
                if i % 100 == 0:
                    print(i,"=====>",sess.run(loss),sess.run(ltn.BIAS))
        for gg in [g1,g2]:
            for x in gg:
                print(x," = ",sess.run(g[x]))
                print("Cancer("+x+")",sess.run(Cancer(g[x])))
                print("Smokes("+x+")",sess.run(Smokes(g[x])))
                for y in gg:
                    print("Friends("+x+","+y+")",sess.run(Friends(g[x],g[y])))
        print("forall x ~Friends(x,x)",
              sess.run(Forall(p,Not(Friends(p,p)))))
        print("Forall x Smokes(x) -> Cancer(x)",
              sess.run(Forall(p,Implies(Smokes(p),Cancer(p)))))
        print("forall x y Friends(x,y) -> Friends(y,x)",
              sess.run(Forall((p,q),Implies(Friends(p,q),Friends(q,p)))))
        print("forall x Exists y (Friends(x,y)",
              sess.run(Forall(p,Exists(q,Friends(p,q)))))
        print("Forall x,y Friends(x,y) -> (Smokes(x)->Smokes(y)",
              sess.run(Forall((p,q),Implies(Friends(p,q),Implies(Smokes(p),Smokes(q))))))




