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
    plt.pcolor(df)
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
    plt.colorbar()



pd.set_option('precision',2)

ltn.LAYERS = 4
ltn.BIAS_factor = 1e-7
ltn.set_universal_aggreg("mean")

size = 20
g1 = {l:ltn.constant(l,min_value=[0.]*size,max_value=[1.]*size) for l in 'abcdefgh'}
g2 = {l:ltn.constant(l,min_value=[0.]*size,max_value=[1.]*size) for l in 'ijklmn'}
g = {**g1,**g2}


friends = [('a','b'),('a','e'),('a','f'),('a','g'),('b','c'),('c','d'),('e','f'),('g','h'),
           ('i','j'),('j','m'),('k','l'),('m','n')]
smokes = ['a','e','f','g','j','n']
cancer = ['a','e']


p = ltn.variable("p",tf.concat(list(g.values()),axis=0))
q = ltn.variable("q",tf.concat(list(g.values()),axis=0))

p1 = ltn.variable("p1",tf.concat(list(g1.values()),axis=0))
q1 = ltn.variable("q1",tf.concat(list(g1.values()),axis=0))

p2 = ltn.variable("p2",tf.concat(list(g2.values()),axis=0))
q2 = ltn.variable("q2",tf.concat(list(g2.values()),axis=0))

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
         Forall((p,q),Equiv(Friends(p,q),Friends(q,p))),
         Equiv(Forall(p1,Implies(Smokes(p1),Cancer(p1))),
               Forall(p2,Implies(Smokes(p2),Cancer(p2)))),
         Equiv(Forall(p1,Implies(Cancer(p1),Smokes(p1))),
               Forall(p2,Implies(Cancer(p2),Smokes(p2))))]

loss = 1.0/tf.reduce_mean(1/tf.concat(facts,axis=0))
opt = tf.train.RMSPropOptimizer(learning_rate=.01,decay=.9)
optimize = opt.minimize(-(loss+ltn.BIAS))

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
                sess.run(optimize)
                if i % 100 == 0:
                    print(i,"=====>",sess.run(loss),sess.run(ltn.BIAS))
        df_smokes_cancer = pd.DataFrame(sess.run(tf.concat([Smokes(p),Cancer(p)],axis=1)),
                           columns=["Smokes","Cancer"],
                           index=list('abcdefghijklmn'))
        pred_friends = sess.run(tf.squeeze(Friends(p,q)))
        df_friends_ah = pd.DataFrame(pred_friends[:8,:8],
                           index=list('abcdefgh'),
                           columns=list('abcdefgh'))
        df_friends_in = pd.DataFrame(pred_friends[8:,8:],
                           index=list('ijklmn'),
                           columns=list('ijklmn'))
        plt.figure(figsize=(17,5))
        plt.subplot(131)
        plt_heatmap(df_smokes_cancer)
        plt.subplot(132)
        plt_heatmap(df_friends_ah)
        plt.subplot(133)
        plt_heatmap(df_friends_in)
        plt.show()

        print("forall x ~Friends(x,x)",
              sess.run(Forall(p,Not(Friends(p,p)))))
        print("Forall x Smokes(x) -> Cancer(x)",
              sess.run(Forall(p,Implies(Smokes(p),Cancer(p)))))
        print("forall x y Friends(x,y) -> Friends(y,x)",
              sess.run(Forall((p,q),Implies(Friends(p,q),Friends(q,p)))))
        print("forall x Exists y (Friends(x,y)",
              sess.run(Forall(p,Exists(q,Friends(p,q)))))
        print("Forall x,y Friends(x,y) -> (Smokes(x)->Smokes(y))",
              sess.run(Forall((p,q),Implies(Friends(p,q),Implies(Smokes(p),Smokes(q))))))
        print("Forall x: smokes(x) -> forall y: friend(x,y) -> smokes(y))",
              sess.run(Forall(p,Implies(Smokes(p),
                                        Forall(q,Implies(Friends(p,q),
                                                         Smokes(q)))))))



