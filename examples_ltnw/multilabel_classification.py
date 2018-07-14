# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger()
logger.basicConfig = logging.basicConfig(level=logging.DEBUG)
import numpy as np
import matplotlib.pyplot as plt

import logictensornetworks_wrapper as ltnw

nr_samples=500

data=np.random.uniform([0,0],[1.,1.],(nr_samples,2)).astype(np.float32)
data_A=data[np.where(np.sum(np.square(data-[.5,.5]),axis=1)<.09)]
data_B=data[np.where(np.sum(np.square(data-[.5,.5]),axis=1)>=.09)]

ltnw.variable("?data",data)
ltnw.variable("?data_A",data_A)
ltnw.variable("?data_B",data_B)

ltnw.predicate("A",2)
ltnw.predicate("B",2)

ltnw.formula("forall ?data_A: A(?data_A)")
ltnw.formula("forall ?data_B: ~A(?data_B)")

ltnw.formula("forall ?data_B: B(?data_B)")
ltnw.formula("forall ?data_A: ~B(?data_A)")

ltnw.formula("forall ?data: A(?data) -> ~B(?data)")
ltnw.formula("forall ?data: B(?data) -> ~A(?data)")

ltnw.initialize_knowledgebase(initial_sat_level_threshold=.1)
sat_level=ltnw.train(track_sat_levels=1000,sat_level_epsilon=.99,max_epochs=20000)

result=ltnw.ask("A(?data)")
plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
plt.title("A(x) - training")
plt.scatter(data[:,0],data[:,1],c=result.squeeze())
plt.colorbar()

plt.subplot(2,2,2)
result=ltnw.ask("B(?data)")
plt.title("B(x) - training")
plt.scatter(data[:,0],data[:,1],c=result.squeeze())
plt.colorbar()

data_test=np.random.uniform([0,0],[1.,1.],(nr_samples,2)).astype(np.float32)
ltnw.variable("?data_test",data_test)
result=ltnw.ask("A(?data_test)")
plt.subplot(2,2,3)
plt.title("A(x) - test")
plt.scatter(data_test[:,0],data_test[:,1],c=result.squeeze())
plt.colorbar()

result=ltnw.ask("B(?data_test)")
plt.subplot(2,2,4)
plt.title("B(x) -test")
plt.scatter(data_test[:,0],data_test[:,1],c=result.squeeze())
plt.colorbar()

plt.show()

ltnw.constant("a",[0.5,.5])
ltnw.constant("b",[0.75,.75])
print("a is in A: %s" % ltnw.ask("A(a)"))
print("b is in A: %s" % ltnw.ask("A(b)"))
print("a is in B: %s" % ltnw.ask("B(a)"))
print("b is in B: %s" % ltnw.ask("B(b)"))
