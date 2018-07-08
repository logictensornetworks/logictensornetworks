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
data_not_A=data[np.where(np.sum(np.square(data-[.5,.5]),axis=1)>=.09)]

ltnw.variable("?data_A",data_A)
ltnw.variable("?data_not_A",data_not_A)
ltnw.variable("?data",data)

ltnw.predicate("A",2)

ltnw.formula("forall ?data_A: A(?data_A)")
ltnw.formula("forall ?data_not_A: ~A(?data_not_A)")

ltnw.initialize_knowledgebase(initial_sat_level_threshold=.1)
sat_level=ltnw.train(track_sat_levels=True,sat_level_epsilon=.99)
    
plt.figure(figsize=(12,8))
result=ltnw.ask("A(?data)")
plt.subplot(2,2,1)
plt.scatter(data[:,0],data[:,1],c=result.squeeze())
plt.title("A(x) - training data")

result=ltnw.ask("~A(?data)")
plt.subplot(2,2,2)
plt.scatter(data[:,0],data[:,1],c=result.squeeze())
plt.title("~A(x) - training data")

data_test=np.random.uniform([0,0],[1.,1.],(500,2)).astype(np.float32)
ltnw.variable("?data_test",data_test)
result=ltnw.ask("A(?data_test)")
plt.subplot(2,2,3)
plt.title("A(x) - test")
plt.scatter(data_test[:,0],data_test[:,1],c=result.squeeze())
plt.title("A(x) - test data")

result=ltnw.ask("~A(?data_test)")
plt.subplot(2,2,4)
plt.scatter(data_test[:,0],data_test[:,1],c=result.squeeze())
plt.title("~A(x) - test data")
plt.show()

ltnw.constant("a",[0.25,.5])
ltnw.constant("b",[1.,1.])
print("a is in A: %s" % ltnw.ask("A(a)"))
print("b is in A: %s" % ltnw.ask("A(b)"))
