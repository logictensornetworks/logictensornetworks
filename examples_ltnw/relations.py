# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger()
logger.basicConfig = logging.basicConfig(level=logging.DEBUG)
import numpy as np
import matplotlib.pyplot as plt

import logictensornetworks_wrapper as ltnw

nr_samples=5
epochs=30000

data_A=np.random.uniform([0,0],[.25,1.],(nr_samples,2)).astype(np.float32)
data_B=np.random.uniform([.75,0],[1.,1.],(nr_samples,2)).astype(np.float32)
data=np.concatenate([data_A,data_B])

ltnw.variable("?data_A",data_A)
ltnw.variable("?data_A_2",data_A)
ltnw.variable("?data_B",data_B)
ltnw.variable("?data_B_2",data_B)
ltnw.variable("?data",data)
ltnw.variable("?data_1",data)
ltnw.variable("?data_2",data)

ltnw.predicate("A",2)
ltnw.predicate("B",2)

ltnw.formula("forall ?data_A: A(?data_A)")
ltnw.formula("forall ?data_B: ~A(?data_B)")

ltnw.formula("forall ?data_B: B(?data_B)")
ltnw.formula("forall ?data_A: ~B(?data_A)")

#ltnw.formula("forall ?data: A(?data) -> ~B(?data)")
#ltnw.formula("forall ?data: B(?data) -> ~A(?data)")

ltnw.predicate("R_A_A",4)
ltnw.predicate("R_B_B",4) 
ltnw.predicate("R_A_B",4)


ltnw.formula("forall ?data, ?data_2: (A(?data) & A(?data_2)) -> R_A_A(?data,?data_2)")
ltnw.formula("forall ?data, ?data_2: R_A_A(?data,?data_2) -> (A(?data) & A(?data_2))")

ltnw.formula("forall ?data, ?data_2: (B(?data) & B(?data_2)) -> R_B_B(?data,?data_2)")
ltnw.formula("forall ?data, ?data_2: R_B_B(?data,?data_2) -> (B(?data) & B(?data_2))")

ltnw.formula("forall ?data, ?data_2: (A(?data) & B(?data_2)) -> R_A_B(?data,?data_2)")
ltnw.formula("forall ?data, ?data_2: R_A_B(?data,?data_2) -> (A(?data) & B(?data_2))")

ltnw.initialize_knowledgebase(initial_sat_level_threshold=.1)
sat_level=ltnw.train(track_sat_levels=True,sat_level_epsilon=.99,max_iterations=epochs)

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.title("data A/B")
plt.scatter(data_A[:,0],data_A[:,1],c="red",alpha=1.,label="A")
plt.scatter(data_B[:,0],data_B[:,1],c="blue",alpha=1.,label="B")
plt.legend()

idx=2
for pred in ["R_A_A","R_A_B","R_B_B"]:
    result_A_A=ltnw.ask("%s(?data_A,?data_A_2)" % pred)
    result_A_B=ltnw.ask("%s(?data_A,?data_B)" % pred)
    result_B_B=ltnw.ask("%s(?data_B,?data_B_2)" % pred)
    plt.subplot(2,2,idx)
    idx+=1
    plt.title(pred)
    for i1,d1 in enumerate(data_A):
        for i2,d2 in enumerate(data_A):
            plt.plot([d1[0],d2[0]],[d1[1],d2[1]],alpha=result_A_A[i1,i2,0],c="black")            
    for i1,d1 in enumerate(data_A):
        for i2,d2 in enumerate(data_B):
            plt.plot([d1[0],d2[0]],[d1[1],d2[1]],alpha=result_A_B[i1,i2,0],c="black")
    for i1,d1 in enumerate(data_B):
        for i2,d2 in enumerate(data_B):
            plt.plot([d1[0],d2[0]],[d1[1],d2[1]],alpha=result_B_B[i1,i2,0],c="black")

plt.show()

