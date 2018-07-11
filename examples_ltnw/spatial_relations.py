# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger()
logger.basicConfig = logging.basicConfig(level=logging.DEBUG)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import logictensornetworks_wrapper as ltnw

import spatial_relations_data

# generate artificial data
nr_examples = 50 # positive and negative examples for each predicate
nr_test_examples=400

# 1) define the language and examples
ltnw.predicate("Left",8)
ltnw.variable("?left_xy",8)
ltnw.variable("?not_left_xy", 8)
ltnw.formula("forall ?left_xy: Left(?left_xy)")
ltnw.formula("forall ?not_left_xy: ~Left(?not_left_xy)")


ltnw.predicate("Right",8)
ltnw.variable("?right_xy",8)
ltnw.variable("?not_right_xy",8)
ltnw.formula("forall ?right_xy: Right(?right_xy)")
ltnw.formula("forall ?not_right_xy: ~Right(?not_right_xy)")

ltnw.predicate("Below",8)
ltnw.variable("?below_xy",8)
ltnw.variable("?not_below_xy",8)
ltnw.formula("forall ?below_xy: Below(?below_xy)")
ltnw.formula("forall ?not_below_xy: ~Below(?not_below_xy)")


ltnw.predicate("Above",8)
ltnw.variable("?above_xy",8)
ltnw.variable("?not_above_xy",8)
ltnw.formula("forall ?above_xy: Above(?above_xy)")
ltnw.formula("forall ?not_above_xy: ~Above(?not_above_xy)")

ltnw.predicate("Contains",8)
ltnw.variable("?contains_xy",8)
ltnw.variable("?not_contains_xy",8)
ltnw.formula("forall ?contains_xy: Contains(?contains_xy)")
ltnw.formula("forall ?not_contains_xy: ~Contains(?not_contains_xy)")

ltnw.predicate("Contained_in",8)
ltnw.variable("?contained_in_xy",8)
ltnw.variable("?not_contained_in_xy",8)
ltnw.formula("forall ?contained_in_xy: Contained_in(?contained_in_xy)")
ltnw.formula("forall ?not_contained_in_xy: ~Contained_in(?not_contained_in_xy)")

# 2) add axioms for relationship between predicates
x = ltnw.variable("?x",4)
y = ltnw.variable("?y",4)
z = ltnw.variable("?z",4)
ltnw.formula("forall ?x,?y: Left(?x,?y) -> Right(?y,?x)")
ltnw.formula("forall ?x,?y: Right(?x,?y) -> Left(?y,?x)")
ltnw.formula("forall ?x,?y: Above(?x,?y) -> Below(?y,?x)")
ltnw.formula("forall ?x,?y: Below(?x,?y) -> Above(?y,?x)")
ltnw.formula("forall ?x,?y: Contains(?x,?y) -> Contained_in(?y,?x)")
ltnw.formula("forall ?x,?y: Contained_in(?x,?y) -> Contains(?y,?x)")
ltnw.formula("forall ?x,?y: ~(Left(?x,?y) & Right(?y,?x))")
ltnw.formula("forall ?x,?y: ~(Above(?x,?y) & Below(?y,?x))")
ltnw.formula("forall ?x,?y: ~(Contains(?x,?y) & Contained_in(?y,?x))")

# 3) generate data
feed_dict={ "?left_xy"  : spatial_relations_data.generate_data(nr_examples,spatial_relations_data.is_left),
            "?not_left_xy" : spatial_relations_data.generate_data(nr_examples,spatial_relations_data.is_not_left),
            "?right_xy" : spatial_relations_data.generate_data(nr_examples,spatial_relations_data.is_right),
            "?not_right_xy" : spatial_relations_data.generate_data(nr_examples,spatial_relations_data.is_not_right),
            "?above_xy" : spatial_relations_data.generate_data(nr_examples,spatial_relations_data.is_above),
            "?not_above_xy" : spatial_relations_data.generate_data(nr_examples,spatial_relations_data.is_not_above),
            "?below_xy" : spatial_relations_data.generate_data(nr_examples,spatial_relations_data.is_below),
            "?not_below_xy" : spatial_relations_data.generate_data(nr_examples,spatial_relations_data.is_not_below),
            "?contains_xy" : spatial_relations_data.generate_data(nr_examples,spatial_relations_data.contains),
            "?not_contains_xy" : spatial_relations_data.generate_data(nr_examples,spatial_relations_data.not_contains),
            "?contained_in_xy" : spatial_relations_data.generate_data(nr_examples,spatial_relations_data.is_in),
            "?not_contained_in_xy" : spatial_relations_data.generate_data(nr_examples,spatial_relations_data.is_not_in),
            "?x" : spatial_relations_data.generate_rectangles(nr_examples),
            "?y" : spatial_relations_data.generate_rectangles(nr_examples),
            "?z" : spatial_relations_data.generate_rectangles(nr_examples)}

# 4) train the model
ltnw.initialize_knowledgebase(feed_dict=feed_dict,
                              optimizer=tf.train.AdamOptimizer(0.05),
                              formula_aggregator=lambda *x: tf.reduce_min(tf.concat(x,axis=0)))
ltnw.train(feed_dict=feed_dict,max_iterations=10000)

# 5) evaluate the truth of formulas not given directly to the model
for f in ["forall ?x,?y,?z: Contained_in(?x,?y) -> (Left(?y,?z)->Left(?x,?z))",
          "forall ?x,?y,?z: Contained_in(?x,?y) -> (Right(?y,?z)->Right(?x,?z))",
          "forall ?x,?y,?z: Contained_in(?x,?y) -> (Above(?y,?z)->Above(?x,?z))",
          "forall ?x,?y,?z: Contained_in(?x,?y) -> (Below(?y,?z)->Below(?x,?z))",
          "forall ?x,?y,?z: Contained_in(?x,?y) -> (Contains(?y,?z)->Contains(?x,?z))",
          "forall ?x,?y,?z: Contained_in(?x,?y) -> (Contained_in(?y,?z)->Contained_in(?x,?z))"]:
    print("%s: %s" % (f,ltnw.ask(f,feed_dict=feed_dict)))

# 6) plot some examples truth values of P(ct,t) where ct is a central rectangle, and
# t is a set of randomly generated rectangles
ltnw.constant("ct",[.5,.5,.3,.3])
test_data=spatial_relations_data.generate_rectangles(nr_test_examples)
ltnw.variable("?t",test_data)

fig = plt.figure(figsize=(12,8))
jet = cm = plt.get_cmap('jet')
cbbst = test_data[:,:2] + 0.5*test_data[:,2:]
for j,p in enumerate(["Left","Right","Above","Below","Contains","Contained_in"]):
    plt.subplot(2, 3, j + 1)
    formula="%s(ct,?t)" % p
    plt.title(formula)
    results=ltnw.ask(formula,feed_dict=feed_dict)
    plt.scatter(cbbst[:,0], cbbst[:,1], c=np.squeeze(results))
    plt.colorbar()
plt.show()