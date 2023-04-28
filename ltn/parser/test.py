import numpy as np
import tensorflow as tf

import ltn
from ltn.parser.parse import Parser


points = np.array(
        [[0.4,0.3],[1.2,0.3],[2.2,1.3],[1.7,1.0],[0.5,0.5],[0.3, 1.5],[1.3, 1.1],[0.9, 1.7],
        [3.4,3.3],[3.2,3.3],[3.2,2.3],[2.7,2.0],[3.5,3.5],[3.3, 2.5],[3.3, 1.1],[1.9, 3.7],[1.3, 3.5],[3.3, 1.1],[3.9, 3.7]])
point_a = [3.3,2.5]
point_b = [1.3,1.1]

class ModelC(tf.keras.Model):
    def __init__(self):
        super(ModelC, self).__init__()
        self.dense1 = tf.keras.layers.Dense(5, activation=tf.nn.elu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.elu)
        self.dense3 = tf.keras.layers.Dense(2)

    def call(self, inputs):
        """inputs[0]: point"""
        x = inputs[0]
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

C = ltn.Predicate.FromLogits(ModelC(), activation_function="softmax", with_class_indexing=True)
x1 = ltn.Variable("x1",points)
x2 = ltn.Variable("x2",points)
a = ltn.Constant([3.3,2.5], trainable=False)
b = ltn.Constant([1.3,1.1], trainable=False)
l_a = ltn.Constant(0, trainable=False)
l_b = ltn.Constant(1, trainable=False)
l = ltn.Variable.from_constants("l",[l_a, l_b])

Sim = ltn.Predicate.Lambda(
    lambda args: tf.exp(-1.*tf.sqrt(tf.reduce_sum(tf.square(args[0]-args[1]),axis=1)))
)

grounding = ltn.wrapper.Grounding(predicates={"C":C,"Sim":Sim},
                    variables={"x1":x1, "x2":x2, "l":l}, 
                    constants={"a":a,"b":b,"l_a":l_a,"l_b":l_b})
op_config = ltn.utils.wrapper.get_stable_operator_config()

parser = Parser()

#parse_results = parser.parse_formula("forall disease ( exists drug (cures(drug,disease)) )")
#print(parse_results)
#print(json.dumps(parse_results, indent=2))

res = parser.parse_formula("forall x1 ~(C(x1,l_a) & C(x1,l_b))")
print(res)
res[0].print_tree()
print(res[0].eval(grounding=grounding, op_config=op_config))

# print(parser.parse_formula("forall x,y P(x,y)"))
#parse_results = parser.parse_formula("forall x ( exists y P(x,y) )")

#parse_results = parser.parse_formula("A(x) & B(x)")
# print(parser.parse_formula("p(f(a,b),c,f(d,f(a)))"))
