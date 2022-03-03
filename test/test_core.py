import unittest
import pdb

import numpy as np
import tensorflow as tf

from ltn import core
from ltn import fuzzy_ops

def array_allclose(actual,desired):
    try:
        return np.allclose(actual, desired) and (actual.shape == desired.shape)
    except AttributeError:
        actual, desired = np.array(actual), np.array(desired)
        return np.allclose(actual, desired) and (actual.shape == desired.shape)

class TestConstant(unittest.TestCase):
    def setUp(self):
        self.c_val_1 = [2.1,3.]
        self.c_val_2 = [[4.2,3,2.5],[4,-1.3,1.8]]
        self.c_val_3 = [4.1,0.5]
        self.c_vals = [self.c_val_1,self.c_val_2,self.c_val_3]
        self.c_val_4 = 1.

    def test_init_from_python(self):
        for c_val in self.c_vals:
            c = core.Constant(c_val, trainable=False)
            self.assertTrue(array_allclose(c.tensor.numpy(), c_val))
            self.assertTrue(isinstance(c.tensor, tf.Tensor))
            self.assertEqual(c.free_vars, [])

    def test_init_from_np(self):
        for c_val in self.c_vals:
            c_val = np.array(c_val)
            c = core.Constant(c_val, trainable=False)
            self.assertTrue(array_allclose(c.tensor.numpy(), c_val))
            self.assertTrue(isinstance(c.tensor, tf.Tensor))
            self.assertEqual(c.free_vars, [])

    def test_init_from_tf(self):
        for c_val in self.c_vals:
            c_val = tf.constant(c_val)
            c = core.Constant(c_val, trainable=False)
            self.assertTrue(array_allclose(c.tensor.numpy(), c_val))
            self.assertTrue(isinstance(c.tensor, tf.Tensor))
            self.assertEqual(c.free_vars, [])

    def test_cast(self):
        pass

    def test_expand_dims(self):
        # rationale for expand dims:
        #  P([x,y]), if x.shape [free_dim,1] and y.shape [free_dim], the broadcast is broken
        pass

    def test_trainable(self):
        for c_val in self.c_vals:
            c = core.Constant(c_val, trainable=True)
            self.assertTrue(array_allclose(c.tensor.numpy(), c_val))
            self.assertTrue(isinstance(c.tensor, tf.Variable))
            self.assertTrue(c.tensor.trainable)
            self.assertEqual(c.free_vars, [])

class TestVariable(unittest.TestCase):
    def setUp(self):
        self.x_val_1 = np.random.rand(10,2) # 10 individuals in R^2
        self.x_val_2 = np.random.rand(5,2) # 5 individuals in R^2
        self.x_val_3 = np.random.rand(7,2,2) # 7 individuals in R^(2x2)
        self.x_vals = [self.x_val_1, self.x_val_2, self.x_val_3]
        self.x_val_4 = np.random.rand(10) # 10 inviduals in R

    def test_init(self):
        for x_val in self.x_vals:
            label = "x"
            x = core.Variable("x",x_val)
            self.assertTrue(array_allclose(x.tensor.numpy(), x_val))
            self.assertTrue(isinstance(x.tensor, tf.Tensor))
            self.assertEqual(x.free_vars, [label])
            self.assertEqual(x.label, label)
    
    def test_from_trainable_constants(self):
        c1 = core.Constant([2.1,3], trainable=True)
        c2 = core.Constant([4.5,0.8], trainable=True)
        with tf.GradientTape() as tape:
            x = core.Variable.from_constants('x', [c1,c2], tape=tape)
        self.assertTrue(tape.gradient(x.tensor,c1.tensor) is not None)
        
        tape = tf.GradientTape()
        with self.assertRaises(ValueError):
            x = core.Variable.from_constants('x', [c1,c2], tape=tape)

    def test_from_non_trainable_constants(self):
        c1 = core.Constant([2.1,3], trainable=False)
        c2 = core.Constant([4.5,0.8], trainable=False)
        with tf.GradientTape() as tape:
            x = core.Variable.from_constants('x', [c1,c2], tape=None)
        self.assertTrue(tape.gradient(x.tensor,c1.tensor) is None)

    def test_cast(self):
        pass

    def test_expand_dims(self):
        pass

class TestPredicate(unittest.TestCase):
    def setUp(self):
        self.c1 = core.Constant([2.1,3],trainable=False)
        self.c2 = core.Constant([4.5,0.8],trainable=False)
        self.n_x = 10
        self.x = core.Variable('x',np.random.normal(0.,1.,(self.n_x,2)))
        self.n_y = 5
        self.y = core.Variable('y',np.random.normal(0.,4.,(self.n_y,2)))

    def test_from_tf_model_1input(self):
        class ModelP(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.dense1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
            def call(self, x):
                return self.dense1(x)
        P = core.Predicate(ModelP())
        # Produces correct result on constant
        self.assertEqual(P(self.c1).tensor, P.model(tf.expand_dims(self.c1.tensor,axis=0)))
        # Produces correct outputs with variable
        self.assertTrue(array_allclose(P(self.x).tensor, tf.squeeze(P.model(self.x.tensor))))
        self.assertEqual(P(self.x).free_vars,['x'])
        self.assertEqual(P(self.x)._get_dim_of_free_var('x'), self.n_x)
        
    def test_from_lambda_1input(self):
        mu = tf.constant([2.,3.])
        P = core.Predicate.Lambda(lambda x: tf.exp(-tf.norm(x-mu,axis=1)))
        # Produces correct result on constant
        self.assertEqual(P(self.c1).tensor, P.model(tf.expand_dims(self.c1.tensor,axis=0)))
        # Produces correct outputs with variable
        self.assertTrue(array_allclose(P(self.x).tensor, P.model(self.x.tensor)))
        self.assertEqual(P(self.x).free_vars,['x'])
        self.assertEqual(P(self.x)._get_dim_of_free_var('x'), self.n_x)

    def test_from_default_MLP_1input(self):
        P = core.Predicate.MLP(input_shapes=[2])
        # Produces correct result on constant
        self.assertEqual(P(self.c1).tensor, P.model(tf.expand_dims(self.c1.tensor,axis=0)))
        # Produces correct outputs with variable
        self.assertTrue(array_allclose(P(self.x).tensor, tf.squeeze(P.model(self.x.tensor))))
        self.assertEqual(P(self.x).free_vars,['x'])
        self.assertEqual(P(self.x)._get_dim_of_free_var('x'), self.n_x)

    def test_from_tf_model_2inputs(self):
        class ModelP(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.dense1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
            def call(self, inputs):
                x = tf.concat([inputs[0],inputs[1]],axis=1)
                return self.dense1(x)
        P = core.Predicate(ModelP())
        # Produces correct result on constant
        self.assertEqual(
            P([self.c1,self.c2]).tensor, 
            P.model([tf.expand_dims(self.c1.tensor,axis=0),
                tf.expand_dims(self.c2.tensor,axis=0)]))
        # Produces correct outputs with variable
        self.assertEqual(
            P([self.x,self.y]).take('x',0).take('y',0).tensor, 
            P.model([tf.gather(self.x.tensor,[0]),tf.gather(self.y.tensor,[0])]))
        self.assertEqual(sorted(P([self.x,self.y]).free_vars),['x','y'])
        self.assertEqual(P([self.x,self.y])._get_dim_of_free_var('x'), self.n_x)
        self.assertEqual(P([self.x,self.y])._get_dim_of_free_var('y'), self.n_y)

    def test_from_lambda_2inputs(self):
        P = core.Predicate.Lambda(lambda args: tf.exp(-tf.norm(args[0]-args[1],axis=1)))
        # Produces correct result on constant
        self.assertEqual(
            P([self.c1,self.c2]).tensor, 
            P.model([tf.expand_dims(self.c1.tensor,axis=0),
                tf.expand_dims(self.c2.tensor,axis=0)]))
        # Produces correct outputs with variable
        self.assertEqual(
            P([self.x,self.y]).take('x',0).take('y',0).tensor, 
            P.model([tf.gather(self.x.tensor,[0]),tf.gather(self.y.tensor,[0])]))
        self.assertEqual(sorted(P([self.x,self.y]).free_vars),['x','y'])
        self.assertEqual(P([self.x,self.y])._get_dim_of_free_var('x'), self.n_x)
        self.assertEqual(P([self.x,self.y])._get_dim_of_free_var('y'), self.n_y)

    def test_from_default_MLP_2inputs(self):
        P = core.Predicate.MLP(input_shapes=[2,2])
        # Produces correct result on constant
        self.assertTrue(array_allclose(
            tf.squeeze(P([self.c1,self.c2]).tensor), 
            tf.squeeze(P.model([tf.expand_dims(self.c1.tensor,axis=0),
                tf.expand_dims(self.c2.tensor,axis=0)])) ))
        # Produces correct outputs with variable
        self.assertTrue(array_allclose(
            tf.squeeze(P([self.x,self.y]).take('x',0).take('y',0).tensor), 
            tf.squeeze(P.model([tf.gather(self.x.tensor,[0]),tf.gather(self.y.tensor,[0])])) ))
        self.assertEqual(sorted(P([self.x,self.y]).free_vars),['x','y'])
        self.assertEqual(P([self.x,self.y])._get_dim_of_free_var('x'), self.n_x)
        self.assertEqual(P([self.x,self.y])._get_dim_of_free_var('y'), self.n_y)

class TestFunction(unittest.TestCase):
    def setUp(self):
        self.c1 = core.Constant([2.1,3],trainable=False)
        self.c2 = core.Constant([4.5,0.8],trainable=False)
        self.n_x = 10
        self.x = core.Variable('x',np.random.normal(0.,1.,(self.n_x,2)))
        self.n_y = 5
        self.y = core.Variable('y',np.random.normal(0.,4.,(self.n_y,2)))

    def test_from_tf_model(self):
        class ModelF(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.dense1 = tf.keras.layers.Dense(5)
            def call(self, x):
                return self.dense1(x)   
        f = core.Function(ModelF())
        # Produces correct result on constant
        self.assertTrue(array_allclose(f(self.c1).tensor, f.model(tf.expand_dims(self.c1.tensor,axis=0))[0]))
        # Produces correct outputs with variable
        self.assertTrue(array_allclose(f(self.x).tensor, f.model(self.x.tensor)))
        self.assertEqual(f(self.x).free_vars,['x'])
        self.assertEqual(f(self.x)._get_dim_of_free_var('x'), self.n_x)
        
    def test_from_lambda(self):
        f = core.Function.Lambda(lambda args: args[0]-args[1])
        # Produces correct result on constant
        self.assertTrue(array_allclose(
            f([self.c1,self.c2]).tensor, 
            f.model([tf.expand_dims(self.c1.tensor,axis=0),
                tf.expand_dims(self.c2.tensor,axis=0)])[0] ))
        # Produces correct outputs with variable
        self.assertTrue(array_allclose(
            f([self.x,self.y]).take('x',0).take('y',0).tensor, 
            f.model([tf.gather(self.x.tensor,[0]),tf.gather(self.y.tensor,[0])])[0] ))
        self.assertEqual(sorted(f([self.x,self.y]).free_vars),['x','y'])
        self.assertEqual(f([self.x,self.y])._get_dim_of_free_var('x'), self.n_x)
        self.assertEqual(f([self.x,self.y])._get_dim_of_free_var('y'), self.n_y)

    def test_from_default_MLP(self):
        f = core.Function.MLP(input_shapes=[2,2],output_shape=[2])
        # Produces correct result on constant
        self.assertTrue(array_allclose(
            f([self.c1,self.c2]).tensor, 
            f.model([tf.expand_dims(self.c1.tensor,axis=0),
                tf.expand_dims(self.c2.tensor,axis=0)])[0] ))
        # Produces correct outputs with variable
        self.assertTrue(array_allclose(
            f([self.x,self.y]).take('x',0).take('y',0).tensor, 
            f.model([tf.gather(self.x.tensor,[0]),tf.gather(self.y.tensor,[0])])[0] ))
        self.assertEqual(sorted(f([self.x,self.y]).free_vars),['x','y'])
        self.assertEqual(f([self.x,self.y])._get_dim_of_free_var('x'), self.n_x)
        self.assertEqual(f([self.x,self.y])._get_dim_of_free_var('y'), self.n_y)

class TestProposition(unittest.TestCase):
    def setUp(self):
        self.And = core.Wrapper_Connective(fuzzy_ops.And_Prod())

    def test_trainable(self):
        a = core.Proposition(0., trainable=True)
        b = core.Proposition(0., trainable=True)
        with tf.GradientTape() as tape:
            res = self.And(a,b)
        self.assertTrue(tape.gradient(res.tensor,a.tensor) is not None)

    def test_non_trainable(self):
        a = core.Proposition(0., trainable=False)
        b = core.Proposition(0., trainable=False)
        with tf.GradientTape() as tape:
            res = self.And(a,b)
        self.assertTrue(tape.gradient(res.tensor,a.tensor) is None)

class TestBroadcast(unittest.TestCase):
    def setUp(self):
        self.var_settings = {
            "x1":{
            "n_individuals":10,
            "shape_individual":[2]
            },
            "x2":{
            "n_individuals":5,
            "shape_individual":[2]
            },
            "x3":{
            "n_individuals":7,
            "shape_individual":[2,2]
            }
        }
        self.xs = {}
        for label,v_s in self.var_settings.items():
            x = core.Variable(
                label,
                np.random.rand(v_s["n_individuals"],*v_s["shape_individual"])
            )
            self.xs[label] = x
        
    def test_free_dims_broadcast_variables(self):
        """Adds the correct dimensions for the free variables."""
        exprs = core.broadcast_exprs(list(self.xs.values()), in_place=False)
        for expr in exprs:
            self.assertEqual(sorted(expr.free_vars), sorted(self.var_settings.keys()))
            for label,v_s in self.var_settings.items():
                self.assertEqual(expr._get_dim_of_free_var(label).numpy(), v_s["n_individuals"])

    def test_broadcast_variables_intact(self):
        """Keeps the correct values"""
        exprs = core.broadcast_exprs(list(self.xs.values()), in_place=False)
        for expr, x in zip(exprs, list(self.xs.values())):
            other_free_vars = [x_.label for x_ in list(self.xs.values()) if x_.label != x.label]
            for free_var in other_free_vars:
                expr = expr.take(free_var,0)
            self.assertTrue(array_allclose(x.tensor, expr.tensor))

    def test_free_dims_wt_broadcast_constant(self):
        """Adds the correct dimensions if a constant is involved"""
        c1 = core.Constant([2.1,3], trainable=False)
        exprs = core.broadcast_exprs(list(self.xs.values())+[c1])
        for expr in exprs:
            self.assertEqual(sorted(expr.free_vars), sorted(self.var_settings.keys()))
            for label,v_s in self.var_settings.items():
                self.assertEqual(expr._get_dim_of_free_var(label).numpy(), v_s["n_individuals"])

    def test_broadcast_formulas(self):
        """Adds the correct dimensions between formulas"""
        p1 = core.Predicate.MLP(input_shapes=[self.var_settings['x1']['shape_individual'],
                self.var_settings['x2']['shape_individual']])
        p2 = core.Predicate.MLP(input_shapes=[self.var_settings['x1']['shape_individual']])
        p3 = core.Predicate.MLP(input_shapes=[self.var_settings['x3']['shape_individual']])
        phi1 = p1([self.xs['x1'],self.xs['x2']])
        phi2 = p2([self.xs['x1']])
        phi3 = p3([self.xs['x3']])
        exprs = core.broadcast_exprs([phi1,phi2,phi3])
        for expr in exprs:
            self.assertEqual(sorted(expr.free_vars), sorted(self.var_settings.keys()))
            for label,v_s in self.var_settings.items():
                self.assertEqual(expr._get_dim_of_free_var(label).numpy(), v_s["n_individuals"])
            
    def test_in_place(self):
        pass

class TestConnective(unittest.TestCase):
    def setUp(self):
        self.And = core.Wrapper_Connective(fuzzy_ops.And_Prod())
        self.Not = core.Wrapper_Connective(fuzzy_ops.Not_Std())
        self.c1 = core.Constant([2.1,3],trainable=True)
        self.n_x = 10
        self.x = core.Variable('x',np.random.normal(0.,1.,(self.n_x,2)))
        self.n_y = 5
        self.y = core.Variable('y',np.random.normal(0.,4.,(self.n_y,2)))
        self.p1 = core.Predicate.MLP([2])
        self.p2 = core.Predicate.MLP([2,2])
        self.a = core.Proposition(0., trainable=True)

    def test_unary_connective_variables(self):
        """Unary connective maintains variables"""
        res = self.Not(self.p1(self.x))
        self.assertEqual(res.free_vars,['x'])
        self.assertEqual(res._get_dim_of_free_var('x'), self.n_x)

    def test_binary_connective_variables(self):
        """Binary connectives joins variables"""
        res = self.And(self.p1(self.x),self.p2([self.x,self.y]))
        self.assertEqual(sorted(res.free_vars),['x','y'])
        self.assertEqual(res._get_dim_of_free_var('x'), self.n_x)
        self.assertEqual(res._get_dim_of_free_var('y'), self.n_y)

    def test_binary_connective_wt_constant(self):
        """Binary connective doesn't break with constant"""
        res = self.And(self.p1(self.c1),self.p2([self.x,self.c1]))
        self.assertEqual(res.free_vars,['x'])
        self.assertEqual(res._get_dim_of_free_var('x'), self.n_x)

    def test_binary_connective_wt_proposition(self):
        """Binary connective doesn't break with proposition"""
        res = self.And(self.a,self.p2([self.x,self.y]))
        self.assertEqual(sorted(res.free_vars),['x','y'])
        self.assertEqual(res._get_dim_of_free_var('x'), self.n_x)
        self.assertEqual(res._get_dim_of_free_var('y'), self.n_y)

    def test_gradients(self):
        """Tracks gradients"""
        with tf.GradientTape() as tape:
            res = self.And(self.a,self.p2([self.x,self.y]))
        self.assertTrue(tape.gradient(res.tensor,self.a.tensor) is not None)

    def test_values_correct(self):
        """Test if the value in the result is correct"""
        phi1 = self.p1(self.x)
        phi2 = self.p2([self.x,self.y])
        res = self.And(phi1,phi2)
        self.assertEqual(
            res.take('x',0).take('y',0).tensor,
            self.And.connective_op(phi1.take('x',0).tensor,phi2.take('x',0).take('y',0).tensor)
        )

class TestQuantifier(unittest.TestCase):
    def setUp(self):
        self.Forall = core.Wrapper_Quantifier(fuzzy_ops.Aggreg_pMeanError(p=2),semantics="forall")
        self.Exists = core.Wrapper_Quantifier(fuzzy_ops.Aggreg_pMean(p=5),semantics="exists")
        self.n_x = 10
        self.x = core.Variable('x',np.random.normal(0.,1.,(self.n_x,2)))
        self.n_y = 5
        self.y = core.Variable('y',np.random.normal(0.,4.,(self.n_y,2)))
        self.n_z = 5
        self.z = core.Variable('z',np.random.normal(0.,4.,(self.n_z,2)))
        self.p1 = core.Predicate.MLP([2,2])
        self.p2 = core.Predicate.MLP([2,2,2])
        
    def test_aggreg_one_var(self):
        res = self.Forall(self.x,self.p1([self.x,self.y]))
        self.assertEqual(sorted(res.free_vars),['y'])
        self.assertEqual(res._get_dim_of_free_var('y'),self.n_y)

        res = self.Forall(self.y,self.p1([self.x,self.y]))
        self.assertEqual(sorted(res.free_vars),['x'])
        self.assertEqual(res._get_dim_of_free_var('x'),self.n_x)

    def test_aggreg_several_vars(self):
        res = self.Forall((self.x,self.y),self.p1([self.x,self.y]))
        self.assertEqual(sorted(res.free_vars),[])
        self.assertEqual(res.tensor.shape,[])

    def test_gradients(self):
        """Tracks gradients"""
        c_s = [core.Constant(np.random.rand(2), trainable=True) for _ in range(self.n_x)]
        with tf.GradientTape() as tape:
            x = core.Variable.from_constants('x', c_s, tape)
            phi = self.p1([x,self.y])
            res = self.Forall(x,phi)
        self.assertTrue(tape.gradient(res.tensor,c_s[0].tensor) is not None)
        
    def test_values_correct(self):
        phi = self.p2([self.x,self.y,self.z])      
        self.assertTrue(array_allclose(
            self.Forall((self.y),phi).take('x',0).take('z',0).tensor,
            self.Forall.aggreg_op(phi.tensor, axis=phi._get_axis_of_free_var('y'))[0,0]
        ))
        self.assertTrue(array_allclose(
            self.Forall((self.x,self.z),phi).take('y',0).tensor,
            self.Forall.aggreg_op(phi.tensor, 
                axis=[phi._get_axis_of_free_var('x'),phi._get_axis_of_free_var('z')])[0]
        ))

class TestTransposeFreeVars(unittest.TestCase):
    def setUp(self):
        self.n_x = 10
        self.x = core.Variable('x',np.random.normal(0.,1.,(self.n_x,2)))
        self.n_y = 5
        self.y = core.Variable('y',np.random.normal(0.,4.,(self.n_y,2)))
        self.n_z = 3
        self.z = core.Variable('z',np.random.normal(0.,4.,(self.n_z,2)))
        self.p = core.Predicate.MLP([2,2,2])

    def test_transpose(self):
        phi = self.p([self.x,self.y,self.z])
        var_order = ['x','y','z']
        phi = core.transpose_free_vars(phi, var_order)
        self.assertEqual(phi.free_vars, var_order)
        self.assertEqual(phi._get_dim_of_free_var('x'),self.n_x)
        self.assertEqual(phi._get_axis_of_free_var('x'),var_order.index('x'))
        var_order = ['z','x','y']
        phi = core.transpose_free_vars(phi, var_order)
        self.assertEqual(phi.free_vars, var_order)
        self.assertEqual(phi._get_dim_of_free_var('x'),self.n_x)
        self.assertEqual(phi._get_axis_of_free_var('x'),var_order.index('x'))
        self.assertEqual(phi._get_dim_of_free_var('z'),self.n_z)
        self.assertEqual(phi._get_axis_of_free_var('z'),var_order.index('z'))
    
    def test_conserves_values(self):
        phi = self.p([self.x,self.y,self.z])
        phi1 = core.transpose_free_vars(phi, ['x','y','z'])
        phi2 = core.transpose_free_vars(phi1, ['z','x','y'])
        phi3 = core.transpose_free_vars(phi2, ['x','y','z'])
        self.assertTrue(array_allclose(phi1.tensor,phi3.tensor))

    def test_in_place(self):
        pass

class TestBroadcastToMask(unittest.TestCase):
    def setUp(self):
        self.n_x = 10
        self.x = core.Variable('x',np.random.normal(0.,1.,(self.n_x,1)))
        self.n_y = 10
        self.y = core.Variable('y',np.random.normal(0.,1.,(self.n_y,1)))
        self.n_z = 10
        self.z = core.Variable('z',np.random.normal(0.,2.,(self.n_z,1)))
        self.is_greater_than = core.Predicate.Lambda(
            lambda inputs: inputs[0] > inputs[1]
        )
        self.add = core.Function.Lambda(lambda inputs: inputs[0]+inputs[1])
        self.mask1 = self.is_greater_than([self.x,self.y])
        self.mask2 = self.is_greater_than([self.add([self.x,self.y]), self.z])

        self.p1 = core.Predicate.MLP([1,1])
        self.p2 = core.Predicate.MLP([1,1,1])

    def test_same_vars_in_wff(self):
        phi1 = self.p1([self.x,self.y])
        casted_phi1 = core.broadcast_wff_and_mask(phi1,self.mask1)
        self.assertEqual(casted_phi1.free_vars[:len(self.mask1.free_vars)], self.mask1.free_vars)
        take_x = np.random.randint(self.n_x)
        self.assertTrue(array_allclose(
            casted_phi1.take('x',take_x).tensor, 
            phi1.take('x', take_x).tensor))
        take_y = np.random.randint(self.n_y)
        self.assertTrue(array_allclose(
            casted_phi1.take('y',take_y).tensor, 
            phi1.take('y', take_y).tensor))
        
    def test_more_vars_in_wff(self):
        phi2 = self.p2([self.x,self.y,self.z])
        casted_phi2 = core.broadcast_wff_and_mask(phi2,self.mask1)
        self.assertEqual(casted_phi2.free_vars[:len(self.mask1.free_vars)], self.mask1.free_vars)
        take_x = np.random.randint(self.n_x)
        self.assertTrue(array_allclose(
            casted_phi2.take('x',take_x).tensor, 
            phi2.take('x', take_x).tensor))
        take_y = np.random.randint(self.n_y)
        self.assertTrue(array_allclose(
            casted_phi2.take('y',take_y).tensor, 
            phi2.take('y', take_y).tensor))

    def test_less_vars_in_wff(self):
        phi1 = self.p1([self.x,self.y])
        casted_phi1 = core.broadcast_wff_and_mask(phi1,self.mask2)
        self.assertEqual(casted_phi1.free_vars[:len(self.mask2.free_vars)], self.mask2.free_vars)
        take_x = np.random.randint(self.n_x)
        self.assertTrue(array_allclose(
            casted_phi1.take('x',take_x).take('z',0).tensor, 
            phi1.take('x', take_x).tensor))
        take_y = np.random.randint(self.n_y)
        self.assertTrue(array_allclose(
            casted_phi1.take('y',take_y).take('z',0).tensor, 
            phi1.take('y', take_y).tensor))

class TestGuardedQuantifier(unittest.TestCase):
    def setUp(self):
        self.n_x = 10
        self.x = core.Variable('x',np.random.normal(0.,1.,(self.n_x,1)))
        self.n_y = 10
        self.y = core.Variable('y',np.random.normal(0.,1.,(self.n_y,1)))
        self.is_greater_than = core.Predicate.Lambda(
            lambda inputs: inputs[0] > inputs[1]
        )
        self.mask1 = self.is_greater_than([self.x,self.y])

        self.p1 = core.Predicate.MLP([1,1])

        self.Forall = core.Wrapper_Quantifier(fuzzy_ops.Aggreg_pMeanError(p=2),semantics="forall")
        self.Exists = core.Wrapper_Quantifier(fuzzy_ops.Aggreg_pMean(p=5),semantics="exists")
        

    def test_values_correct(self):
        take_x = np.random.randint(self.n_x)
        actual = self.Forall(self.y,self.p1([self.x,self.y]),mask=self.mask1).take('x', take_x)
        safe_y = core.Variable('safe_y',tf.boolean_mask(self.y.tensor, self.x.take('x',take_x).tensor > self.y.tensor))
        desired = self.Forall(safe_y, self.p1([self.x,safe_y])).take('x',take_x)
        self.assertTrue(array_allclose(actual.tensor,desired.tensor))

    def test_gradients(self):
        pass
    
    def test_empty_semantics_forall(self):
        x = core.Variable('x',np.random.rand(self.n_x,1))
        y = core.Variable('y',np.random.rand(self.n_y,1)+1.) # all y are greater
        mask1 = self.is_greater_than([x,y])
        take_y = np.random.randint(self.n_y)
        actual = self.Forall(x, self.p1([x,y]), mask=mask1).take("y",take_y).tensor
        desired = 1.
        self.assertEqual(actual,desired)

    def test_empty_semantics_exist(self):
        x = core.Variable('x',np.random.rand(self.n_x,1))
        y = core.Variable('y',np.random.rand(self.n_y,1)+1.) # all y are greater
        mask1 = self.is_greater_than([x,y])
        take_y = np.random.randint(self.n_y)
        actual = self.Exists(x, self.p1([x,y]), mask=mask1).take("y",take_y).tensor
        desired = 0.
        self.assertEqual(actual,desired)

    def test_aggreg_first_var_of_mask(self):
        x = core.Variable('x',np.random.rand(3,1))
        y = core.Variable('y',np.random.rand(4,1))
        mask = core.Formula(
            tf.constant(
                [[1.,1.,0.,0.],
                [0.,1.,1.,0.],
                [0.,0.,1.,0.]]),
            free_vars=['x','y'])
        wff = core.Formula(
            tf.constant(
                [[.4,.2,.8,.0],
                [.9,.1,.5,.2],
                [.3,.0,.8,.9]]),
            free_vars=['x','y'])
        # Result after mask
        #       [[.4,.2,  ,  ],
        #        [  ,.1,.5,  ],
        #        [  ,  ,.8,  ]]),
        Exists = core.Wrapper_Quantifier(fuzzy_ops.Aggreg_Mean(),semantics="exists")
        
        actual = Exists(x,wff,mask=mask)
        expected = np.array([.4,.15,.65,.0])
        self.assertTrue(array_allclose(actual.tensor,expected))

class TestDiag(unittest.TestCase):
    pass

class TestInTfFunction(unittest.TestCase):
    pass

class TestTypeCheck(unittest.TestCase):
    def setUp(self):
        self.x = core.Variable('x',np.random.rand(3,1))
        self.y = core.Variable('y',np.random.rand(4,1))
        self.c = core.Constant([3.], trainable=False)
        self.f1 = core.Function.MLP(input_shapes=[1],output_shape=[1])
        self.f2 = core.Function.MLP(input_shapes=[1,1],output_shape=[1])
        self.p1 = core.Predicate.MLP(input_shapes=[1])
        self.p2 = core.Predicate.MLP(input_shapes=[1,1])
        self.q = core.Proposition(0., trainable=False)
        self.And = core.Wrapper_Connective(fuzzy_ops.And_Prod())
        self.Not = core.Wrapper_Connective(fuzzy_ops.Not_Std())
        self.Exists = core.Wrapper_Quantifier(fuzzy_ops.Aggreg_Mean(),semantics="exists")
        self.mask = core.Formula(
            tf.constant(
                [[1.,1.,0.,0.],
                [0.,1.,1.,0.],
                [0.,0.,1.,0.]]),
            free_vars=['x','y'])

    def test_predicate(self):
        try:
            self.p1(self.x)
            self.p2([self.x,self.c])
            self.p1(self.f1(self.x))
        except TypeError:
            self.fail("TypeError raised unexpectedly.")
        with self.assertRaises(TypeError):
            self.p1(self.x.tensor)
        with self.assertRaises(TypeError):
            self.p2([self.x, self.p1(self.x)])

    def test_function(self):
        try:
            self.f1(self.x)
            self.f2([self.x,self.c])
            self.f1(self.f1(self.x))
        except TypeError:
            self.fail("TypeError raised unexpectedly.")
        with self.assertRaises(TypeError):
            self.f1(self.x.tensor)
        with self.assertRaises(TypeError):
            self.f2([self.x, self.p1(self.x)])

    def test_connective(self):
        try:
            self.And(self.p1(self.x),self.q)
            self.Not(self.p2([self.x,self.c]))
        except TypeError:
            self.fail("TypeError raised unexpectedly.")
        with self.assertRaises(TypeError):
            self.And(self.q, self.f1(self.x))
        with self.assertRaises(TypeError):
            self.Not(self.p1(self.x).tensor)

    def test_quantifier(self):
        try:
            self.Exists(self.x,self.p1(self.x))
            self.Exists((self.x,self.y),self.p2([self.x,self.y]))
            self.Exists((self.x,self.y),self.p2([self.x,self.y]),mask=self.mask)
        except TypeError:
            self.fail("TypeError raised unexpectedly.")
        with self.assertRaises(TypeError):
            self.Exists(self.q, self.p1(self.x))
        with self.assertRaises(TypeError):
            self.Exists(self.x,self.f1(self.x))
        with self.assertRaises(TypeError):
            self.Exists((self.x,self.y),self.p2([self.x,self.y]),mask=self.mask.tensor)

        