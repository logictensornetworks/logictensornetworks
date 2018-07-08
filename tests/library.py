#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import unittest
import tensorflow as tf

import logictensornetworks as ltn
import logictensornetworks_library as ltnl

class TestEqual(unittest.TestCase):

    def testDim1(self):
        eq=ltn.predicate("test",2,lambda x,y: ltnl.equal_euclidian(x,y))
        
        a=ltn.constant("a",[1.])
        b=ltn.constant("b",[1.])
        c=ltn.constant("b",[100.])
        
        sess=tf.Session()
        self.assertEqual(sess.run(eq(a,b)),[1.])
        self.assertEqual(sess.run(eq(a,c)),[0.])

    def testDim2(self):
        eq=ltn.predicate("test",4,lambda x,y: ltnl.equal_euclidian(x,y))
        
        a=ltn.constant("a",[1.,1.])
        b=ltn.constant("b",[1.,1.])
        c=ltn.constant("b",[100.,100.])
        
        sess=tf.Session()
        self.assertEqual(sess.run(eq(a,b)),[1.])
        self.assertEqual(sess.run(eq(a,c)),[0.])

    def testDimRadius(self):
        eq=ltn.predicate("test",2,lambda x,y: ltnl.equal_euclidian(x,y,diameter=100000))
        
        a=ltn.constant("a",[1.])
        b=ltn.constant("b",[1.])
        c=ltn.constant("b",[100.])
        
        sess=tf.Session()
        self.assertEqual(sess.run(eq(a,b)),[1.])
        self.assertTrue(sess.run(eq(a,c)) > [.99])

        eq=ltn.predicate("test",2,lambda x,y: ltnl.equal_euclidian(x,y,diameter=0.000001))
        
        a=ltn.constant("a",[1.])
        b=ltn.constant("b",[1.])
        c=ltn.constant("b",[1.00001])
        
        sess=tf.Session()
        self.assertEqual(sess.run(eq(a,b)),[1.])
        self.assertTrue(sess.run(eq(a,c)) < [.001])

if __name__ == "__main__":
    unittest.main()
