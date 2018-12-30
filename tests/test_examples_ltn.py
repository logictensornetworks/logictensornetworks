#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import unittest

import logictensornetworks as ltn

class TestExamples(unittest.TestCase):

    def testClustering(self):
        import examples_ltn.clustering

    def testMultilabelClassifierSimple(self):
        import examples_ltn.multilabel_classifier_simple

    def testRegressionLinear(self):
        import examples_ltn.regression_linear

    def testSmokesFriendsCancer(self):
        import examples_ltn.smokes_friends_cancer

    def testSpatialRelations(self):
        import examples_ltn.spatial_relations

    def testPropositionalVariables(self):
        import examples_ltn.propositional_variables
        
        
if __name__ == "__main__":
    unittest.main()

