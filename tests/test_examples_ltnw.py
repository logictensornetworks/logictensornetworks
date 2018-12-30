#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import unittest

import logictensornetworks_wrapper as ltnw

class TestExamples(unittest.TestCase):
    def setUp(self):
        ltnw._reset()

    def testBinaryClassifier(self):
        import examples_ltnw.binary_classifier

    def testMultilabelClassification(self):
        import examples_ltnw.multilabel_classification

    def testMultilabelClassificationAxiomatized(self):
        import examples_ltnw.multilabel_classification_axiomatized

    def testRegressionLinar(self):
        import examples_ltnw.regression_linear

    def testRelations(self):
        import examples_ltnw.relations

    def testClustering(self):
        import examples_ltnw.clustering

    def testClusteringEuclidianDistance(self):
        import examples_ltnw.clustering_euclidian_distance
        
if __name__ == "__main__":
    unittest.main()

