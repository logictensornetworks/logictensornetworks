#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest

import tests.wrapper
import tests.library

if __name__ == "__main__":
    # initialize the test suite
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    # add tests to the test suite
    suite.addTests(loader.loadTestsFromModule(tests.wrapper))
    suite.addTests(loader.loadTestsFromModule(tests.library))
    
    # initialize a runner, pass it your suite and run it
    runner = unittest.TextTestRunner(verbosity=3)
    result = runner.run(suite)
