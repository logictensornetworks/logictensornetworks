# Logic Tensor Networks (LTN)

### Dependencies

The following is what we are using for development. Basically similar versions should run fine.

* python3.6
* tensorflow >=1.8 (for running the core, wrapper etc)
* numpy >= 1.13.3 (for examples and tests)
* matplotlib >= 2.1 (for examples)

Installing is easy using ``pip install tensorflow numpy matplotlib`` or using a virtualenv.

## Running tests

Tests are in ``tests`` and should be run from the project root. To run all available tests
use ``python3.6 tests/_all.py``.

Currently, tests are for the wrapper.

## Running examples

There are various examples for LTN core  ``examples_ltn`` and how to use the wrapper ``examples_ltnw``.

Run examples from the project root, e.g. ``python3.6 examples_ltn/multilable_classifier_simple.py``

## Documentation

Checkout recent tutorials on Logic Tensor Networks (LTN)

* [IJCNN 2018 tutorial](https://sites.google.com/fbk.eu/ltn/tutorial-ijcnn-2018)
* [IJCAI 2018 tutorial](https://sites.google.com/fbk.eu/ltn/tutorial-ijcai-2018)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

LTN has been developed thanks to active contributions and discussions with the following people:
* Alessandro Daniele (FBK)
* Artur d’Avila Garces (City)
* Francesco Giannini (UniSiena)
* Giuseppe Marra (UniSiena)
* Ivan Donadello (FBK)
* Lucas Brukberger (UniOsnabruck)
* Luciano Serafini (FBK)
* Marco Gori (UniSiena)
* Michael Spranger (Sony CSL)
* Michelangelo Diligenti (UniSiena)
