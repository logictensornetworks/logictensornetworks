# Logic Tensor Networks (LTN)

## Dependencies

The following is what we are using for development. Basically similar versions should run fine.

* python3.6
* tensorflow >=1.8 (for running the core, wrapper etc)
* numpy >= 1.13.3 (for examples and tests)
* matplotlib >= 2.1 (for examples)

Installing dependencies is easy. Just use ``pip install tensorflow numpy matplotlib`` or use a virtualenv.

## Repository structure

* ``logictensornetworks.py`` -- core system for defining constants, variables, predicates, functions and formulas. 
* ``logictensornetworks_wrapper.py`` -- a simple wrapper that allows to express constants, variables, predicates, functions and formulas using strings. 
* ``logictensornetworks_library.py`` -- a collection of useful functions. 
* ``examples_ltn`` -- examples using the core system
* ``examples_ltnw`` -- examples using the wrapper
* ``tests`` -- tests

## Running tests

Tests are in ``tests`` and should be run from the project root. To run all available tests
use ``python3.6 tests/_all.py``.

Currently, tests are for the wrapper.

## Running examples

There are various examples for LTN core  ``examples_ltn`` and how to use the wrapper ``examples_ltnw``.

Run examples from the project root, e.g. ``python3.6 examples_ltn/multilable_classifier_simple.py``


## Papers 

* [Logic Tensor Networks: Deep Learning and Logical Reasoning from Data and Knowledge, Luciano Serafini and  Artur d'Avila Garcez, Arxiv.org](https://arxiv.org/abs/1606.04422)
* [Learning and Reasoning with Logic Tensor Networks, Luciano Serafini and rtur d'Avila Garces, Proc. AI*IA 2016](https://link.springer.com/chapter/10.1007/978-3-319-49130-1_25)
* [Learning and reasoning in logic tensor networks: theory and application to semantic image interpretation, Luciano Serafini, Ivan Donadello, Artur d'Avila Garces, Proc. ACM SAC 2017](https://dl.acm.org/citation.cfm?id=3019642)
* [Logic tensor networks for semantic image interpretation, Ivan Donadello, Luciano Serafini and Artur d'Avila Garces. Proc. IJCAI 2017](https://www.ijcai.org/proceedings/2017/0221.pdf)

## Tutorias 

Checkout recent tutorials on Logic Tensor Networks (LTN)

* [IJCNN 2018 tutorial](https://sites.google.com/fbk.eu/ltn/tutorial-ijcnn-2018)
* [IJCAI 2018 tutorial](https://sites.google.com/fbk.eu/ltn/tutorial-ijcai-2018)

## Other resources
* [What are “Logic Tensor Networks”?, Lucas Buchberger](https://lucas-bechberger.de/2017/11/16/what-are-logic-tensor-networks/)
* [Dagstuhl Seminar “Human-Like Neural-Symbolic Computation, Lucas Buchberger](https://lucas-bechberger.de/2017/05/17/dagstuhl-seminar-human-like-neural-symbolic-computation/)
* [Human-Like Neural-Symbolic Computing. 2017, Dagstuhl Seminar 17192](https://www.dagstuhl.de/17192)

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
