


Roadmap to version 1.0
======================

- Automatic stub generation for the python bindings: working autocompletion is a must.
    - Using pybind11-stubs, we must first make sure no c++ types propagate to the docstrings. For torch types, this might mean some amount of post-processing of the stubs. (torch::<anything> -> torch.<anything> in most cases.)
    - mypy can also generate stubs, but supposedly that it struggles a bit more with pybind11. pybind11-stubs is currently integrating his stub generating logic into mypy, we should keep an eye on the state of mypy
- Proper support of python slices (in python) and torch::slices (in c++)
- Improve documentation
- Automatic differentiation without having to extract the underlying torch tensors.
- Python autoray compatible interface for usage as a backend in other libraries: I believe the biggest hurdle is slice support.
- Implement `__array_function__` protocol: see `NEP18 <https://numpy.org/neps/nep-0018-array-function-protocol.html>`__  