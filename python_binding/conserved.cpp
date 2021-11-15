/*
 * File: conserved.cpp
 * Project: quantt
 * File Created: Monday, 15th November 2021 10:26:52 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * Copyright (c) 2021 Alexandre Foley
 * All rights reserved
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
namespace py = pybind11;

#include "Conserved/quantity.h"
using namespace quantt::conserved;


void init_conserved_qtt(py::module &m)
{
	auto conserved_sub = m.def_submodule("conserved");
	py::class_<quantt::conserved::Z>(conserved_sub, "zqtt") // I don't really want to expose those type... but pratice!
	.def(py::init<int16_t>())
	.def(py::self * py::self)
	.def(py::self *= py::self)
	.def("__repr__", [](const Z& val){return fmt::format("{}",val);})
	.def("inv",&Z::inverse)
	.def("inv_",&Z::inverse_)
	.def(py::self == py::self)
	.def(py::self != py::self)
	.def(py::self < py::self)
	.def(py::self > py::self);
}
