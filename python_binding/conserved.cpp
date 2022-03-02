/*
 * File: conserved.cpp
 * Project: QuantiT
 * File Created: Monday, 15th November 2021 10:26:52 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * Copyright (c) 2021 Alexandre Foley
 * Licensed under GPL v3
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
namespace py = pybind11;

#include "Conserved/Composite/cquantity.h"
#include "Conserved/Composite/quantity_vector.h"

using namespace quantit::conserved;
void init_conserved_qtt(py::module &m)
{
	auto conserved_sub = m.def_submodule("conserved");

	//I might instead create factory function for any_quantity that initialize with these instead of exposing them as python types.

	py::class_<quantit::any_quantity>(conserved_sub,"quantity",py::is_final())
	.def(py::init<>())
	.def(py::self * py::self)
	.def(py::self *= py::self)
	.def("__repr__", [](const quantit::any_quantity& val){return fmt::format("{}",val);})
	.def("inv",[](const quantit::any_quantity& self){return self.inverse();})
	.def("inv_",&quantit::any_quantity::inverse_)
	.def(py::self == py::self)
	.def(py::self != py::self)
	.def(py::self < py::self)
	.def(py::self > py::self);

	conserved_sub.def("Z", [](int x){return quantit::any_quantity(Z(x));});
	conserved_sub.def("ZC2", [](int x,int z){return quantit::any_quantity(Z(x),C<2>(z));});
	conserved_sub.def("ZC3", [](int x,int z){return quantit::any_quantity(Z(x),C<3>(z));});
	conserved_sub.def("ZC4", [](int x,int z){return quantit::any_quantity(Z(x),C<4>(z));});
	conserved_sub.def("ZC5", [](int x,int z){return quantit::any_quantity(Z(x),C<5>(z));});
	conserved_sub.def("ZC6", [](int x,int z){return quantit::any_quantity(Z(x),C<6>(z));});
	conserved_sub.def("ZZ", [](int x,int y){return quantit::any_quantity(Z(x),Z(y));});
	conserved_sub.def("ZZC2", [](int x,int y,int z){return quantit::any_quantity(Z(x),Z(y),C<2>(z));});
	conserved_sub.def("ZZC3", [](int x,int y,int z){return quantit::any_quantity(Z(x),Z(y),C<3>(z));});
	conserved_sub.def("ZZC4", [](int x,int y,int z){return quantit::any_quantity(Z(x),Z(y),C<4>(z));});
	conserved_sub.def("ZZC5", [](int x,int y,int z){return quantit::any_quantity(Z(x),Z(y),C<5>(z));});
	conserved_sub.def("ZZC6", [](int x,int y,int z){return quantit::any_quantity(Z(x),Z(y),C<6>(z));});
	conserved_sub.def("C2C2", [](int x,int z){return quantit::any_quantity(C<2>(x),C<2>(z));});
	conserved_sub.def("C2C3", [](int x,int z){return quantit::any_quantity(C<2>(x),C<3>(z));});
	conserved_sub.def("C2C4", [](int x,int z){return quantit::any_quantity(C<2>(x),C<4>(z));});
	conserved_sub.def("C2C5", [](int x,int z){return quantit::any_quantity(C<2>(x),C<5>(z));});
	conserved_sub.def("C2C6", [](int x,int z){return quantit::any_quantity(C<2>(x),C<6>(z));});
	conserved_sub.def("C2", [](int z){return quantit::any_quantity(C<2>(z));});
	conserved_sub.def("C3", [](int z){return quantit::any_quantity(C<3>(z));});
	conserved_sub.def("C4", [](int z){return quantit::any_quantity(C<4>(z));});
	conserved_sub.def("C5", [](int z){return quantit::any_quantity(C<5>(z));});
	conserved_sub.def("C6", [](int z){return quantit::any_quantity(C<6>(z));}) ;

}
