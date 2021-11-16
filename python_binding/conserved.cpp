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

#include "Conserved/Composite/cquantity.h"


template<class X>
using to_int_16t = int16_t; 


template<class... QTT> void pywrap_quanttquantity(py::module &m, std::string name)
{
	using this_type = quantt::quantity<QTT...>;
	py::class_< this_type >(m, name.c_str() ,py::is_final()) 
	.def(py::init< to_int_16t<QTT>... >())// initialize from as many int16_t as the template has arguements.
	.def(py::self * py::self)
	.def(py::self *= py::self)
	.def("__repr__", [](const this_type& val){return fmt::format("{}",val);})
	.def("inv",&this_type::inverse)
	.def("inv_",&this_type::inverse_)
	.def(py::self == py::self)
	.def(py::self != py::self)
	.def(py::self < py::self)
	.def(py::self > py::self);
}

using namespace quantt::conserved;
void init_conserved_qtt(py::module &m)
{
	auto conserved_sub = m.def_submodule("conserved");

	pywrap_quanttquantity<Z>(conserved_sub,"Z");
	pywrap_quanttquantity<Z,Z>(conserved_sub,"ZZ");
	pywrap_quanttquantity<Z,Z,C<2> >(conserved_sub,"ZZC2");
	pywrap_quanttquantity<Z,Z,C<3> >(conserved_sub,"ZZC3");
	pywrap_quanttquantity<Z,Z,C<4> >(conserved_sub,"ZZC4");
	pywrap_quanttquantity<Z,Z,C<6> >(conserved_sub,"ZZC6");
	pywrap_quanttquantity<C<2> >(conserved_sub,"C2");
	pywrap_quanttquantity<C<3> >(conserved_sub,"C3");
	pywrap_quanttquantity<C<4> >(conserved_sub,"C4");
	pywrap_quanttquantity<C<6> >(conserved_sub,"C6");
	pywrap_quanttquantity<C<2>,C<2> >(conserved_sub,"C2C2");
	pywrap_quanttquantity<C<2>,C<3> >(conserved_sub,"C2C3");
	pywrap_quanttquantity<C<2>,C<4> >(conserved_sub,"C2C4");
	pywrap_quanttquantity<C<2>,C<6> >(conserved_sub,"C2C6");

	py::class_<quantt::any_quantity>(conserved_sub,"quantity",py::is_final())
	.def(py::init<>())
	.def(py::init([](const quantt::quantity<Z>& x) {return quantt::any_quantity(x);}))
	.def(py::init([](const quantt::quantity<Z,Z>& x) {return quantt::any_quantity(x);}))
	.def(py::init([](const quantt::quantity<Z,Z,C<2> >& x) {return quantt::any_quantity(x);}))
	.def(py::init([](const quantt::quantity<Z,Z,C<3> >& x) {return quantt::any_quantity(x);}))
	.def(py::init([](const quantt::quantity<Z,Z,C<4> >& x) {return quantt::any_quantity(x);}))
	.def(py::init([](const quantt::quantity<Z,Z,C<6> >& x) {return quantt::any_quantity(x);}))
	.def(py::init([](const quantt::quantity<C<2>,C<2> >& x) {return quantt::any_quantity(x);}))
	.def(py::init([](const quantt::quantity<C<2>,C<3> >& x) {return quantt::any_quantity(x);}))
	.def(py::init([](const quantt::quantity<C<2>,C<4> >& x) {return quantt::any_quantity(x);}))
	.def(py::init([](const quantt::quantity<C<2>,C<6> >& x) {return quantt::any_quantity(x);}))
	.def(py::init([](const quantt::quantity<C<2> >& x) {return quantt::any_quantity(x);}))
	.def(py::init([](const quantt::quantity<C<3> >& x) {return quantt::any_quantity(x);}))
	.def(py::init([](const quantt::quantity<C<4> >& x) {return quantt::any_quantity(x);}))
	.def(py::init([](const quantt::quantity<C<6> >& x) {return quantt::any_quantity(x);}))
	.def(py::self * py::self)
	.def(py::self *= py::self)
	.def("__repr__", [](const quantt::any_quantity& val){return fmt::format("{}",val);})
	.def("inv",[](const quantt::any_quantity& self){return self.inverse();})
	.def("inv_",&quantt::any_quantity::inverse_)
	.def(py::self == py::self)
	.def(py::self != py::self)
	.def(py::self < py::self)
	.def(py::self > py::self);
}
