/*
 * File: quantt.cpp
 * Project: quantt
 * File Created: Thursday, 11th November 2021 11:17:35 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * Copyright (c) 2021 Alexandre Foley
 * All rights reserved
 */

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

#include "blockTensor/btensor.h"
#include <string_view>
#include <torch/extension.h>

/*
 * torch must also be already loaded in the module.
 * For the module to load we need to define TORCH_USE_RTLD_GLOBAL=YES environment variable, otherwise missing symbol
 * prevent loading Seems like that env_var is a stopgap solution, correct approach is to link with
 * PYTHON_TORCH_LIBRARIES or TORCH_PYTHON_LIBRARIES (which?) cmake variable.
 */
namespace py = pybind11;
namespace pybind11
{
namespace detail
{

template <>
struct type_caster<torch::ScalarType>
{
  public:
	PYBIND11_TYPE_CASTER(torch::ScalarType, _("dtype"));

	bool load(handle src, bool)
	{
		PyObject *source = src.ptr();
		// if(!source)
		//     return false;
		if (THPDtype_Check(source))
			value = reinterpret_cast<THPDtype *>(source)->scalar_type;
		else
			return false;
		return !PyErr_Occurred();
	}
	static handle cast(torch::ScalarType src, return_value_policy, handle)
	{
		return THPDtype_New(src, std::string(c10::scalarTypeToTypeMeta(src).name()));
	}
};
} // namespace detail
} // namespace pybind11
namespace
{

torch::Tensor test2()
{
	auto label_map = torch::zeros({100, 100}).to(torch::kFloat32);
	return label_map;
}

template <quantt::btensor (*F)(const quantt::btensor &, torch::TensorOptions)>
quantt::btensor like_factory_caster(const quantt::btensor &btens, torch::ScalarType dt, torch::Device dev,
                                    bool req_grad, bool pin_memory)
{
	auto opt = btens.options().dtype(dt).device(dev).requires_grad(req_grad).pinned_memory(pin_memory);
	return F(btens, opt);
}

template <quantt::btensor (*F)(const quantt::btensor::vec_list_t &, quantt::any_quantity, torch::TensorOptions)>
quantt::btensor factory_caster(const quantt::btensor::vec_list_t &shape, quantt::any_quantity sel, torch::ScalarType dt,
                               torch::Device dev, bool req_grad, bool pin_memory)
{
	auto opt = btens.options().dtype(dt).device(dev).requires_grad(req_grad).pinned_memory(pin_memory);
	return F(shape, sel, opt);
}

template <quantt::btensor (*F)(const quantt::btensor &, torch::TensorOptions)>
auto like_factory_binder(py::module &m, std::string function_name, std::string help_string = "")
{
	return m.def(function_name, &like_factory_caster<F>, help_string.c_str(), py::arg("btensor"), py::kw_only(),
	             py::arg("dtype") = torch::get_default_dtype_as_scalartype(),
	             py::arg("device") = torch::TensorOptions().device(),
	             py::arg("requires_grad") = torch::TensorOptions().requires_grad(),
	             py::arg("pin_memory") = torch::TensorOptions().pinned_memory());
}

template <quantt::btensor (*F)(const quantt::btensor &, torch::TensorOptions)>
auto factory_binder(py::module &m, std::string function_name, std::string help_string = "")
{
	return m.def(function_name, &factory_caster<F>, help_string.c_str(), py::arg("shape_specification"),
	             py::arg("selection_rule"), py::kw_only(), py::arg("dtype") = torch::get_default_dtype_as_scalartype(),
	             py::arg("device") = torch::TensorOptions().device(),
	             py::arg("requires_grad") = torch::TensorOptions().requires_grad(),
	             py::arg("pin_memory") = torch::TensorOptions().pinned_memory());
}

} // namespace
void init_conserved_qtt(py::module &m);
// The first argument needs to match the name of the *.so in the BUILD file.
PYBIND11_MODULE(QuantiT, m)
{
	using namespace quantt;
	auto PYTORCH = py::module_::import("torch");
	m.doc() = "quantiT";
	init_conserved_qtt(m);
	m.def("test2",

	      &test2, "test2");
	py::class_<quantt::btensor>(m, "btensor")
	    .def(py::init())
	    .def(py::init(
	             [](btensor::vec_list_t init_list, any_quantity sel_rule, torch::ScalarType dt, torch::Device dev,
	                bool req_grad, bool pin_memory)
	             {
		             auto opt =
		                 torch::TensorOptions().dtype(dt).device(dev).requires_grad(req_grad).pinned_memory(pin_memory);
		             auto out = btensor(init_list, sel_rule, opt);
		             btensor::throw_bad_tensor(out);
		             // No matter how the program is compiled, always check that the resulting tensor is ok. The
		             // constructor used by this function only checks in debug mode.
		             return out;
	             }),
	         py::arg("init_list"), py::arg("sel_rule"), py::kw_only(),
	         py::arg("dtype") = torch::get_default_dtype_as_scalartype(),
	         py::arg("device") = torch::TensorOptions().device(),
	         py::arg("requires_grad") = torch::TensorOptions().requires_grad(),
	         py::arg("pin_memory") = torch::TensorOptions().pinned_memory())
	    .def_property(
	        "dtype", [](const btensor &self) { return c10::typeMetaToScalarType(self.options().dtype()); },
	        [](btensor &self, torch::ScalarType dtype) { self.to(dtype); })
	    .def_property(
	        "device", [](const btensor &self) { return self.options().device(); },
	        [](btensor &self, torch::Device device) { self.to(device); })
	    .def_property(
	        "requires_grad", [](const btensor &self) { return self.options().requires_grad(); },
	        [](btensor &self, bool requires_grad) { self.to(self.options().requires_grad(requires_grad)); })
	    .def(
	        "block_at",
	        [](quantt::btensor &btens, const quantt::btensor::index_list &key)
	        {
		        try
		        {
			        return btens.block_at(key);
		        }
		        catch (const std::out_of_range &)
		        {
			        throw py::key_error(fmt::format("key '{}' does not exist", key));
		        }
	        },
	        py::keep_alive<0, 1>())
	    .def(
	        "block",
	        [](quantt::btensor &btens, const quantt::btensor::index_list &key)
	        {
		        try
		        {
			        return btens.block(key);
		        }
		        catch (const std::invalid_argument &exc)
		        {
			        throw py::key_error(exc.what());
		        }
	        },
	        py::keep_alive<0, 1>())
	    .def(
	        "__iter__", [](const quantt::btensor &btens) { return py::make_key_iterator(btens.begin(), btens.end()); },
	        py::keep_alive<0, 1>())
	    .def(
	        "block_quantities",
	        [](const btensor &self, const quantt::btensor::index_list &block_index)
	        {
		        auto view = self.block_quantities(block_index);
		        return py::make_iterator(view.begin(), view.end());
	        },
	        py::keep_alive<0, 1>())
	    .def(
	        "block_sizes",
	        [](const btensor &self, const quantt::btensor::index_list &block_index)
	        {
		        auto view = self.block_sizes(block_index);
		        return py::make_iterator(view.begin(), view.end());
	        },
	        py::keep_alive<0, 1>())
	    .def("sizes", &quantt::btensor::sizes)
	    .def("dim", &btensor::dim)
	    .def("__repr__", [](const quantt::btensor &val) { return fmt::format("{}", val); });

	factory_binder<&quantt::rand>(m, "rand");
	like_factory_binder<&quantt::rand_like>(m, "rand_like");
	factory_binder<&quantt::sparse_zeros>(m, "sparse_zeros");
	like_factory_binder<&quantt::sparse_zeros_like>(m,"sparse_zeros_like");
	factory_binder<&quantt::zeros>(m, "zeros");
	like_factory_binder<&quantt::zeros_like>(m,"zeros_like");
	factory_binder<&quantt::ones>(m,"ones");
	like_factory_binder<&quantt::ones_like>(m,"ones_like");
	m.def("full", &quantt::full);
}