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
	auto opt = torch::dtype(dt).device(dev).requires_grad(req_grad).pinned_memory(pin_memory);
	return F(shape, sel, opt);
}

template <class... Args>
struct binder
{
	template<class ARGS>
	static c10::TensorOptions generate_option(const quantt::btensor& opt_ten, ARGS...)
	{
		return opt_ten.options();
	}
	static c10::TensorOptions generate_option(const quantt::btensor& opt_ten)
	{
		return opt_ten.options();
	}
	static auto bind(quantt::btensor (*f)(Args..., c10::TensorOptions))
	{
		using FirstEntityType = std::tuple_element_t<0, std::tuple<Args...>>;

		if constexpr (std::is_same_v<quantt::remove_cvref_t<FirstEntityType>, quantt::btensor>)
		{
			return [f](Args... args, torch::optional<torch::ScalarType> dt, torch::optional<torch::Device> dev, torch::optional<bool> req_grad, torch::optional<bool> pin_memory)
			{
				auto opt = generate_option(args...).dtype(dt).device(dev).requires_grad(req_grad).pinned_memory(pin_memory);
				return f(args..., opt);
			};
		}
		else
		{
			return [f](Args... args, torch::optional<torch::ScalarType> dt, torch::optional<torch::Device> dev, torch::optional<bool> req_grad, torch::optional<bool> pin_memory)
			{
				auto opt = torch::TensorOptions().dtype(dt).device(dev).requires_grad(req_grad).pinned_memory(pin_memory);
				return f(args..., opt);
			};
		}
	}
};

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

	m.def("sparse_zeros", binder<const btensor::vec_list_t &, any_quantity>::bind(&quantt::zeros),
	      "Generate an empty block tensor",
		  py::arg("shape_specification"), py::arg("selection_rule"), py::kw_only(),
	      py::arg("dtype") ,
	      py::arg("device") ,
	      py::arg("requires_grad"),
	      py::arg("pin_memory"));
	m.def("sparse_zeros_like", binder<const btensor &>::bind(&quantt::zeros_like),
	      "Generate an empty block tensor with the same shape and selection rule",
	      py::arg("shape_tensor"), py::kw_only(),
	      py::arg("dtype"),
	      py::arg("device"),
	      py::arg("requires_grad") ,
	      py::arg("pin_memory"));
	m.def("zeros", binder<const btensor::vec_list_t &, any_quantity>::bind(&quantt::zeros),
	      "Generate a block tensor with every permited block filled with zeros",
		  py::arg("shape_specification"), py::arg("selection_rule"), py::kw_only(),
	      py::arg("dtype") ,
	      py::arg("device") ,
	      py::arg("requires_grad"),
	      py::arg("pin_memory"));
	m.def("zeros_like", binder<const btensor &>::bind(&quantt::zeros_like),
	      "Generate a block tensor with the same shape and selection rule as the input and with every permited block "
	      "filled with zeros",
	      py::arg("shape_tensor"), py::kw_only(),
	      py::arg("dtype"),
	      py::arg("device"),
	      py::arg("requires_grad") ,
	      py::arg("pin_memory"));
	m.def("ones", binder<const btensor::vec_list_t &, any_quantity>::bind(&quantt::ones),
	      "Generate a block tensor with every permited block filled with ones",
		  py::arg("shape_specification"), py::arg("selection_rule"), py::kw_only(),
	      py::arg("dtype") ,
	      py::arg("device") ,
	      py::arg("requires_grad"),
	      py::arg("pin_memory"));
	m.def("ones_like", binder<const btensor &>::bind(&quantt::ones_like),
	      "Generate a block tensor with the same shape and selection rule as the input and with every permited block "
	      "filled with ones",
	      py::arg("shape_tensor"), py::kw_only(),
	      py::arg("dtype"),
	      py::arg("device"),
	      py::arg("requires_grad") ,
	      py::arg("pin_memory"));
	m.def("empty", binder<const btensor::vec_list_t &, any_quantity>::bind(&quantt::empty),
	      "Generate a block tensor with every permited block filled with uninitialized data",
		  py::arg("shape_specification"), py::arg("selection_rule"), py::kw_only(),
	      py::arg("dtype") ,
	      py::arg("device") ,
	      py::arg("requires_grad"),
	      py::arg("pin_memory"));
	m.def("empty_like", binder<const btensor &>::bind(&quantt::empty_like),
	      "Generate a block tensor with the same shape and selection rule as the input and with every permited block "
	      "filled with uninitialized data",
	      py::arg("shape_tensor"), py::kw_only(),
	      py::arg("dtype"),
	      py::arg("device"),
	      py::arg("requires_grad") ,
	      py::arg("pin_memory"));
	m.def("rand", binder<const btensor::vec_list_t &, any_quantity>::bind(&quantt::rand),
	      "Generate a block tensor with every permited block filled with random values",
		  py::arg("shape_specification"), py::arg("selection_rule"), py::kw_only(),
	      py::arg("dtype") ,
	      py::arg("device") ,
	      py::arg("requires_grad"),
	      py::arg("pin_memory"));
	m.def("rand_like", binder<const btensor &>::bind(&quantt::rand_like),
	      "Generate a block tensor with the same shape and selection rule as the input and with every permited block "
	      "filled with random value",
	      py::arg("shape_tensor"), py::kw_only(),
	      py::arg("dtype"),
	      py::arg("device"),
	      py::arg("requires_grad") ,
	      py::arg("pin_memory"));
	m.def("full", binder<const btensor::vec_list_t &, any_quantity, btensor::Scalar>::bind(&quantt::full),
	      "Generate a block tensor with every permited block filled with the specified value",
	      py::arg("shape_specification"), py::arg("selection_rule"), py::arg("fill_value"), py::kw_only(),
	      py::arg("dtype") ,
	      py::arg("device") ,
	      py::arg("requires_grad"),
	      py::arg("pin_memory"));
	m.def("full_like", binder<const btensor &, btensor::Scalar>::bind(&quantt::full_like),
	      "Generate a block tensor with the same shape and selection rule as the input and with every permited block "
	      "filled with the specified value",
	      py::arg("shape_tensor"), py::arg("fill_value"), py::kw_only(),
	      py::arg("dtype"),
	      py::arg("device"),
	      py::arg("requires_grad") ,
	      py::arg("pin_memory"));
	m.def("randint", binder<const btensor::vec_list_t &, any_quantity, int64_t,int64_t>::bind(&quantt::randint),
	      "Generate a block tensor with every permited block filled with random integers",
	      py::arg("shape_specification"), py::arg("selection_rule"), py::arg("low"),py::arg("high"), py::kw_only(),
	      py::arg("dtype") ,
	      py::arg("device") ,
	      py::arg("requires_grad"),
	      py::arg("pin_memory"));
	m.def("randint_like", binder<const btensor &, int64_t,int64_t>::bind(&quantt::randint_like),
	      "Generate a block tensor with the same shape and selection rule as the input and with every permited block "
	      "filled with random integers",
	      py::arg("shape_tensor"), py::arg("low"),py::arg("high"), py::kw_only(),
	      py::arg("dtype"),
	      py::arg("device"),
	      py::arg("requires_grad") ,
	      py::arg("pin_memory"));
	m.def("randint", binder<const btensor::vec_list_t &, any_quantity, int64_t>::bind(&quantt::randint),
	      "Generate a block tensor with every permited block filled with random integers",
	      py::arg("shape_specification"), py::arg("selection_rule"), py::arg("high"), py::kw_only(),
	      py::arg("dtype") ,
	      py::arg("device") ,
	      py::arg("requires_grad"),
	      py::arg("pin_memory"));
	m.def("randint_like", binder<const btensor &,int64_t>::bind(&quantt::randint_like),
	      "Generate a block tensor with the same shape and selection rule as the input and with every permited block "
	      "filled with random integers",
	      py::arg("shape_tensor"),py::arg("high"), py::kw_only(),
	      py::arg("dtype"),
	      py::arg("device"),
	      py::arg("requires_grad") ,
	      py::arg("pin_memory"));
	m.def("randn", binder<const btensor::vec_list_t &, any_quantity>::bind(&quantt::randn),
	      "Generate a block tensor with every permited block filled with normally distributed random values",
		  py::arg("shape_specification"), py::arg("selection_rule"), py::kw_only(),
	      py::arg("dtype") ,
	      py::arg("device") ,
	      py::arg("requires_grad"),
	      py::arg("pin_memory"));
	m.def("randn_like", binder<const btensor &>::bind(&quantt::randn_like),
	      "Generate a block tensor with the same shape and selection rule as the input and with every permited block "
	      "filled with normally distributed random value",
	      py::arg("shape_tensor"), py::kw_only(),
	      py::arg("dtype"),
	      py::arg("device"),
	      py::arg("requires_grad") ,
	      py::arg("pin_memory"));
	m.def("from_torch_tensor", binder<const btensor::vec_list_t &, any_quantity,const torch::Tensor& , torch::Scalar >::bind(&quantt::from_basic_tensor),
	      "Generate a block tensor from a torch tensor. Overall size must match with the specified shape, forbidden elements and elements bellow the cutoff are ignored.",
		  py::arg("shape_specification"), py::arg("selection_rule"),py::arg("values"), py::arg("cutoff") = 1e-16, py::kw_only(),
	      py::arg("dtype") ,
	      py::arg("device") ,
	      py::arg("requires_grad"),
	      py::arg("pin_memory"));
	m.def("from_torch_tensor_like", binder<const btensor &,const torch::Tensor&, const torch::Scalar>::bind(&quantt::from_basic_tensor_like),
	      "Generate a block tensor from a torch tensor. Overall size must match with the shape tensor, forbidden elements and elements bellow the cutoff are ignored.",
	      "filled with normally distributed random value",
	      py::arg("shape_tensor"),py::arg("values"), py::arg("cutoff") = 1e-16, py::kw_only(),
	      py::arg("dtype"),
	      py::arg("device"),
	      py::arg("requires_grad") ,
	      py::arg("pin_memory"));

}