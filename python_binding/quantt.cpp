/*
 * File: QuantiT.cpp
 * Project: QuantiT
 * File Created: Thursday, 11th November 2021 11:17:35 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * Copyright (c) 2021 Alexandre Foley
 * Licensed under GPL v3
 */

#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

#include "blockTensor/btensor.h"
#include "utilities.h"
#include <string_view>
#include <torch/csrc/MemoryFormat.h>
#include <torch/extension.h>

using namespace utils;
using namespace quantit;
/*
 * torch must also be already loaded in the module.
 * For the module to load we need to define TORCH_USE_RTLD_GLOBAL=YES environment variable, otherwise missing symbol
 * prevent loading Seems like that env_var is a stopgap solution, correct approach is to link with
 * PYTHON_TORCH_LIBRARIES or TORCH_PYTHON_LIBRARIES (which?) cmake variable.
 */
namespace py = pybind11;

/**
 * @brief register the funtion in a uniform call syntax: register both a free standing function and a class method
 *
 * This definition is made lazily (by me, i'm the lazy bum) and forces the user to supply a docstring and argument
 * names.
 * A fancier function with more template black magic could strip the first py::arg for the class method
 * registration, but it's not worth the effort as argument name and docstring should be supplied anyway.
 *
 * @tparam T
 * @tparam F
 * @tparam Args
 * @param m
 * @param c
 * @param name
 * @param f
 * @param docstring
 * @param first_arg
 * @param args
 */
template <class T, class F, class... Args>
void uniform_call(py::module_ &m, py::class_<T> &c, const char *name, F &&f, const char *docstring, py::arg first_arg,
                  Args &&...args)
{
	m.def(name, f, docstring, first_arg, args...);
	// class method don't accept a name for the self arguement, as it is implicit.
	c.def(name, f, docstring, std::forward<Args>(args)...);
}

void init_conserved_qtt(py::module &m);
void init_linalg_qtt(py::module &m);
void init_networks(py::module &m);
void init_operators(py::module &m);
void init_algorithms(py::module &m);
class block_helper
{
  public:
	btensor &owner;
	block_helper(btensor &_owner) : owner(_owner) {}
	torch::Tensor &access(const btensor::index_list &block_index) { return owner.block(block_index); }
	torch::Tensor &access_at(const btensor::index_list &block_index) { return owner.block_at(block_index); }
};
// The first argument needs to match the name of the *.so in the BUILD file.
PYBIND11_MODULE(quantit, m)
{
	constexpr auto pol_internal_ref = py::return_value_policy::reference_internal;
	auto x = torch::optional<int>();
	using namespace quantit;
	auto PYTORCH = py::module_::import("torch");
	m.doc() = "QuantiT";
	init_conserved_qtt(m);
	init_linalg_qtt(m);
	init_networks(m);
	init_operators(m);
	init_algorithms(m);
	auto pyblocklist =
	    py::class_<block_helper>(m, "block_list")
	        .def("__setitem__",
	             [](block_helper &self, const btensor::index_list ind, const torch::Tensor &val)
	             {
		             if (ind.size() != self.owner.dim())
			             throw py::key_error("key dimension incompatible with the tensor");
		             for (int i = 0; i < self.owner.dim(); ++i)
		             {
			             if (ind[i] >= self.owner.section_numbers()[i])
				             throw py::key_error(fmt::format(
				                 "key value {} for dimension {} is greater or equal to the size of that dimension {}.",
				                 ind[i], i, self.owner.section_numbers()[i]));
		             }
		             int i = 0;
		             auto val_size = val.sizes();
		             for (const auto &s : self.owner.block_sizes(ind))
		             {
			             if (val_size[i] != s)
				             throw py::value_error(fmt::format(
				                 "input tensor has wrong size {} at dim {}. {} was expected", val_size[i], i, s));
			             ++i;
		             }
		             // And set the options to match the rest of the btensor.
		             self.access(ind) = val.to(self.owner.options());
	             })
	        .def("__getitem__", &block_helper::access_at);
	auto pybtensor =
	    py::class_<quantit::btensor>(m, "btensor")
	        .def(py::init())
	        .def(py::init(
	                 [](btensor::vec_list_t init_list, any_quantity sel_rule, torch::optional<torch::ScalarType> dt,
	                    torch::optional<torch::Device> dev, torch::optional<bool> req_grad,
	                    torch::optional<bool> pin_memory)
	                 {
		                 auto opt = torch::TensorOptions().dtype(dt).device(dev).requires_grad(req_grad).pinned_memory(
		                     pin_memory);
		                 auto out = btensor(init_list, sel_rule, opt);
		                 btensor::throw_bad_tensor(out);
		                 // No matter how the program is compiled, always check that the resulting tensor is ok. The
		                 // constructor used by this function only checks in debug mode.
		                 return out;
	                 }),
	             py::arg("init_list"), py::arg("sel_rule"), py::kw_only(), py::arg("dtype") = opt<stype>(),
	             py::arg("device") = opt<tdev>(), py::arg("requires_grad") = opt<bool>(),
	             py::arg("pin_memory") = opt<bool>())
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
	            "__iter__", [](quantit::btensor &btens) { return py::make_key_iterator(btens.begin(), btens.end()); },
	            py::keep_alive<0, 1>())
	        .def(py::self + py::self)
	        .def(py::self - py::self)
	        .def(py::self / py::self)
	        .def(py::self * py::self)
	        .def(py::self += py::self)
	        .def(py::self -= py::self)
	        .def(py::self /= py::self)
	        .def(py::self *= py::self)
	        .def(- py::self)
	        .def("__repr__", [](const quantit::btensor &val) { return fmt::format("{}", val); });
	pybtensor.def("__mul__", wrap_scalar([](const btensor &A, c10::Scalar B) { return A.mul(B); }));
	pybtensor.def("__mul__", wrap_scalar([](c10::Scalar B, const btensor &A) { return A.mul(B); }));
	pybtensor.def("__div__", wrap_scalar([](const btensor &A, c10::Scalar B) { return A.div(B); }));
	pybtensor.def("__div__", wrap_scalar([](c10::Scalar B, const btensor &A) { return B / A; }));
	// btensor t() const {return transpose(dim()-1,dim()-2);}
	pybtensor.def("t", &btensor::t, "permute the last two dimension of the tensor");
	// btensor& t_() {return transpose_(dim()-1,dim()-2);}
	pybtensor.def("t_", &btensor::t_, "permute the last two dimension of the tensor",
	              py::return_value_policy::reference_internal);
	// block, must become a property of btensor, expose the underlying blocklist, which would have a very limited (and
	// safe) interface To expose it to python without making it public, we have to do some funny stuff.
	pybtensor.def_property_readonly(
	    "blocks", [](quantit::btensor &self) { return block_helper(self); }, "blocks contained within the btensor",
	    py::return_value_policy::reference_internal);
	// uniform_call(m,pybtensor, "add" ,[](const btensor& self, const btensor& other, py::object){});
	// J'ai besoin d'une façon de convertir les nombre python vers des btensor::Scalar et vis-versa. Idéalement,
	// btensor::Scalar n'est pas explicitement exposé en python. toute fonction qui retourne un btensor::Scalar va faire
	// la conversion vers le bon type numérique python: wrap_scalar prenant en arguement une fonction avec des
	// c10::scalar dans sa signature encompli ceci.
	uniform_call(
	    m, pybtensor, "pow", [](const btensor &self, const btensor &x) { return self.pow(x); },
	    "apply element by element exponentiation", py::arg("self"), py::arg("exponent"));
	uniform_call(
	    m, pybtensor, "pow_", [](btensor &self, const btensor &x) { return self.pow_(x); },
	    "in-place application of element by element exponentiation", py::arg("self"), py::arg("exponent"),
	    py::return_value_policy::reference_internal);
	uniform_call(m, pybtensor, "pow", wrap_scalar([](const btensor &self, c10::Scalar x) { return self.pow(x); }),
	             "apply element by element exponentiation", py::arg("self"), py::arg("exponent"));
	uniform_call(m, pybtensor, "pow_", wrap_scalar([](btensor &self, c10::Scalar x) { return self.pow_(x); }),
	             "in-place application of element by element exponentiation", py::arg("self"), py::arg("exponent"),
	             py::return_value_policy::reference_internal);
	m.def("sparse_zeros", TOPT_binder<const btensor::vec_list_t &, any_quantity>::bind(&quantit::sparse_zeros),
	      "Generate an empty block tensor", py::arg("shape_specification"), py::arg("selection_rule"), py::kw_only(),
	      py::arg("dtype") = opt<stype>(), py::arg("device") = opt<tdev>(), py::arg("requires_grad") = opt<bool>(),
	      py::arg("pin_memory") = opt<bool>());
	m.def("sparse_zeros_like", TOPT_binder<const btensor &>::bind(&quantit::sparse_zeros_like),
	      "Generate an empty block tensor with the same shape and selection rule as the input btensor",
	      py::arg("shape_tensor"), py::kw_only(), py::arg("dtype") = opt<stype>(), py::arg("device") = opt<tdev>(),
	      py::arg("requires_grad") = opt<bool>(), py::arg("pin_memory") = opt<bool>());
	m.def("zeros", TOPT_binder<const btensor::vec_list_t &, any_quantity>::bind(&quantit::zeros),
	      "Generate a block tensor with every permited block filled with zeros", py::arg("shape_specification"),
	      py::arg("selection_rule"), py::kw_only(), py::arg("dtype") = opt<stype>(), py::arg("device") = opt<tdev>(),
	      py::arg("requires_grad") = opt<bool>(), py::arg("pin_memory") = opt<bool>());
	m.def("zeros_like", TOPT_binder<const btensor &>::bind(&quantit::zeros_like),
	      "Generate a block tensor with the same shape and selection rule as the input btensor and with every permited "
	      "block "
	      "filled with zeros",
	      py::arg("shape_tensor"), py::kw_only(), py::arg("dtype") = opt<stype>(), py::arg("device") = opt<tdev>(),
	      py::arg("requires_grad") = opt<bool>(), py::arg("pin_memory") = opt<bool>());
	m.def("ones", TOPT_binder<const btensor::vec_list_t &, any_quantity>::bind(&quantit::ones),
	      "Generate a block tensor with every permited block filled with ones", py::arg("shape_specification"),
	      py::arg("selection_rule"), py::kw_only(), py::arg("dtype") = opt<stype>(), py::arg("device") = opt<tdev>(),
	      py::arg("requires_grad") = opt<bool>(), py::arg("pin_memory") = opt<bool>());
	m.def("ones_like", TOPT_binder<const btensor &>::bind(&quantit::ones_like),
	      "Generate a block tensor with the same shape and selection rule as the input btensor and with every permited "
	      "block "
	      "filled with ones",
	      py::arg("shape_tensor"), py::kw_only(), py::arg("dtype") = opt<stype>(), py::arg("device") = opt<tdev>(),
	      py::arg("requires_grad") = opt<bool>(), py::arg("pin_memory") = opt<bool>());
	m.def("empty", TOPT_binder<const btensor::vec_list_t &, any_quantity>::bind(&quantit::empty),
	      "Generate a block tensor with every permited block filled with uninitialized data",
	      py::arg("shape_specification"), py::arg("selection_rule"), py::kw_only(), py::arg("dtype") = opt<stype>(),
	      py::arg("device") = opt<tdev>(), py::arg("requires_grad") = opt<bool>(), py::arg("pin_memory") = opt<bool>());
	m.def("empty_like", TOPT_binder<const btensor &>::bind(&quantit::empty_like),
	      "Generate a block tensor with the same shape and selection rule as the input btensor and with every permited "
	      "block "
	      "filled with uninitialized data",
	      py::arg("shape_tensor"), py::kw_only(), py::arg("dtype") = opt<stype>(), py::arg("device") = opt<tdev>(),
	      py::arg("requires_grad") = opt<bool>(), py::arg("pin_memory") = opt<bool>());
	m.def("rand", TOPT_binder<const btensor::vec_list_t &, any_quantity>::bind(&quantit::rand),
	      "Generate a block tensor with every permited block filled with random values", py::arg("shape_specification"),
	      py::arg("selection_rule"), py::kw_only(), py::arg("dtype") = opt<stype>(), py::arg("device") = opt<tdev>(),
	      py::arg("requires_grad") = opt<bool>(), py::arg("pin_memory") = opt<bool>());
	m.def("rand_like", TOPT_binder<const btensor &>::bind(&quantit::rand_like),
	      "Generate a block tensor with the same shape and selection rule as the input btensor and with every permited "
	      "block "
	      "filled with random value",
	      py::arg("shape_tensor"), py::kw_only(), py::arg("dtype") = opt<stype>(), py::arg("device") = opt<tdev>(),
	      py::arg("requires_grad") = opt<bool>(), py::arg("pin_memory") = opt<bool>());
	m.def("full",
	      wrap_scalar(TOPT_binder<const btensor::vec_list_t &, any_quantity, btensor::Scalar>::bind(&quantit::full)),
	      "Generate a block tensor with every permited block filled with the specified value",
	      py::arg("shape_specification"), py::arg("selection_rule"), py::arg("fill_value"), py::kw_only(),
	      py::arg("dtype") = opt<stype>(), py::arg("device") = opt<tdev>(), py::arg("requires_grad") = opt<bool>(),
	      py::arg("pin_memory") = opt<bool>());
	m.def("full_like", wrap_scalar(TOPT_binder<const btensor &, btensor::Scalar>::bind(&quantit::full_like)),
	      "Generate a block tensor with the same shape and selection rule as the input btensor and with every permited "
	      "block "
	      "filled with the specified value",
	      py::arg("shape_tensor"), py::arg("fill_value"), py::kw_only(), py::arg("dtype") = opt<stype>(),
	      py::arg("device") = opt<tdev>(), py::arg("requires_grad") = opt<bool>(), py::arg("pin_memory") = opt<bool>());
	m.def("randint", TOPT_binder<const btensor::vec_list_t &, any_quantity, int64_t, int64_t>::bind(&quantit::randint),
	      "Generate a block tensor with every permited block filled with random integers",
	      py::arg("shape_specification"), py::arg("selection_rule"), py::arg("low"), py::arg("high"), py::kw_only(),
	      py::arg("dtype") = opt<stype>(), py::arg("device") = opt<tdev>(), py::arg("requires_grad") = opt<bool>(),
	      py::arg("pin_memory") = opt<bool>());
	m.def("randint_like", TOPT_binder<const btensor &, int64_t, int64_t>::bind(&quantit::randint_like),
	      "Generate a block tensor with the same shape and selection rule as the input btensor and with every permited "
	      "block "
	      "filled with random integers",
	      py::arg("shape_tensor"), py::arg("low"), py::arg("high"), py::kw_only(), py::arg("dtype") = opt<stype>(),
	      py::arg("device") = opt<tdev>(), py::arg("requires_grad") = opt<bool>(), py::arg("pin_memory") = opt<bool>());
	m.def("randint", TOPT_binder<const btensor::vec_list_t &, any_quantity, int64_t>::bind(&quantit::randint),
	      "Generate a block tensor with every permited block filled with random integers",
	      py::arg("shape_specification"), py::arg("selection_rule"), py::arg("high"), py::kw_only(),
	      py::arg("dtype") = opt<stype>(), py::arg("device") = opt<tdev>(), py::arg("requires_grad") = opt<bool>(),
	      py::arg("pin_memory") = opt<bool>());
	m.def("randint_like", TOPT_binder<const btensor &, int64_t>::bind(&quantit::randint_like),
	      "Generate a block tensor with the same shape and selection rule as the input btensor and with every permited "
	      "block "
	      "filled with random integers",
	      py::arg("shape_tensor"), py::arg("high"), py::kw_only(), py::arg("dtype") = opt<stype>(),
	      py::arg("device") = opt<tdev>(), py::arg("requires_grad") = opt<bool>(), py::arg("pin_memory") = opt<bool>());
	m.def("randn", TOPT_binder<const btensor::vec_list_t &, any_quantity>::bind(&quantit::randn),
	      "Generate a block tensor with every permited block filled with normally distributed random values",
	      py::arg("shape_specification"), py::arg("selection_rule"), py::kw_only(), py::arg("dtype") = opt<stype>(),
	      py::arg("device") = opt<tdev>(), py::arg("requires_grad") = opt<bool>(), py::arg("pin_memory") = opt<bool>());
	m.def("randn_like", TOPT_binder<const btensor &>::bind(&quantit::randn_like),
	      "Generate a block tensor with the same shape and selection rule as the input btensor and with every permited "
	      "block "
	      "filled with normally distributed random value",
	      py::arg("shape_tensor"), py::kw_only(), py::arg("dtype") = opt<stype>(), py::arg("device") = opt<tdev>(),
	      py::arg("requires_grad") = opt<bool>(), py::arg("pin_memory") = opt<bool>());
	m.def(
	    "from_torch_tensor",
	    wrap_scalar(TOPT_binder<const btensor::vec_list_t &, any_quantity, const torch::Tensor &, torch::Scalar>::bind(
	        &quantit::from_basic_tensor)),
	    "Generate a block tensor from a torch tensor. Overall size must match with the specified shape, forbidden "
	    "elements and elements with magnitude below the cutoff are ignored.",
	    py::arg("shape_specification"), py::arg("selection_rule"), py::arg("values"), py::arg("cutoff") = 1e-16,
	    py::kw_only(), py::arg("dtype") = opt<stype>(), py::arg("device") = opt<tdev>(),
	    py::arg("requires_grad") = opt<bool>(), py::arg("pin_memory") = opt<bool>());
	m.def("from_torch_tensor_like",
	      wrap_scalar(TOPT_binder<const btensor &, const torch::Tensor &, const torch::Scalar>::bind(
	          &quantit::from_basic_tensor_like)),
	      "Generate a block tensor from a torch tensor. Overall size must match with the shape tensor, forbidden "
	      "elements and elements with magnitude below the cutoff are ignored.",
	      py::arg("shape_tensor"), py::arg("values"), py::arg("cutoff") = 1e-16, py::kw_only(),
	      py::arg("dtype") = opt<stype>(), py::arg("device") = opt<tdev>(), py::arg("requires_grad") = opt<bool>(),
	      py::arg("pin_memory") = opt<bool>());

	// python implicitly define the other order for the operators with heterogenous types.
	//  inline btensor operator>(const btensor &A, btensor::Scalar other) { return greater(A, other); }
	pybtensor.def(py::self > py::self);
	// inline btensor operator>(const btensor &A, const btensor &other) { return greater(A, other); }
	// inline btensor operator>(btensor::Scalar A, const btensor &other) { return greater(A, other); }
	pybtensor.def("__gt__", wrap_scalar([](const btensor &a, c10::Scalar b) { return a > b; }));
	// inline btensor operator<(const btensor &A, btensor::Scalar other) { return less(A, other); }
	pybtensor.def(py::self < py::self);
	// inline btensor operator<(const btensor &A, const btensor &other) { return less(A, other); }
	pybtensor.def("__lt__", wrap_scalar([](const btensor &a, c10::Scalar b) { return a < b; }));
	// inline btensor operator<(btensor::Scalar A, const btensor &other) { return less(A, other); }
	// inline btensor operator>=(const btensor &A, btensor::Scalar other) { return ge(A, other); }
	pybtensor.def(py::self >= py::self);
	// inline btensor operator>=(const btensor &A, const btensor &other) { return ge(A, other); }
	// inline btensor operator>=(btensor::Scalar A, const btensor &other) { return ge(A, other); }
	pybtensor.def("__ge__", wrap_scalar([](const btensor &a, c10::Scalar b) { return a >= b; }));
	// inline btensor operator<=(const btensor &A, btensor::Scalar other) { return le(A, other); }
	pybtensor.def(py::self <= py::self);
	// inline btensor operator<=(const btensor &A, const btensor &other) { return le(A, other); }
	// inline btensor operator<=(btensor::Scalar A, const btensor &other) { return le(A, other); }
	pybtensor.def("__le__", wrap_scalar([](const btensor &a, c10::Scalar b) { return a <= b; }));
	// inline btensor operator==(const btensor &A, btensor::Scalar other) { return eq(A, other); }
	pybtensor.def(py::self == py::self);
	// inline btensor operator==(const btensor &A, const btensor &other) { return eq(A, other); }
	// inline btensor operator==(btensor::Scalar A, const btensor &other) { return eq(A, other); }
	pybtensor.def("__eq__", wrap_scalar([](const btensor &a, c10::Scalar b) { return a == b; }));
	// inline btensor operator!=(const btensor &A, btensor::Scalar other) { return not_equal(A, other); }
	pybtensor.def(py::self != py::self);
	// inline btensor operator!=(const btensor &A, const btensor &other) { return not_equal(A, other); }
	// inline btensor operator!=(btensor::Scalar A, const btensor &other) { return not_equal(A, other); }
	pybtensor.def("__ne__", wrap_scalar([](const btensor &a, c10::Scalar b) { return a != b; }));

	// uniform calls
	//  const auto &get_cvals() const { return c_vals; }
	uniform_call(
	    m, pybtensor, "get_cvals", [](const btensor &self) { return self.get_cvals(); },
	    "return the list of all conserved quantities, ordered by dimension.", py::arg("self"));
	//  btensor::Scalar item() const;
	uniform_call(m, pybtensor, "item", wrap_scalar([](const btensor &self) { return self.item(); }),
	             "return the raw value contained in the tensor of it is rank 0", py::arg("self"));
	// block_quantities
	class deref_any_block_iter : public btensor::const_block_qtt_iter
	{
	  public:
		using btensor::const_block_qtt_iter::const_block_qtt_iter;
		deref_any_block_iter(const btensor::const_block_qtt_iter it) : btensor::const_block_qtt_iter(it) {}

		any_quantity operator*() { return btensor::const_block_qtt_iter::operator*(); }
		// any_quantity_cref do not exist in python, only any_quantity.
	};
	uniform_call(
	    m, pybtensor, "block_quantities",
	    [](const btensor &self, const quantit::btensor::index_list &block_index)
	    {
		    auto view = self.block_quantities(block_index);
		    return py::make_iterator<py::return_value_policy::move>(deref_any_block_iter(view.begin()),
		                                                            deref_any_block_iter(view.end()));
	    },
	    "return the list conserved quantity of each dimension of the block with the given index", py::arg("self"),
	    py::arg("block_index"), py::keep_alive<0, 1>());
	class deref_size_block_iter : public btensor::const_block_size_iter
	{
	  public:
		using btensor::const_block_size_iter::const_block_size_iter;
		deref_size_block_iter(const btensor::const_block_size_iter it) : btensor::const_block_size_iter(it) {}

		int operator*() { return btensor::const_block_size_iter::operator*(); }
	};
	uniform_call(
	    m, pybtensor, "block_sizes",
	    [](const btensor &self, const quantit::btensor::index_list &block_index)
	    {
		    auto view = self.block_sizes(block_index);
		    return py::make_iterator<py::return_value_policy::move>(view.begin(), view.end());
	    },
	    "return the list sizes of each dimension of the block with the given index", py::arg("self"),
	    py::arg("block_index"), py::keep_alive<0, 1>());
	// std::tuple<index_list::const_iterator, index_list::const_iterator> section_sizes(size_t dim) const;
	uniform_call(
	    m, pybtensor, "sections_size",
	    [](const btensor &self, size_t dim)
	    {
		    auto its = self.section_sizes(dim);
		    return py::make_iterator(std::get<0>(its), std::get<1>(its));
	    },
	    "return an iterator on the size of the section of a dimension", py::arg("self"), py::arg("dim"),
	    py::return_value_policy::reference_internal);
	class deref_any_vector_iter : public any_quantity_vector::const_iterator
	{
	  public:
		using any_quantity_vector::const_iterator::const_iterator;
		deref_any_vector_iter(const any_quantity_vector::const_iterator it) : any_quantity_vector::const_iterator(it) {}

		any_quantity operator*() { return any_quantity_vector::const_iterator::operator*(); }
		// any_quantity_cref do not exist in python, only any_quantity.
	};
	// std::tuple<any_quantity_vector::const_iterator, any_quantity_vector::const_iterator> section_cqtts(size_t dim)
	// const; sizes
	uniform_call(
	    m, pybtensor, "sections_quantity",
	    [](const btensor &self, size_t dim)
	    {
		    auto its = self.section_cqtts(dim);
		    return py::make_iterator<py::return_value_policy::move>(deref_any_vector_iter(std::get<0>(its)),
		                                                            deref_any_vector_iter(std::get<1>(its)));
	    },
	    "return the conserved quantity of each section of a dimension", py::arg("self"), py::arg("dim"),
	    py::return_value_policy::reference_internal);
	uniform_call(
	    m, pybtensor, "sizes", [](const btensor &a) { return a.sizes(); }, "return the full sizes of the tensor",
	    py::arg("self"));
	// dim
	uniform_call(
	    m, pybtensor, "dim", [](const btensor &a) { return a.dim(); }, "the rank of the tensor", py::arg("self"));
	// btensor add(const btensor &other, Scalar alpha = 1) const;
	uniform_call(m, pybtensor, "add",
	             wrap_scalar([](const btensor &self, const btensor &other, c10::Scalar alpha)
	                         { return self.add(other, alpha); }),
	             "perform addition with a scalar (default=1) prefactor on the other tensor", py::arg("self"),
	             py::arg("other"), py::arg("alpha") = 1);
	// btensor &add_(const btensor &other, Scalar alpha = 1);
	uniform_call(
	    m, pybtensor, "add_",
	    wrap_scalar([](btensor &self, const btensor &other, c10::Scalar alpha) { return self.add_(other, alpha); }),
	    "perform in place addition of self with the other tensor mulitplied by a scalar (default=1).", py::arg("self"),
	    py::arg("other"), py::arg("scalar") = 1, py::return_value_policy::reference_internal);
	// btensor bmm(const btensor &mat) const;
	uniform_call(
	    m, pybtensor, "bmm", [](const btensor &self, const btensor &mat) { return self.bmm(mat); },
	    "batch matrix multiply", py::arg("self"), py::arg("mat"));
	// btensor sum() const;
	uniform_call(m, pybtensor, "sum", wrap_scalar([](const btensor self) { return self.sum(); }),
	             "perform the sum of all the element of the tensor", py::arg("self"));
	// btensor sqrt() const;
	uniform_call(
	    m, pybtensor, "sqrt", [](const btensor &self) { return self.sqrt(); },
	    "compute the element by element square root", py::arg("self"));
	// btensor &sqrt_();
	uniform_call(
	    m, pybtensor, "sqrt_", [](btensor &self) { return self.sqrt_(); },
	    "in-place compute the element by element square root", py::arg("self"),
	    py::return_value_policy::reference_internal);
	// btensor abs() const;
	uniform_call(
	    m, pybtensor, "abs", [](const btensor &self) { return self.abs(); },
	    "compute the element by element absolute value", py::arg("self"));
	// btensor &abs_();
	uniform_call(
	    m, pybtensor, "abs_", [](btensor &self) { return self.abs_(); },
	    "in-place compute the element by element square root", py::arg("self"),
	    py::return_value_policy::reference_internal);
	// btensor div(btensor::Scalar) const;
	uniform_call(m, pybtensor, "div", wrap_scalar([](const btensor &self, c10::Scalar x) { return self.div(x); }),
	             "divide each element by the given value", py::arg("self"), py::arg("value"));
	// btensor &div_(btensor::Scalar);
	uniform_call(m, pybtensor, "div_", wrap_scalar([](btensor &self, c10::Scalar x) { return self.div_(x); }),
	             "in-place divide each element by the given value", py::arg("self"), py::arg("value"),
	             py::return_value_policy::reference_internal);
	// btensor div(const btensor &other) const;
	uniform_call(
	    m, pybtensor, "div", [](const btensor &self, const btensor &other) { return self.div(other); },
	    "element by element division of one tensor by another", py::arg("self"), py::arg("other"));
	// btensor &div_(const btensor &other);
	uniform_call(
	    m, pybtensor, "div_", [](btensor &self, const btensor &other) { return self.div_(other); },
	    "in place element by element division of one tensor by another", py::arg("self"), py::arg("other"),
	    pol_internal_ref);
	// btensor &mul_(const btensor &other);
	uniform_call(
	    m, pybtensor, "mul_", [](btensor &self, const btensor &other) { return self.mul_(other); },
	    "in place element by element product of one tensor by another", py::arg("self"), py::arg("other"),
	    pol_internal_ref);
	// btensor mul(const btensor &other) const;
	uniform_call(
	    m, pybtensor, "mul", [](const btensor &self, const btensor &other) { return self.mul(other); },
	    "in place element by element product of one tensor by another", py::arg("self"), py::arg("other"));
	// btensor mul(Scalar other) const;
	uniform_call(m, pybtensor, "mul",
	             wrap_scalar([](const btensor &self, c10::Scalar other) { return self.mul(other); }),
	             "product with a scalar", py::arg("self"), py::arg("other"));
	// btensor &mul_(Scalar other);
	uniform_call(m, pybtensor, "mul_", wrap_scalar([](btensor &self, c10::Scalar other) { return self.mul_(other); }),
	             "in place product with a scalar", py::arg("self"), py::arg("other"), pol_internal_ref);
	// btensor multiply(const btensor &other) const;
	uniform_call(
	    m, pybtensor, "multiply", [](const btensor &self, const btensor &other) { return self.mul(other); },
	    "in place element by element product of one tensor by another", py::arg("self"), py::arg("other"));
	// btensor &multiply_(const btensor &other);
	uniform_call(
	    m, pybtensor, "multiply_", [](btensor &self, const btensor &other) { return self.mul_(other); },
	    "in place element by element product of one tensor by another", py::arg("self"), py::arg("other"),
	    pol_internal_ref);
	// btensor multiply(Scalar other) const { return mul(other); }
	uniform_call(m, pybtensor, "multiply",
	             wrap_scalar([](const btensor &self, c10::Scalar other) { return self.mul(other); }),
	             "product with a scalar", py::arg("self"), py::arg("other"));
	// btensor &multiply_(Scalar other) { return mul_(other); }
	uniform_call(m, pybtensor, "multiply_",
	             wrap_scalar([](btensor &self, c10::Scalar other) { return self.mul_(other); }),
	             "in place product with a scalar", py::arg("self"), py::arg("other"), pol_internal_ref);
	// btensor permute(torch::IntArrayRef) const;
	uniform_call(
	    m, pybtensor, "permute", [](const btensor &self, torch::IntArrayRef perm) { return self.permute(perm); },
	    "permute the dimensions of the tensor", py::arg("self"), py::arg("perm"));
	// btensor &permute_(torch::IntArrayRef);
	uniform_call(
	    m, pybtensor, "permute_", [](btensor &self, torch::IntArrayRef perm) { return self.permute_(perm); },
	    "in-place (not any more efficient than permute() ) permute the dimensions of the tensor", py::arg("self"),
	    py::arg("perm"), pol_internal_ref);
	// btensor reshape(torch::IntArrayRef index_group) const;
	uniform_call(
	    m, pybtensor, "reshape",
	    [](const btensor &self, torch::IntArrayRef index_group) { return self.reshape(index_group); },
	    "reshape the tensor by grouping together dimensions. The group are specified by giving the (excluded) upper "
	    "boundary of each group. the dimensions of the tensor are group together in order. Perform a permutation "
	    "before hand if this order is not satisfactory. ",
	    py::arg("self"), py::arg("index_groups"));
	// btensor reshape_as(const btensor &other) const;
	uniform_call(
	    m, pybtensor, "reshape_as", [](const btensor &self, const btensor &shape) { return self.reshape_as(shape); },
	    "reshape this tensor to the shape of another tensor. dimensions are joined and split as necessary, in their "
	    "conventionnal order. perform permutation before hand if necessary",
	    py::arg("self"), py::arg("shape"));
	// btensor transpose(int64_t dim0, int64_t dim1) const;
	uniform_call(
	    m, pybtensor, "transpose",
	    [](const btensor &self, int64_t dim0, int64_t dim1) { return self.transpose(dim0, dim1); },
	    "exchange the two specified dimensions", py::arg("self"), py::arg("dim0"), py::arg("dim1"));
	// btensor &transpose_(int64_t dim0, int64_t dim1);
	uniform_call(
	    m, pybtensor, "transpose_",
	    [](btensor &self, int64_t dim0, int64_t dim1) { return self.transpose_(dim0, dim1); },
	    "in place exchange of the two specified dimensions", py::arg("self"), py::arg("dim0"), py::arg("dim1"),
	    pol_internal_ref);
	// btensor sub(const btensor &other, Scalar alpha = 1) const { return add(other, -alpha); }
	uniform_call(m, pybtensor, "sub",
	             wrap_scalar([](const btensor &self, const btensor &other, c10::Scalar alpha)
	                         { return self.sub(other, alpha); }),
	             "perform the substraction with another tensor muliplied by a scalar prefactor", py::arg("self"),
	             py::arg("other"), py::arg("alpha") = 1);
	// btensor &sub_(const btensor &other, Scalar alpha = 1) { return add_(other, -alpha); }
	uniform_call(
	    m, pybtensor, "sub_",
	    wrap_scalar([](btensor &self, const btensor &other, c10::Scalar alpha) { return self.sub_(other, alpha); }),
	    "perform the in-place substraction with another tensor muliplied by a scalar prefactor", py::arg("self"),
	    py::arg("other"), py::arg("alpha") = 1, pol_internal_ref);
	// btensor sub(Scalar other, Scalar alpha = 1) const;
	uniform_call(
	    m, pybtensor, "sub",
	    wrap_scalar([](const btensor &self, c10::Scalar other, c10::Scalar alpha) { return self.sub(other, alpha); }),
	    "perform the substraction with another tensor muliplied by a scalar prefactor", py::arg("self"),
	    py::arg("other"), py::arg("alpha") = 1);
	// btensor &sub_(Scalar other, Scalar alpha = 1);
	uniform_call(
	    m, pybtensor, "sub_",
	    wrap_scalar([](btensor &self, c10::Scalar other, c10::Scalar alpha) { return self.sub_(other, alpha); }),
	    "perform the in-place substraction with another tensor muliplied by a scalar prefactor", py::arg("self"),
	    py::arg("other"), py::arg("alpha") = 1, pol_internal_ref);
	// btensor subtract(const btensor &other, Scalar alpha = 1) const { return sub(other, alpha); }
	uniform_call(m, pybtensor, "subtract",
	             wrap_scalar([](const btensor &self, const btensor &other, c10::Scalar alpha)
	                         { return self.sub(other, alpha); }),
	             "perform the substraction with another tensor muliplied by a scalar prefactor", py::arg("self"),
	             py::arg("other"), py::arg("alpha") = 1);
	// btensor &subtract_(const btensor &other, Scalar alpha = 1) { return sub_(other, alpha); }
	uniform_call(
	    m, pybtensor, "subtract_",
	    wrap_scalar([](btensor &self, const btensor &other, c10::Scalar alpha) { return self.sub_(other, alpha); }),
	    "perform the in-place substraction with another tensor muliplied by a scalar prefactor", py::arg("self"),
	    py::arg("other"), py::arg("alpha") = 1, pol_internal_ref);
	// btensor subtract(Scalar other, Scalar alpha = 1) const { return sub(other, alpha); }
	uniform_call(
	    m, pybtensor, "subtract",
	    wrap_scalar([](const btensor &self, c10::Scalar other, c10::Scalar alpha) { return self.sub(other, alpha); }),
	    "perform the substraction with another tensor muliplied by a scalar prefactor", py::arg("self"),
	    py::arg("other"), py::arg("alpha") = 1);
	// btensor &subtract_(Scalar other, Scalar alpha = 1) { return sub_(other, alpha); }
	uniform_call(
	    m, pybtensor, "subtract_",
	    wrap_scalar([](btensor &self, c10::Scalar other, c10::Scalar alpha) { return self.sub_(other, alpha); }),
	    "perform the in-place substraction with another tensor muliplied by a scalar prefactor", py::arg("self"),
	    py::arg("other"), py::arg("alpha") = 1, pol_internal_ref);
	// btensor tensordot(const btensor &other, torch::IntArrayRef dim_self, torch::IntArrayRef dims_other) const;
	uniform_call(
	    m, pybtensor, "tensordot",
	    [](const btensor &self, const btensor &other, std::tuple<torch::IntArrayRef, torch::IntArrayRef> dims)
	    { return self.tensordot(other, std::get<0>(dims), std::get<1>(dims)); },
	    "perform the tensor contraction of the specified dimensions of the two tensors", py::arg("self"),
	    py::arg("other"), py::arg("dims"));
	uniform_call(
	    m, pybtensor, "tensordot",
	    [](const btensor &self, const btensor &other, size_t dims)
	    {
		    std::vector<int64_t> dim_self(dims);
		    std::vector<int64_t> dim_other(dims);
		    std::iota(dim_other.begin(), dim_other.end(), 0);
		    std::iota(dim_self.begin(), dim_self.end(), self.dim() - dims);
		    return self.tensordot(other, dim_self, dim_other);
	    },
	    "perform the tensor contraction of the last dims dimensions of the first tensor with the first dims dimension "
	    "of the second tensor",
	    py::arg("self"), py::arg("other"), py::arg("dims"));
	// btensor squeeze() const;
	uniform_call(
	    m, pybtensor, "squeeze", [](const btensor &self) { return self.squeeze(); },
	    "reshape the tensor such that all size one dimensions are removed from the tensor", py::arg("self"));
	// btensor squeeze(int64_t dim) const;
	uniform_call(
	    m, pybtensor, "squeeze", [](const btensor &self, int64_t dim) { return self.squeeze(dim); },
	    "squeeze the specified size one dimensions", py::arg("self"), py::arg("dim"));
	// btensor& squeeze_(int64_t dim);
	uniform_call(
	    m, pybtensor, "squeeze_", [](btensor &self, int64_t dim) { return self.squeeze_(dim); },
	    "in-place squeeze the specified size one dimensions", py::arg("self"), py::arg("dim"), pol_internal_ref);
	// btensor& squeeze_();
	uniform_call(
	    m, pybtensor, "squeeze_", [](btensor &self) { return self.squeeze_(); },
	    "in-place reshape the tensor such that all size one dimensions are removed from the tensor", py::arg("self"),
	    pol_internal_ref);
	// btensor isnan() const;
	uniform_call(
	    m, pybtensor, "isnan", [](const btensor &self) { return self.isnan(); }, "test elements for nan",
	    py::arg("self"));
	// torch::Tensor any() const;
	uniform_call(
	    m, pybtensor, "any", [](const btensor &self) { return self.any(); },
	    "verify wether any element of the tensor is true.", py::arg("self"));
	// bool anynan() const;
	uniform_call(
	    m, pybtensor, "anynan", [](const btensor &self) { return self.anynan(); }, "test whether any elements is nan",
	    py::arg("self"));
	// btensor conj() const;
	uniform_call(
	    m, pybtensor, "conj", [](const btensor &self) { return self.conj(); },
	    "apply complex conjugation to every element, and inverse all conserved quantity.", py::arg("self"));
	// btensor conj_only() const;
	uniform_call(
	    m, pybtensor, "conj_only", [](const btensor &self) { return self.conj_only(); },
	    "apply complex conjugation to every element.", py::arg("self"));
	// btensor inverse_cvals() const;
	uniform_call(
	    m, pybtensor, "inverse_cvals", [](const btensor &self) { return self.inverse_cvals(); },
	    "inverse all conserved quantity.", py::arg("self"));
	// btensor &inverse_cvals_();
	uniform_call(
	    m, pybtensor, "inverse_cvals_", [](btensor &self) { return self.inverse_cvals(); },
	    "in place inverse all conserved quantity.", py::arg("self"), pol_internal_ref);
	// btensor cval_shift(any_quantity_cref shift, int64_t dim) const;
	uniform_call(
	    m, pybtensor, "cval_shift",
	    [](const btensor &self, any_quantity qt, int64_t dim) { return self.cval_shift(qt, dim); },
	    "Shifts the conserved quantities of one dimension of the tensor, applies the opposite shift to the "
	    "conservation rule.",
	    py::arg("self"), py::arg("shift"), py::arg("dim"));
	// btensor &cval_shift_(any_quantity_cref shift, int64_t dim);
	uniform_call(
	    m, pybtensor, "cval_shift_",
	    [](btensor &self, any_quantity qt, int64_t dim) { return self.cval_shift_(qt, dim); },
	    "Shifts the conserved quantities of one dimension of the tensor, applies the opposite shift to the "
	    "conservation rule.",
	    py::arg("self"), py::arg("shift"), py::arg("dim"), pol_internal_ref);
	// btensor &non_conserving_cval_shift_(any_quantity_cref shift, int64_t dim);
	uniform_call(
	    m, pybtensor, "non_conserving_cval_shift_",
	    [](btensor &self, any_quantity qt, int64_t dim) { return self.non_conserving_cval_shift_(qt, dim); },
	    "Shifts the conserved quantities of one dimension of the tensor. only for empty tensors", py::arg("self"),
	    py::arg("shift"), py::arg("dim"), pol_internal_ref);
	// btensor &shift_selection_rule_(any_quantity_cref shift);
	uniform_call(
	    m, pybtensor, "shift_selection_rule_",
	    [](btensor &self, any_quantity qt) { return self.shift_selection_rule_(qt); },
	    "Apply shift to the selection rule, only for empty tensors", py::arg("self"), py::arg("shift"),
	    pol_internal_ref);
	// btensor &set_selection_rule_(any_quantity_cref value);
	uniform_call(
	    m, pybtensor, "set_selection_rule_",
	    [](btensor &self, any_quantity qt) { return self.set_selection_rule_(qt); },
	    " Set a new selection rule, only for empty tensors", py::arg("self"), py::arg("shift"), pol_internal_ref);
	uniform_call(
	    m, pybtensor, "set_selection_rule",
	    [](btensor &self, any_quantity qt) { return btensor(self).set_selection_rule_(qt); },
	    "Create a new tensor with a new selection rule, only for empty tensors", py::arg("self"), py::arg("shift"),
	    pol_internal_ref);
	// btensor &neutral_selection_rule_() {return set_selection_rule_(selection_rule->neutral());}
	uniform_call(
	    m, pybtensor, "neutral_selection_rule_", [](btensor &self) { return self.neutral_selection_rule_(); },
	    "set the selection rule to the neutral element", py::arg("self"), pol_internal_ref);
	// btensor neutral_selection_rule()
	uniform_call(
	    m, pybtensor, "neutral_selection_rule", [](btensor &self) { return self.neutral_selection_rule(); },
	    "set the selection rule to the nuetral element", py::arg("self"));
	// btensor to(const torch::TensorOptions &options = {}, bool non_blocking = false, bool copy = false,
	//            c10::optional<c10::MemoryFormat> memory_format = c10::nullopt) const
	pybtensor.def(
	    "to",
	    [](const btensor &self, torch::ScalarType dtype, c10::optional<torch::Device> dev, bool non_blocking, bool copy,
	       c10::MemoryFormat fmt) { return self.to(torch::TensorOptions(dtype).device(dev), non_blocking, copy, fmt); },
	    "perform tensor dtype conversion", py::arg("dtype")= c10::optional<torch::Dtype>(), py::kw_only(),
	    py::arg("device") = c10::optional<torch::Device>(), py::arg("non_blocking") = false, py::arg("copy") = false,
	    py::arg("memory_format") = c10::MemoryFormat::Preserve);
	// btensor to(const torch::Tensor &other, bool non_blocking = false, bool copy = false,
	//            c10::optional<c10::MemoryFormat> memory_format = c10::nullopt) const
	pybtensor.def(
	    "to",
	    [](const btensor &self, const torch::Tensor &other, bool non_blocking, bool copy, c10::MemoryFormat fmt)
	    { return self.to(other, non_blocking, copy, fmt); },
	    "perform tensor dtype and device conversion using supplied tensor options", py::arg("other"), py::kw_only(),
	    py::arg("non_blocking") = false, py::arg("copy") = false,
	    py::arg("memory_format") = c10::MemoryFormat::Preserve);
	// btensor to(const btensor &other, bool non_blocking = false, bool copy = false,
	//            c10::optional<c10::MemoryFormat> memory_format = c10::nullopt) const
	pybtensor.def(
	    "to",
	    [](const btensor &self, const btensor &other, bool non_blocking, bool copy, c10::MemoryFormat fmt)
	    { return self.to(other, non_blocking, copy, fmt); },
	    "perform tensor dtype and device conversion using supplied tensor options", py::arg("other"), py::kw_only(),
	    py::arg("non_blocking") = false, py::arg("copy") = false,
	    py::arg("memory_format") = c10::MemoryFormat::Preserve);
	// btensor shape_from(const std::vector<int64_t> &dims) const;
	uniform_call(
	    m, pybtensor, "shape_from",
	    [](const btensor &self, const std::vector<int64_t> &dims) { return self.shape_from(dims); },
	    "return the shape of the tensor specifed. supply -1 to keep the whole dimension, otherwise specify the index "
	    "value.",
	    py::arg("self"), py::arg("dims"));
	// btensor basic_create_view(const std::vector<int64_t> &dims, bool preserve_rank = false);
	uniform_call(
	    m, pybtensor, "basic_create_view",
	    [](btensor &self, const std::vector<int64_t> &dims, bool preserve_rank)
	    { return self.basic_create_view(dims, preserve_rank); },
	    "return the shape of the tensor specifed. supply -1 to keep the whole dimension, otherwise specify the index "
	    "value.",
	    py::arg("self"), py::arg("dims"), py::arg("preserve_rank") = false);
	uniform_call(
	    m, pybtensor, "basic_index_put_",
	    [](btensor &self, std::vector<int64_t> index, const btensor &value)
	    { return self.basic_index_put_(index, value); },
	    "insert the values at the specified view", py::arg("self"), py::arg("index"), py::arg("values"), pol_internal_ref);
	uniform_call(
	    m, pybtensor, "basic_index_put_",
	    [](btensor &self, std::vector<int64_t> index, const torch::Tensor &value)
	    { return self.basic_index_put_(index, value); },
	    "insert the values at the specified view, drop any element that do not satisfy the selection rule.",
	    py::arg("self"), py::arg("index"), py::arg("values"), pol_internal_ref);
	// Free standing only
	//  inline btensor disambiguated_shape_from(const std::vector<btensor> &btens_list) -> shape_from
	m.def(
	    "shape_from", [](const std::vector<btensor> &tensors) { return disambiguated_shape_from(tensors); },
	    "return an empty tensor with the shape of the tensor product of all the tensors.", py::arg("tensors"));
	m.def("find_selection_rule", wrap_scalar(&quantit::find_selection_rule),
	      "find the selection rule for a torch tensor with a given candidate shape.", py::arg("tensor"),
	      py::arg("shape"), py::arg("cutoff") = 0);
}