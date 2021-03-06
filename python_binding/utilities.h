
#ifndef UTILITIES_H
#define UTILITIES_H

#include <type_traits>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/csrc/MemoryFormat.h>

#include "blockTensor/btensor.h"

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
template <>
struct type_caster<at::MemoryFormat>
{
  public:
	PYBIND11_TYPE_CASTER(at::MemoryFormat, _("memory_format"));

	bool load(handle src, bool)
	{
		PyObject *source = src.ptr();
		// if(!source)
		//     return false;
		if (THPMemoryFormat_Check(source))
			value = reinterpret_cast<THPMemoryFormat *>(source)->memory_format;
		else
			return false;
		return !PyErr_Occurred();
	}
	static handle cast(at::MemoryFormat src, return_value_policy, handle)
	{
		std::stringstream name;
		name << src;
		return THPMemoryFormat_New(src, name.str());
	}
};
} // namespace detail
} // namespace pybind11

namespace utils
{

// Nasty shenanigans 
// This evil trickery allow us to modify and read private values. We must NEVER modify it.
namespace Evil
{
template <typename Tag, typename Tag::type M>
struct Rob
{
	friend typename Tag::type get(Tag) { return M; }
};
template <class stolen_type, class Victim, size_t tag = 0>
struct Thieving_tag
{
	typedef stolen_type Victim::*type;
#if defined(__GNUC__) and not defined(__clang__) // clang doesn't understand this, and defines __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-template-friend" // yeah i know the next function ain't a template.
#endif
	friend type get(Thieving_tag);
#if defined(__GNUC__) and not defined(__clang__)
#pragma GCC diagnostic pop
#endif
};
// usage exemple
// using refcount_theft = Thieving_tag<std::atomic<size_t>, c10::intrusive_ptr_target, 0>;
// using weakcount_theft = Thieving_tag<std::atomic<size_t>, c10::intrusive_ptr_target, 1>;
// template struct Rob<refcount_theft, &c10::intrusive_ptr_target::refcount_>;
// template struct Rob<weakcount_theft, &c10::intrusive_ptr_target::weakcount_>;
// size_t get_refcount(const torch::Tensor &tens)
// {
// 	using namespace Evil;
// 	auto refcounted_ptr_1 = tens.unsafeGetTensorImpl();
// 	auto refcount_1 = ((*refcounted_ptr_1).*get(refcount_theft())).load(); // this is an atomic load
// 	// auto weakcount_1 = ((*refcounted_ptr_1).*get(weakcount_theft())).load();
// 	// auto refcounted_ptr_2 = tens.unsafeGetTensorImpl();
// 	// auto weakcount_2 = ((*refcounted_ptr_2).*get(weakcount_theft())).load();
// 	// fmt::print("weakcount {} {}\n", weakcount_1, weakcount_2);
// 	return refcount_1;
// }
} // namespace Evil
namespace function_sig_fiddling
{
// Taken from pybind11::detail, Because i have no trust that this isn't an implementation *detail* of pybind.
/// Strip the class from a method type
template <typename T>
struct remove_class
{
};
template <typename C, typename R, typename... A>
struct remove_class<R (C::*)(A...)>
{
	using type = R(A...);
};
template <typename C, typename R, typename... A>
struct remove_class<R (C::*)(A...) const>
{
	using type = R(A...);
};

template <typename F>
struct strip_function_object
{
	// If you are encountering an
	// 'error: name followed by "::" must be a class or namespace name'
	// with the Intel compiler and a noexcept function here,
	// try to use noexcept(true) instead of plain noexcept.
	using type = typename remove_class<decltype(&F::operator())>::type;
};

// Extracts the function signature from a function, function pointer or lambda.
template <typename Function, typename F = std::remove_reference_t<Function>>
using function_signature_t =
    std::conditional_t<std::is_function<F>::value, F,
                       typename std::conditional_t<std::is_pointer<F>::value || std::is_member_pointer<F>::value,
                                                   std::remove_pointer<F>, strip_function_object<F>>::type>;
} // namespace function_sig_fiddling

template <class T>
struct Cast_Scalars
{
	template<class X>
	using remove_cvref_t = std::remove_reference_t<std::remove_cv_t<X>>;
	using out_type = T;
	using in_type = T;
	using sT = remove_cvref_t<T>;
	static sT &cast(sT &t) { return t; }
	static sT &&cast(sT &&t) { return std::move(t); }
};

template <>
struct Cast_Scalars<c10::Scalar>
{
	using out_type = py::object;
	using in_type = c10::Scalar;
	static py::object cast(const c10::Scalar &S)
	{
		using namespace c10;
		switch (S.type())
		{
		case ScalarType::ComplexDouble:
		{
			auto src = S.toComplexDouble();
			return py::reinterpret_steal<py::object>(PyComplex_FromDoubles((double)src.real(), (double)src.imag()));
		}
		case ScalarType::Double:
			return py::reinterpret_steal<py::object>(PyFloat_FromDouble(S.toDouble()));
		case ScalarType::Long:
			return py::reinterpret_steal<py::object>(PyLong_FromLong(S.toLong()));
		case ScalarType::Bool:
			return py::reinterpret_steal<py::object>(PyBool_FromLong(S.toBool()));
		default:
			throw std::runtime_error("invalid type");
		}
	}
	static c10::Scalar cast(const py::object &obj)
	{
		if (PyBool_Check(obj.ptr()))
			return THPUtils_unpackBool(obj.ptr());
		else if (THPUtils_checkLong(obj.ptr()))
			return THPUtils_unpackLong(obj.ptr());
		else if (THPUtils_checkDouble(obj.ptr()))
			return THPUtils_unpackDouble(obj.ptr());
		else if (PyComplex_Check(obj.ptr()))
			return THPUtils_unpackComplexDouble(obj.ptr());
		else
			throw std::runtime_error("invalid type");
	}
};

template <class F, class Ret, class... Args>
auto wrap_impl(F &&f, Ret (*)(Args...))
{
	// we use template type deduction to decompose the arguments and return from the arguement.
	// The second argument MUST be the function signature associated with the function or function-like object f
	// we apply the type transformation defined by cast scalar to the return and arguements.
	// It the identity for any type that is not c10::Scalar.
	if constexpr (std::is_same_v<void,Ret>)
		return [f](typename Cast_Scalars<Args>::out_type... args)
	{ return f(Cast_Scalars<Args>::cast(args)...); };	
	else
	return [f](typename Cast_Scalars<Args>::out_type... args)
	{ return Cast_Scalars<Ret>::cast(f(Cast_Scalars<Args>::cast(args)...)); };
}
/**
 * @brief wrap a function such that it is a new function like object that does not expose c10:scalar in it's function
 * signature.
 *
 * This is done so that c10::Scalar need not be (and aren't) exposed on the python side. regular python number can be
 * used directly.
 *
 * @tparam F
 * @param f
 * @return auto
 */
template <class F>
auto wrap_scalar(F &&f)
{
	// just because direct usage of wrap_impl is a bit too complicated.
	return wrap_impl(f, (function_sig_fiddling::function_signature_t<F> *)nullptr);
}

template <class... Args>
struct TOPT_binder
{
	template <class ARGS>
	static c10::TensorOptions generate_option(const quantit::btensor &opt_ten, ARGS &&...)
	{
		return opt_ten.options();
	}
	static c10::TensorOptions generate_option(const quantit::btensor &opt_ten) { return opt_ten.options(); }
	
	template<class Ret =quantit::btensor>
	static auto bind(Ret (*f)(Args..., c10::TensorOptions))
	{
		using FirstEntityType = std::tuple_element_t<0, std::tuple<Args...>>;

		if constexpr (std::is_same_v<quantit::remove_cvref_t<FirstEntityType>, quantit::btensor>)
		{
			return [f](Args... args, torch::optional<torch::ScalarType> dt, torch::optional<torch::Device> dev,
			           torch::optional<bool> req_grad, torch::optional<bool> pin_memory)
			{
				auto opt = generate_option(std::forward<Args>(args)...)
				               .dtype(dt)
				               .device(dev)
				               .requires_grad(req_grad)
				               .pinned_memory(pin_memory);
				return f(args..., opt);
			};
		}
		else
		{
			return [f](Args... args, torch::optional<torch::ScalarType> dt, torch::optional<torch::Device> dev,
			           torch::optional<bool> req_grad, torch::optional<bool> pin_memory)
			{
				auto opt =
				    torch::TensorOptions().dtype(dt).device(dev).requires_grad(req_grad).pinned_memory(pin_memory);
				return f(args..., opt);
			};
		}
	}
	template<class F>
	static auto bind_fl(F&& f)
	{
		return TOPT_binder::bind_fl_impl(f,(function_sig_fiddling::function_signature_t<F> *)nullptr);
	}

	private:
	template<class F, class Ret> 
	static auto bind_fl_impl(F&& f,Ret(*)(Args...,c10::TensorOptions))
	{
		using FirstEntityType = std::tuple_element_t<0, std::tuple<Args...>>;

		if constexpr (std::is_same_v<quantit::remove_cvref_t<FirstEntityType>, quantit::btensor>)
		{
			return [f](Args... args, torch::optional<torch::ScalarType> dt, torch::optional<torch::Device> dev,
			           torch::optional<bool> req_grad, torch::optional<bool> pin_memory)
			{
				auto opt = generate_option(std::forward<Args>(args)...)
				               .dtype(dt)
				               .device(dev)
				               .requires_grad(req_grad)
				               .pinned_memory(pin_memory);
				return f(args..., opt);
			};
		}
		else
		{
			return [f](Args... args, torch::optional<torch::ScalarType> dt, torch::optional<torch::Device> dev,
			           torch::optional<bool> req_grad, torch::optional<bool> pin_memory)
			{
				auto opt =
				    torch::TensorOptions().dtype(dt).device(dev).requires_grad(req_grad).pinned_memory(pin_memory);
				return f(args..., opt);
			};
		}

	}
};

template <class T>
using opt = torch::optional<T>;
using stype = torch::ScalarType;
using tdev = torch::Device;

} // namespace

#endif // UTILITIES_H
