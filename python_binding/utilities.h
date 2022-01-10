
#ifndef UTILITIES_H
#define UTILITIES_H

#include <type_traits>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/csrc/MemoryFormat.h>

namespace utils
{

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
	using out_type = T;
	using in_type = T;
	using sT = quantt::remove_cvref_t<T>;
	static sT &cast(sT &t) { return t; }
	static const sT &cast(const sT &t) { return t; }
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
struct binder
{
	template <class ARGS>
	static c10::TensorOptions generate_option(const quantt::btensor &opt_ten, ARGS &&...)
	{
		return opt_ten.options();
	}
	static c10::TensorOptions generate_option(const quantt::btensor &opt_ten) { return opt_ten.options(); }
	static auto bind(quantt::btensor (*f)(Args..., c10::TensorOptions))
	{
		using FirstEntityType = std::tuple_element_t<0, std::tuple<Args...>>;

		if constexpr (std::is_same_v<quantt::remove_cvref_t<FirstEntityType>, quantt::btensor>)
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
