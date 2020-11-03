/*
 * File: vquantity.h
 * Project: quantt
 * File Created: Tuesday, 15th September 2020 2:56:09 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Tuesday, 15th September 2020 2:56:09 pm
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#ifndef F2547C1C_9177_4373_9C66_8D4C8621C7CC
#define F2547C1C_9177_4373_9C66_8D4C8621C7CC

#include "Conserved/quantity_utils.h"
#include "templateMeta.h"
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <memory>
#include <type_traits>
#include <vector>

namespace quantt
{

/**
 * @brief forward declaration of the iterator type used for type erased container
 * 
 */
struct cgroup_iterator;
struct const_cgroup_iterator;
class vquantity_vector;

/**
 * @brief interface type for the implementation of a any_quantity
 * any_quantity stands for "composite group" and is the tensor product of multiple simple groups.
 * 
 * The function that take a vquantity as an argument are not expected to work
 * if the actual type of the argument isn't the same derived type as this.
 */
class vquantity
{
	friend struct cgroup_iterator;
	friend struct const_cgroup_iterator;

public:
	/**
	* @brief in place implementation of the group operation.
	*        *this = (*this)*other, where "other" is the argument givent
	* @return vquantity& : a reference to the current object.
	*/
	virtual vquantity&
	op(const vquantity&) = 0;
	/**
	 * @brief in place implementation of the group operation, the result is 
	 * stored in the given argument.
	 * 
	 */
	virtual void op_to(vquantity&) const = 0;
	/**
	 * @brief in place computation of the inverse of *this.
	 * 
	 * @return vquantity& : reference to the current object.
	 */
	virtual vquantity& inverse_() = 0;

	virtual std::unique_ptr<vquantity> clone() const = 0;
	virtual std::unique_ptr<vquantity_vector> make_vector(size_t cnt) const = 0;

	/**
	 * @brief create the neutral element of whatever underlying type
	 * 
	 * @return std::unique_ptr<vquantity> : the neutral element
	 */
	virtual std::unique_ptr<vquantity> neutral() const = 0;

	virtual vquantity& operator=(const vquantity&) = 0;
	virtual bool operator==(const vquantity&) const = 0;
	virtual bool operator!=(const vquantity&) const = 0;
	virtual void swap(vquantity&) = 0;
	virtual auto format_to(fmt::format_context& ctx) const -> decltype(ctx.out()) = 0;
	virtual ~vquantity() {}
};

/**
 * @brief template implementation of the concrete composite group types.
 * This template of class is used by the type any_quantity, any_quantity_ref and any_quantity_cref
 * defined in composite_group.h
 * @tparam Groups 
 */
template <class... Groups>
class quantity final : public vquantity
{
public:
	// has default constructor and assigment operator as well.
	quantity(Groups... grp) : val(grp...) {}
	quantity() = default;
	quantity(const quantity&) = default;
	quantity(quantity&&) = default;
	~quantity() override{};
	quantity& operator=(quantity other) noexcept;

	quantity operator*(const quantity&);
	quantity operator*(quantity&&);
	quantity operator*=(const quantity&);
	quantity operator+(const quantity&);
	quantity operator+(quantity&&);
	quantity operator+=(const quantity&);
	void swap(quantity& other);

	std::unique_ptr<vquantity> clone() const override;
	std::unique_ptr<vquantity> neutral() const override;

	quantity& op(const quantity& other);
	vquantity& op(const vquantity& other) override;
	void op_to(quantity& other) const;
	void op_to(vquantity& other) const override;
	quantity& inverse_() override;
	vquantity& operator=(const vquantity& other) override;
	bool operator==(const quantity& other) const;
	bool operator==(const vquantity& other) const override;
	bool operator!=(const quantity& other) const;
	bool operator!=(const vquantity& other) const override;
	void swap(vquantity& other) override;

	std::unique_ptr<vquantity_vector> make_vector(size_t cnt) const override;

	friend struct fmt::formatter<quantt::quantity<Groups...>>;
	auto format_to(fmt::format_context& ctx) const -> decltype(ctx.out()) override
	{
		return fmt::formatter<quantt::quantity<Groups...>>().format(*this, ctx);
	}

private:
	/**
	 * @brief implementation of the polymophic pointer difference
	 * 
	 * this is the way to implement this, basically no matter what.
	 * 
	 */
	std::tuple<Groups...> val;
};
template <class... Qts>
quantity<Qts...> quantity<Qts...>::operator*(const quantity<Qts...>& other)
{
	return quantity<Qts...>(*this).op(other);
}
template <class... Qts>
quantity<Qts...> quantity<Qts...>::operator*(quantity<Qts...>&& other)
{
	op_to(other);
	return other;
}
template <class... Qts>
quantity<Qts...> quantity<Qts...>::operator*=(const quantity<Qts...>& other)
{
	op(other);
	return *this;
}
template <class... Qts>
quantity<Qts...> quantity<Qts...>::operator+(const quantity<Qts...>& other)
{
	return *this * other;
}
template <class... Qts>
quantity<Qts...> quantity<Qts...>::operator+(quantity<Qts...>&& other)
{
	return *this * std::move(other);
}
template <class... Qts>
quantity<Qts...> quantity<Qts...>::operator+=(const quantity<Qts...>& other)
{
	return *this *= other;
}
template <class... T>
quantity<T...>& quantity<T...>::operator=(quantity other) noexcept
{
	using std::swap;
	swap(val, other.val);
	return *this;
}

template <class... T>
quantity<T...>& quantity<T...>::op(const quantity<T...>& other)
{
	for_each2(val, other.val, [](auto&& vl, auto&& ovl) {
		vl.op(ovl);
	});
	return *this;
}
template <class... T>
vquantity& quantity<T...>::op(const vquantity& other)
{
	return op(dynamic_cast<const quantity&>(other));
}

template <class... T>
std::unique_ptr<vquantity> quantity<T...>::neutral() const
{
	return std::make_unique<quantity<T...>>();
}

template <class... T>
std::unique_ptr<vquantity> quantity<T...>::clone() const
{
	return std::make_unique<quantity<T...>>(*this);
}
template <class... T>
void quantity<T...>::op_to(quantity<T...>& other) const
{
	for_each2(val, other.val, [](auto&& vl, auto&& ovl) {
		ovl = conserved::op(vl, ovl);
	});
}
template <class... T>
void quantity<T...>::op_to(vquantity& other) const
{
	return op_to(dynamic_cast<quantity<T...>&>(other));
}
template <class... T>
quantity<T...>& quantity<T...>::inverse_()
{
	for_each(val, [](auto&& vl) {
		vl.inverse_();
	});
	return *this;
}
template <class... T>
vquantity& quantity<T...>::operator=(const vquantity& other)
{
	return (*this) = dynamic_cast<const quantity<T...>&>(other);
}
template <class... T>
bool quantity<T...>::operator==(const quantity<T...>& other) const
{
	return val == other.val;
}
template <class... T>
bool quantity<T...>::operator==(const vquantity& other) const
{
	return operator==(dynamic_cast<const quantity<T...>&>(other));
}
template <class... T>
bool quantity<T...>::operator!=(const quantity<T...>& other) const
{
	return val != other.val;
}
template <class... T>
bool quantity<T...>::operator!=(const vquantity& other) const
{
	return operator!=(dynamic_cast<const quantity<T...>&>(other));
}
template <class... T>
void quantity<T...>::swap(quantity<T...>& other)
{
	using std::swap;
	swap(val, other.val);
}
template <class... T>
void quantity<T...>::swap(vquantity& other)
{
	swap(dynamic_cast<quantity<T...>&>(other));
}

template <class>
struct is_conc_cgroup_impl : public std::false_type
{
};
template <class... S>
struct is_conc_cgroup_impl<quantity<S...>> : public conserved::all_conserved_quantt<S...>
{
};

} // namespace quantt

template <class... Groups>
struct fmt::formatter<quantt::quantity<Groups...>>
{
	constexpr auto parse(format_parse_context& ctx)
	{
		auto it = ctx.begin(), end = ctx.end();
		if (it != end and *it != '}')
			throw format_error("invalid format, no formatting option for quantt::quantity");
		// if (*it != '}')
		// {
		// 	++it;
		// 	auto first_pos = it;
		// 	while (it != end && *it != '}')
		// 	{
		// 		++it;
		// 	}
		// 	auto last_pos = it - 1;
		// 	uint width = 80;
		// 	if (last_pos > first_pos)
		// 	{
		// 		auto code = std::from_chars(first_pos, last_pos, width);
		// 		if (code.ptr != last_pos || std::errc::invalid_argument == code.ec)
		// 			throw format_error("invalid format, no formatting option for quatt::quantity");
		// 		else
		// 		{
		// 			linelenght = width;
		// 		}
		// 	}
		// }
		if (*it != '}')
			throw format_error("invalid format,closing brace missing");

		// Return an iterator past the end of the parsed range:
		return it;
	}

	template <typename FormatContext>
	auto format(const quantt::quantity<Groups...>& qt, FormatContext& ctx)
	{
		return format_to(ctx.out(), "[{}]", fmt::join(qt.val, ","));
	}
};

template <>
struct fmt::formatter<quantt::vquantity>
{
	constexpr auto parse(format_parse_context& ctx)
	{
		auto it = ctx.begin(), end = ctx.end();
		if (it != end and *it != '}')
			throw format_error("invalid format, no formatting option for quantt::quantity");
		if (*it != '}')
			throw format_error("invalid format,closing brace missing");

		// Return an iterator past the end of the parsed range:
		return it;
	}

	template <class FormatContext>
	auto format(const quantt::vquantity& qt, FormatContext& ctx)
	{
		return qt.format_to(ctx); //right now qt.format_to is only defined for fmt::format_context. Should work for any output stream.
	}
};

#endif /* F2547C1C_9177_4373_9C66_8D4C8621C7CC */
