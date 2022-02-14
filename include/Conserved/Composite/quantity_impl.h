/*
 * File: vquantity.h
 * Project: QuantiT
 * File Created: Tuesday, 15th September 2020 2:56:09 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Tuesday, 15th September 2020 2:56:09 pm
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * Licensed under GPL v3
 */

#ifndef F2547C1C_9177_4373_9C66_8D4C8621C7CC
#define F2547C1C_9177_4373_9C66_8D4C8621C7CC

#include "Conserved/quantity_utils.h"
#include "templateMeta.h"
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <ios>
#include <memory>
#include <type_traits>
#include <vector>

namespace quantit
{

/**
 * @brief forward declaration of the iterator type used for type erased container
 *
 */
struct cgroup_iterator;
struct const_cgroup_iterator;
class vquantity_vector;
class any_quantity;

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
	virtual vquantity &op(const vquantity &) = 0;
	/**
	 * @brief conditionnal, in place application of the group operation
	 *
	 * @param cond does nothing to this when false
	 * @return vquantity& reference to the current object
	 */
	virtual vquantity &op(const vquantity &, bool cond) = 0;
	/**
	 * @brief in place implementation of the group operation, the result is
	 * stored in the given argument.
	 *
	 */
	virtual void op_to(vquantity &) const = 0;
	/**
	 * @brief in place computation of the inverse of *this.
	 *
	 * @return vquantity& : reference to the current object.
	 */
	virtual vquantity &inverse_() = 0;
	any_quantity inverse() const;
	any_quantity inv() const;
	vquantity &inv_() { return inverse_(); }
	virtual std::unique_ptr<vquantity> clone() const = 0;
	virtual std::unique_ptr<vquantity_vector> make_vector(size_t cnt) const = 0;

	any_quantity neutral() const;
	virtual vquantity &operator=(const vquantity &) = 0;
	virtual vquantity &operator*=(const vquantity &) = 0;
	virtual vquantity &operator+=(const vquantity &) = 0;
	virtual bool is_equal(const vquantity &) const = 0;
	virtual bool is_different(const vquantity &) const = 0;
	virtual bool is_lesser(const vquantity &) const = 0;
	virtual bool is_greater(const vquantity &) const = 0;
	virtual bool same_type(const vquantity &other) const = 0;
	virtual void swap(vquantity &) = 0;
	virtual auto format_to(fmt::format_context &ctx) const -> decltype(ctx.out()) = 0;
	virtual double distance(const vquantity &) const = 0;
	virtual int64_t distance2(const vquantity &) const = 0;
	virtual ~vquantity() {}

  protected:
	/**
	 * @brief create the neutral element of whatever underlying type
	 *
	 * @return std::unique_ptr<vquantity> : the neutral element
	 */
	virtual std::unique_ptr<vquantity> make_neutral() const = 0;
};
inline bool operator==(const vquantity &left, const vquantity &right) { return left.is_equal(right); }
inline bool operator!=(const vquantity &left, const vquantity &right) { return left.is_different(right); }
inline bool operator<(const vquantity &left, const vquantity &right) { return left.is_lesser(right); }
inline bool operator>(const vquantity &left, const vquantity &right) { return left.is_greater(right); }

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
	static_assert(quantit::conserved::all_group_v<Groups...>,
	              "all template argument must statisfy the constraints of a group. See QuantiT/Conserved/quantity.h for "
	              "2 working exemples: C<N> and Z");
	quantity(Groups... grp) : val(grp...) {}
	quantity() = default;
	quantity(const quantity &) = default;
	quantity(quantity &&) = default;
	~quantity() override{};
	quantity &operator=(const quantity &other) noexcept;

	quantity operator*(const quantity &);
	quantity operator*(quantity &&);
	quantity operator*=(const quantity &);
	quantity operator+(const quantity &);
	quantity operator+(quantity &&);
	quantity operator+=(const quantity &);
	vquantity &operator*=(const vquantity &) override;
	vquantity &operator+=(const vquantity &) override;
	void swap(quantity &other);

	std::unique_ptr<vquantity> clone() const override;
	std::unique_ptr<vquantity> make_neutral() const override;

	quantity &op(const quantity &other);
	vquantity &op(const vquantity &other) override;
	quantity &op(const quantity &other, bool cond);
	vquantity &op(const vquantity &other, bool cond) override;
	void op_to(quantity &other) const;
	void op_to(vquantity &other) const override;
	quantity &inverse_() override;
	vquantity &operator=(const vquantity &other) override;
	bool operator==(const quantity &other) const;
	bool is_equal(const vquantity &other) const override;
	bool operator!=(const quantity &other) const;
	bool is_different(const vquantity &other) const override;
	bool same_type(const vquantity &other) const override;
	bool is_lesser(const vquantity &) const override;
	bool operator<(const quantity &other) const;
	bool is_greater(const vquantity &) const override;
	bool operator>(const quantity &other) const;
	void swap(vquantity &other) override;

	std::unique_ptr<vquantity_vector> make_vector(size_t cnt) const override;

	double distance(const vquantity &) const override;
	double distance(const quantity &) const;
	int64_t distance2(const vquantity &) const override;
	int64_t distance2(const quantity &) const;
	friend struct fmt::formatter<quantit::quantity<Groups...>>;
	auto format_to(fmt::format_context &ctx) const -> decltype(ctx.out()) override
	{
		return fmt::formatter<quantit::quantity<Groups...>>().format(*this, ctx);
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
bool quantity<Qts...>::same_type(const vquantity &other) const
{ // dynamic cast on pointers return a null pointer which convert to false when the types are incompatible.
	return dynamic_cast<const quantity<Qts...> *>(&other);
}
template <class... Qts>
quantity<Qts...> quantity<Qts...>::operator*(const quantity<Qts...> &other)
{
	return quantity<Qts...>(*this).op(other);
}
template <class... Qts>
quantity<Qts...> quantity<Qts...>::operator*(quantity<Qts...> &&other)
{
	op_to(other);
	return other;
}
template <class... Qts>
quantity<Qts...> quantity<Qts...>::operator*=(const quantity<Qts...> &other)
{
	op(other);
	return *this;
}
template <class... Qts>
quantity<Qts...> quantity<Qts...>::operator+(const quantity<Qts...> &other)
{
	return *this * other;
}
template <class... Qts>
quantity<Qts...> quantity<Qts...>::operator+(quantity<Qts...> &&other)
{
	return *this * std::move(other);
}
template <class... Qts>
quantity<Qts...> quantity<Qts...>::operator+=(const quantity<Qts...> &other)
{
	return *this *= other;
}
template <class... T>
quantity<T...> &quantity<T...>::operator=(const quantity &other) noexcept
{
	val = other.val;
	return *this;
}
template <class... T>
int64_t quantity<T...>::distance2(const quantity<T...> &other) const
{
	int64_t out = 0;
	for_each2(val, other.val, [&out](auto &&vl, auto &&ovl) { out += vl.distance2(ovl); });
	return out;
}
template <class... T>
int64_t quantity<T...>::distance2(const vquantity &other) const
{
	return distance2(dynamic_cast<const quantity<T...> &>(other));
}
template <class... T>
double quantity<T...>::distance(const quantity<T...> &other) const
{
	return std::sqrt(distance2(other));
}
template <class... T>
double quantity<T...>::distance(const vquantity &other) const
{
	return distance(dynamic_cast<const quantity<T...> &>(other));
}
template <class... T>
quantity<T...> &quantity<T...>::op(const quantity<T...> &other)
{
	for_each2(val, other.val, [](auto &&vl, auto &&ovl) { vl.op(ovl); });
	return *this;
}
template <class... T>
quantity<T...> &quantity<T...>::op(const quantity<T...> &other, bool cond)
{
	for_each2(val, other.val, [cond](auto &&vl, auto &&ovl) { vl.op(ovl, cond); });
	return *this;
}
template <class... T>
vquantity &quantity<T...>::op(const vquantity &other, bool cond)
{
	return op(dynamic_cast<const quantity &>(other), cond);
}

template <class... T>
vquantity &quantity<T...>::op(const vquantity &other)
{
	return op(dynamic_cast<const quantity &>(other));
}

template <class... T>
std::unique_ptr<vquantity> quantity<T...>::make_neutral() const
{
	return std::make_unique<quantity<T...>>();
}

template <class... T>
std::unique_ptr<vquantity> quantity<T...>::clone() const
{
	return std::make_unique<quantity<T...>>(*this);
}
template <class... T>
void quantity<T...>::op_to(quantity<T...> &other) const
{
	for_each2(val, other.val, [](auto &&vl, auto &&ovl) { ovl = conserved::op(vl, ovl); });
}
template <class... T>
void quantity<T...>::op_to(vquantity &other) const
{
	return op_to(dynamic_cast<quantity<T...> &>(other));
}
template <class... T>
quantity<T...> &quantity<T...>::inverse_()
{
	for_each(val, [](auto &&vl) { vl.inverse_(); });
	return *this;
}
template <class... T>
vquantity &quantity<T...>::operator=(const vquantity &other)
{
	return this->operator=(dynamic_cast<const quantity<T...> &>(other));
}
template <class... T>
bool quantity<T...>::operator==(const quantity<T...> &other) const
{
	return val == other.val;
}
template <class... T>
bool quantity<T...>::is_equal(const vquantity &other) const
{
	return operator==(dynamic_cast<const quantity<T...> &>(other));
}
template <class... T>
bool quantity<T...>::operator!=(const quantity<T...> &other) const
{
	return val != other.val;
}
template <class... T>
bool quantity<T...>::is_different(const vquantity &other) const
{
	return operator!=(dynamic_cast<const quantity<T...> &>(other));
}
template <class... T>
bool quantity<T...>::operator<(const quantity<T...> &other) const
{
	return val < other.val;
}
template <class... T>
bool quantity<T...>::is_lesser(const vquantity &other) const
{
	return operator<(dynamic_cast<const quantity<T...> &>(other));
}

template <class... T>
bool quantity<T...>::operator>(const quantity<T...> &other) const
{
	return val > other.val;
}
template <class... T>
bool quantity<T...>::is_greater(const vquantity &other) const
{
	return operator>(dynamic_cast<const quantity<T...> &>(other));
}
template <class... T>
void quantity<T...>::swap(quantity<T...> &other)
{
	using std::swap;
	swap(val, other.val);
}
template <class... T>
void quantity<T...>::swap(vquantity &other)
{
	swap(dynamic_cast<quantity<T...> &>(other));
}

template <class>
struct is_conc_cgroup_impl : public std::false_type
{
};
template <class... S>
struct is_conc_cgroup_impl<quantity<S...>> : public conserved::all_conserved_QuantiT<S...>
{
};

template <class... Groups>
inline vquantity &quantity<Groups...>::operator*=(const vquantity &other)
{
	return op(other);
}

template <class... Groups>
inline vquantity &quantity<Groups...>::operator+=(const vquantity &other)
{
	return op(other);
}

} // namespace quantit

template <class... Groups>
struct fmt::formatter<quantit::quantity<Groups...>>
{
	constexpr auto parse(format_parse_context &ctx)
	{
		auto it = ctx.begin(), end = ctx.end();
		if (it and it != end and *it != '}')
			throw format_error("invalid format, no formatting option for quantit::quantity");
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
		if (it and *it != '}')
			throw format_error("invalid format,closing brace missing");

		// Return an iterator past the end of the parsed range:
		return it;
	}

	template <typename FormatContext>
	auto format(const quantit::quantity<Groups...> &qt, FormatContext &ctx)
	{
		return format_to(ctx.out(), "[{}]", fmt::join(qt.val, ","));
	}
};

template <>
struct fmt::formatter<quantit::vquantity>
{
	constexpr auto parse(format_parse_context &ctx)
	{
		auto it = ctx.begin(), end = ctx.end();
		if (it and it != end and *it != '}')
			throw format_error("invalid format, no formatting option for quantit::quantity");
		if (it and *it != '}')
			throw format_error("invalid format,closing brace missing");

		// Return an iterator past the end of the parsed range:
		return it;
	}

	template <class FormatContext>
	auto format(const quantit::vquantity &qt, FormatContext &ctx) const
	{
		return qt.format_to(
		    ctx); // right now qt.format_to is only defined for fmt::format_context. Should work for any output stream.
	}
};

#endif /* F2547C1C_9177_4373_9C66_8D4C8621C7CC */
