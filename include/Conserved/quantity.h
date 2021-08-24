/*
 * File:quantity.h
 * Project: quantt
 * File Created: Tuesday, 15th September 2020 12:19:54 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Tuesday, 15th September 2020 12:19:54 pm
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#ifndef EF30AFAC_8403_46CD_A139_264F626DA567
#define EF30AFAC_8403_46CD_A139_264F626DA567
#include "Conserved/quantity_utils.h"
#include "templateMeta.h"
#include <cstdint>
#include <fmt/core.h>
#include <fmt/format.h>
#include <ostream>

#include "doctest/doctest_proxy.h"
namespace quantt
{
/**
 * Groups tend to have **very** short names in the litterature.
 * I want it to be easy to refer to litterature, so we use those short names.
 * So a namespace will protect us from name clashes.
 */
namespace conserved
{

namespace
{

template <class T>
constexpr int64_t distance2_impl(T a, T b)
{
	auto x = static_cast<int64_t>(a.get_val()) - b.get_val();
	return x * x;
}

template <class T>
constexpr double distance_impl(T a, T b)
{
	return std::sqrt(distance2(a, b));
}

} // anonymous namespace

/**
 * @brief C_N, the cyclic group with N elements. The conserved quantities
 * associated with discrete rotationnal symmetry are element of this familly of groups.
 *
 * Oftentime called Z_N in the literature.
 * This would create a name clash with the infinite group Z.
 *
 * Note that the implementation limits the length of the cycle to less than
 * 2^16-1
 */
template <uint16_t mod>
class C
{
  public:
	static constexpr uint16_t N = mod;

  private:
	static_assert(N > 0, "only value greater than zero make sense, only "
	                     "greater than 1 are useful.");
	uint16_t val;

  public:
	constexpr static bool is_Abelian = true; // Make sure that your group is actually Abelian. I can't think of a way to
	                                         // check this property in finite time using the compiler.

	// Can't have this one with the signed integer case because of interference from implicit conversions of fundamental
	// types explicit constexpr C(uint16_t _val) // constexpr value contructor, necessary for one of the checks for
	// any_quantity
	//     noexcept
	//     : val(_val)
	// {
	// 	val %= N; // only usage of modulo. that thing is expensive.
	// 	          // this can be bad, if the input is greater than N,
	// 	          // it is unclear that the user realize what they're doing.
	// 	          // Perhaps they made a mistake, but the code will happily keep
	// 	          // going.
	// }
	/**
	 * @brief constructor using singed integers
	 *
	 * allow construction using minus sign as the inverse operation.
	 *
	 *
	 */
	constexpr C(int16_t _val) noexcept : val((_val < 0) * N + (_val % N))
	{
		/*
		 * A quick explanation of the init line for val:
		 * The modulo operation on signed integers preserve the sign of the value, and give the same absolute value for
		 * both possible input sign. To get the correct (positive) value, the inverse of the absolute of the input, we
		 * must first take the modulo of the negative number to bring it in range ]-N,N[. Then if it is negative, we add
		 * N to bring it in the range [0,N[, and assign to the unsigned storage of this type. After this initial
		 * assigment business, no modulo operation are ever needed.
		 */
	}
	constexpr C() : val(0) {} // default to the neutral element.

	void swap(C &other) noexcept
	{
		using std::swap;
		swap(other.val, val);
	}
	constexpr operator uint16_t() const noexcept { return val; }
	constexpr uint16_t get_val() const noexcept { return val; }
	constexpr C &operator+=(C other) noexcept { return op(other); }
	// this function is what is actually used by the group compositor.
	constexpr C &op(C other) noexcept { return op(other, true); }

	constexpr C &op(C other, bool cond) noexcept
	{
		val += cond * other.val;
		val -= (val >= N) * N;
		return *this;
	}

	constexpr C &operator*=(C other) noexcept // in group theory we typically talk of a
	                                          // product operator.
	{
		return (*this) += other;
	}
	constexpr friend C operator+(C lhs, C rhs) noexcept { return lhs += rhs; }
	friend constexpr C operator*(C lhs, C rhs) noexcept { return lhs *= rhs; }
	constexpr C &inverse_() noexcept
	{
		val = bool(val) * (N - val);
		return *this;
	}
	constexpr C inverse() const noexcept
	{
		C out(*this);
		return out.inverse_();
	}
	int64_t distance2(C other) const 
	{
		return distance2_impl(*this, other);
	}
	double distance( C other) const
	{
		return distance_impl(*this, other);
	}

	constexpr bool operator==(C other) const noexcept { return val == other.val; }
	constexpr bool operator<(C other) const noexcept { return val < other.val; }
	constexpr bool operator>(C other) const noexcept { return val > other.val; }
	constexpr bool operator!=(C other) const noexcept { return val != other.val; }

	friend std::ostream &operator<<(std::ostream &out, const C &c)
	{
		out << fmt::format("grp::C<{}>({})", C::N, c.val);
		return out;
	}
	friend struct fmt::formatter<quantt::conserved::C<N>>;
};

/**
 * Abelian group formed by integers under the action of addition.
 * Useful for particles and spin conservation (along the quantization axis only).
 * In principle Z has an infinite domain, but here it is limited
 * to [-32767,32767] by usage of int16_t in the implementation
 *
 * This group can represent the conservation of the charges associated with U1 symmetries
 */
class Z
{
	int16_t val;

  public:
	constexpr static bool is_Abelian = true; // Tag that the conserved quantity emerge from an Abelian group.
	// conserved quantities that emerge from non-abelian symmetries are currently not supported.
	// This tag MUST be set to true for the conserved quantitie class to be accepted.
	constexpr Z(int16_t _val) // constexpr value contructor, necessary for one of the checks for any_quantity
	    noexcept
	    : val(_val)
	{
	}
	constexpr Z() : val(0) {} // constexpr value contructor, necessary for one of the checks for any_quantity
	constexpr operator int16_t() noexcept { return val; }
	constexpr int16_t get_val() const noexcept { return val; }
	void swap(Z &other) noexcept
	{
		using std::swap;
		swap(other.val, val);
	}
	constexpr Z &operator+=(Z other) noexcept { return op(other); }
	constexpr Z &operator*=(Z other) noexcept // in group theory we typically talk of a
	                                          // product operator.
	{
		return (*this) += other;
	}
	friend constexpr Z operator+(Z lhs, Z rhs) noexcept { return lhs += rhs; }
	friend constexpr Z operator*(Z lhs, Z rhs) noexcept { return lhs *= rhs; }
	// Z& op( other) is the function used by any_quantity.
	constexpr Z &op(Z other) noexcept { return op(other, true); }
	constexpr Z &op(Z other, bool cond) noexcept
	{

		val += cond * other.val;
		return *this;
	}
	constexpr Z &inverse_() noexcept
	{
		val = -val;
		return *this;
	}
	constexpr Z inverse() const noexcept // must be constexpr, allow compiler to verify a necessary property
	{
		Z out(*this);
		return out.inverse_();
	}
	constexpr bool operator==(Z other) const // must be constexpr, allow compiler to verify a necessary property
	{
		return val == other.val;
	}
	int64_t distance2(Z other) const
	{
		return distance2_impl(*this, other);
	}
	double distance( Z other) const
	{
		return distance_impl(*this, other);
	}
	constexpr bool operator!=(Z other) const { return val != other.val; }
	constexpr bool operator<(Z other) const { return val < other.val; }
	constexpr bool operator>(Z other) const { return val > other.val; }
	friend std::ostream &operator<<(std::ostream &out, const Z &c);
	friend struct fmt::formatter<quantt::conserved::Z>;
};


template <uint16_t N>
int64_t distance2(C<N> a, C<N> b)
{
	return distance2_impl(a, b);
}
template <uint16_t N>
double distance(C<N> a, C<N> b)
{
	return distance_impl(a, b);
}
inline int64_t distance2(Z a, Z b) { return distance2_impl(a, b); }
inline double distance(Z a, Z b) { return distance_impl(a, b); }

inline void swap(Z &lhs, Z &rhs) noexcept { lhs.swap(rhs); }
template <uint16_t N>
void swap(C<N> &lhs, C<N> &rhs) noexcept
{
	lhs.swap(rhs);
}

// using is_conversed_quantt =     and_<default_to_neutral<T>, has_op<T>, has_inverse_<T>,
//         has_comparatorequal<T>, has_comparatornotequal<T>, is_Abelian<T>>;
static_assert(has_constexpr_equal<Z>::value, "debug");

static_assert(is_conserved_quantt_v<Z>, "Z isn't a group?! something is very wrong");
static_assert(is_conserved_quantt_v<C<5>>, "C<5> isn't a group?! something is very wrong");

} // namespace conserved

qtt_TEST_CASE("simple conserved")
{
	using namespace conserved;
	qtt_SUBCASE("Cyclical conserved values")
	{
		C<2> c2_1(1);
		C<2> c2_0(0);
		C<2> c2_11 = c2_1 * c2_1;
		qtt_CHECK(c2_0 == c2_11);

		C<5> c5_3(3);
		C<5> c5_2(2);
		qtt_CHECK(c5_3 != c5_2);
		qtt_CHECK(c5_3.inverse() * c5_3 == C<5>(0)); // the product with one's own inverse give the trivial element.
		qtt_CHECK(c5_3.inverse() == c5_2);
		qtt_CHECK(c5_2.inverse() == c5_3);
		qtt_CHECK(c5_2.inverse().inverse() == c5_2);         // inverse twice gives back the original value
		qtt_CHECK(C<5>(c5_2).inverse_().inverse_() == c5_2); // inverse in place twice gives back the original value
		qtt_CHECK(c5_2.op(c5_2) == C<5>(4));
		qtt_CHECK(c5_2.op(c5_2, false) == C<5>(4));
	}
	qtt_SUBCASE("signed integer conserved values")
	{
		Z Z_1(1);
		Z Z_2(2);
		Z Z_11 = Z_1 * Z_1;
		qtt_CHECK(Z_2 == Z_11);

		Z Z_3(3);
		Z Z_m3(-3);
		qtt_CHECK(Z_3 != Z_m3);
		qtt_CHECK(Z_3.inverse() * Z_3 == Z(0)); // the product with one's own inverse give the trivial element.
		qtt_CHECK(Z_3.inverse() == Z_m3);
		qtt_CHECK(Z_m3.inverse() == Z_3);
		qtt_CHECK(Z_m3.inverse().inverse() == Z_m3);      // inverse twice gives back the original value
		qtt_CHECK(Z(Z_m3).inverse_().inverse_() == Z_m3); // inverse in place twice gives back the original value
		qtt_CHECK(Z_3.op(Z_3) == Z(6));
		qtt_CHECK(Z_3.op(Z_3, false) == Z(6));
	}
	qtt_SUBCASE("distance")
	{
		Z Z_1(1);
		Z Z_2(2);
		qtt_CHECK(distance2(Z_1,Z_2) == 1);	
		C<2> C2_1(1);
		C<2> C2_0(0);
		qtt_CHECK(distance2(C2_1,C2_0) == 1);
	}
}

} // namespace quantt

template <uint16_t N>
struct fmt::formatter<quantt::conserved::C<N>>
{
	constexpr auto parse(format_parse_context &ctx)
	{
		auto it = ctx.begin(), end = ctx.end();
		if (it)
		{
			if (it != end and *it != '}')
				throw format_error("invalid format, no formatting option for quantt::quantity");
			if (*it != '}')
				throw format_error("invalid format,closing brace missing");
		}
		// Return an iterator past the end of the parsed range:
		return it;
	}

	template <class FormatContext>
	auto format(const quantt::conserved::C<N> &z, FormatContext &ctx)
	{

		return format_to(
		    ctx.out(), "C<{}>({})", N,
		    z.val); // right now qt.format_to is only define for fmt::format_context. Should work for any output stream.
	}
};
template <>
struct fmt::formatter<quantt::conserved::Z>
{
	constexpr auto parse(format_parse_context &ctx)
	{
		auto it = ctx.begin(), end = ctx.end();
		if (it)
		{
			if (it != end and *it != '}')
				throw format_error("invalid format, no formatting option for quantt::conserved::Z");
			if (*it != '}')
				throw format_error("invalid format,closing brace missing");
		}
		// Return an iterator past the end of the parsed range:
		return it;
	}

	template <class FormatContext>
	auto format(const quantt::conserved::Z &z, FormatContext &ctx)
	{
		return format_to(
		    ctx.out(), "Z({})",
		    z.val); // right now qt.format_to is only define for fmt::format_context. Should work for any output stream.
	}
};
#endif /* EF30AFAC_8403_46CD_A139_264F626DA567 */
