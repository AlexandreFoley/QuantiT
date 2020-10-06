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
#include <ostream>

#include "doctest/cond_doctest.h"
namespace quantt
{
/**
 * Groups tend to have **very** short names in the litterature.
 * I want it to be easy to refer to litterature, so we use those short names.
 * So a namespace will protect us from name clashes.
 */
namespace conserved
{

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
	constexpr static bool is_Abelian = true; //Make sure that your group is actually Abelian. I can't think of a way to check this property in finite time using the compiler.

	constexpr C(uint16_t _val) // constexpr value contructor, necessary for one of the checks for any_quantity
	    noexcept : val(_val)
	{
		val %= N; // only usage of modulo. that thing is expensive.
		          // this can be bad, if the input is greater than N,
		          // it is unclear that the user realize what they're doing.
		          // Perhaps they made a mistake, but the code will happily keep
		          // going.
	}
	constexpr C() : val(0) {} //default to the neutral element.

	void swap(C& other) noexcept
	{
		using std::swap;
		swap(other.val, val);
	}
	constexpr operator uint16_t() noexcept
	{
		return val;
	}
	constexpr C& operator+=(C other) noexcept
	{
		val += other.val;
		val -= (val >= N) * N;
		return *this;
	}
	// this function is what is actually used by the group compositor.
	constexpr C& op(C other) noexcept
	{
		return (*this) += other;
	}
	constexpr C& operator*=(C other) noexcept // in group theory we typically talk of a
	                                          // product operator.
	{
		return (*this) += other;
	}
	constexpr friend C operator+(C lhs, C rhs) noexcept
	{
		return lhs += rhs;
	}
	friend constexpr C operator*(C lhs, C rhs) noexcept
	{
		return lhs *= rhs;
	}
	constexpr C& inverse_() noexcept
	{
		val = N - val;
		return *this;
	}
	constexpr C inverse() const noexcept
	{
		C out(*this);
		return out.inverse_();
	}

	constexpr bool operator==(C other) const noexcept
	{
		return val == other.val;
	}
	constexpr bool operator!=(C other) const noexcept
	{
		return val != other.val;
	}

	friend std::ostream& operator<<(std::ostream& out, const C& c)
	{
		out << "grp::C<" << C::N << ">(" << c.val << ')';
		return out;
	}
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
	constexpr static bool is_Abelian = true; //Tag that the conserved quantity emerge from an Abelian group.
	//conserved quantities that emerge from non-abelian symmetries are currently not supported.
	//This tag MUST be set to true for the conserved quantitie class to be accepted.
	constexpr Z(int16_t _val) // constexpr value contructor, necessary for one of the checks for any_quantity
	    noexcept : val(_val)
	{
	}
	constexpr Z() : val(0) {} // constexpr value contructor, necessary for one of the checks for any_quantity
	constexpr operator uint16_t() noexcept
	{
		return val;
	}
	void swap(Z& other) noexcept
	{
		using std::swap;
		swap(other.val, val);
	}
	constexpr Z& operator+=(Z other) noexcept
	{
		val += other.val;
		return *this;
	}
	constexpr Z& operator*=(Z other) noexcept // in group theory we typically talk of a
	                                          // product operator.
	{
		return (*this) += other;
	}
	friend constexpr Z operator+(Z lhs, Z rhs) noexcept
	{
		return lhs += rhs;
	}
	friend constexpr Z operator*(Z lhs, Z rhs) noexcept
	{
		return lhs *= rhs;
	}
	// Z& op( other) is the function used by any_quantity.
	constexpr Z& op(Z other) { return (*this) += other; }
	constexpr Z& inverse_() noexcept
	{
		val = -val;
		return *this;
	}
	constexpr Z inverse() const noexcept //must be constexpr, allow compiler to verify a necessary property
	{
		Z out(*this);
		return out.inverse_();
	}
	constexpr bool operator==(Z other) const //must be constexpr, allow compiler to verify a necessary property
	{
		return val == other.val;
	}
	constexpr bool operator!=(Z other) const
	{
		return val != other.val;
	}

	friend std::ostream& operator<<(std::ostream& out, const Z& c);
};

inline void swap(Z& lhs, Z& rhs) noexcept
{
	lhs.swap(rhs);
}
template <uint16_t N>
void swap(C<N>& lhs, C<N>& rhs) noexcept
{
	lhs.swap(rhs);
}

// using is_conversed_quantt =     and_<default_to_neutral<T>, has_op<T>, has_inverse_<T>,
//         has_comparatorequal<T>, has_comparatornotequal<T>, is_Abelian<T>>;
static_assert(has_constexpr_equal<Z>::value, "debug");

static_assert(is_conserved_quantt_v<Z>, "Z isn't a group?! something is very wrong");
static_assert(is_conserved_quantt_v<C<5>>, "C<5> isn't a group?! something is very wrong");

} // namespace conserved

TEST_CASE("simple conserved")
{
	using namespace conserved;
	C<2> c2_1(1);
	C<2> c2_0(0);
	C<2> c2_11 = c2_1 * c2_1;
	CHECK(c2_0 == c2_11);

	C<5> c5_3(3);
	C<5> c5_2(2);
	CHECK(c5_3 != c5_2);
	CHECK(
	    c5_3.inverse() * c5_3 ==
	    C<5>(
	        0)); // the product with one's own inverse give the trivial element.
	CHECK(c5_3.inverse() == c5_2);
	CHECK(c5_2.inverse() == c5_3);
	CHECK(c5_2.inverse().inverse() ==
	      c5_2); // inverse twice gives back the original value
	CHECK(C<5>(c5_2).inverse_().inverse_() ==
	      c5_2); // inverse in place twice gives back the original value
	CHECK(c5_2.op(c5_2) == C<5>(4));

	Z Z_1(1);
	Z Z_2(2);
	Z Z_11 = Z_1 * Z_1;
	CHECK(Z_2 == Z_11);

	Z Z_3(3);
	Z Z_m3(-3);
	CHECK(Z_3 != Z_m3);
	CHECK(Z_3.inverse() * Z_3 ==
	      Z(0)); // the product with one's own inverse give the trivial element.
	CHECK(Z_3.inverse() == Z_m3);
	CHECK(Z_m3.inverse() == Z_3);
	CHECK(Z_m3.inverse().inverse() ==
	      Z_m3); // inverse twice gives back the original value
	CHECK(Z(Z_m3).inverse_().inverse_() ==
	      Z_m3); // inverse in place twice gives back the original value
	CHECK(Z_3.op(Z_3) == Z(6));
}

} // namespace quantt

#endif /* EF30AFAC_8403_46CD_A139_264F626DA567 */
