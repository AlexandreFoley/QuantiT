/*
 * File: groups.h
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
#include "groups_utils.h"
#include "templateMeta.h"
#include <cstdint>
#include <ostream>

#include "cond_doctest.h"
namespace quantt
{
/**
 * Groups tend to have **very** short names in the litterature.
 * I want it to be easy to refer to litterature, so we use those short names.
 * So a namespace will protect us from name clashes.
 */
namespace groups
{

/**
 * C_N: the cyclic group with N elements.
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
	C(uint16_t _val)
	noexcept : val(_val)
	{
		val %= N; // only usage of modulo. that thing is expensive.
		          // this can be bad, if the input is greater than N,
		          // it is unclear that the user realize what they're doing.
		          // Perhaps they made a mistake, but the code will happily keep
		          // going.
	}
	void swap(C& other) noexcept
	{
		using std::swap;
		swap(other.val, val);
	}
	operator uint16_t() noexcept
	{
		return val;
	}
	C& operator+=(C other) noexcept
	{
		val += other.val;
		val -= (val >= N) * N;
		return *this;
	}
	// this function is what is actually used by the group compositor.
	C& op(C other) noexcept
	{
		return (*this) += other;
	}
	C& operator*=(C other) noexcept // in group theory we typically talk of a
	                                // product operator.
	{
		return (*this) += other;
	}
	friend C operator+(C lhs, C rhs) noexcept
	{
		return lhs += rhs;
	}
	friend C operator*(C lhs, C rhs) noexcept
	{
		return lhs *= rhs;
	}
	C& inverse_() noexcept
	{
		val = N - val;
		return *this;
	}
	C inverse() const noexcept
	{
		C out(*this);
		return out.inverse_();
	}

	bool operator==(C other) const noexcept
	{
		return val == other.val;
	}
	bool operator!=(C other) const noexcept
	{
		return val != other.val;
	}

	// compute z such that *this*other = z*(*this), and store the result in
	// other. Cn is Abelian, therefor this function does nothing. Necessary to
	// support non abelian groups.
	void commute(C& other) const {}
	void commute_(const C& other) {}

	friend std::ostream& operator<<(std::ostream& out, const C& c)
	{
		out << "grp::C<" << C::N << ">(" << c.val << ')';
		return out;
	}
};

/**
 * Abelian group formed by integers under the action of addition.
 * Useful for particles and spin conservation.
 * In principle Z has an infinite domain, but here it is limited
 * to [-32767,32767] by usage of int16_t in the implementation
 *
 * Note: In the case of particles conservation it is related to the U(1)
 * symmetry of the phase of the wavefunction. It is sometime (an abuse of
 * language) called U(1). U(1) is a continuous group with a finite domain while
 * N is a discrete group with an infinite domain. They are isomorphic.
 */
class Z
{
	int16_t val;

public:
	Z(int16_t _val)
	noexcept : val(_val) {}
	operator uint16_t() noexcept
	{
		return val;
	}
	void swap(Z& other) noexcept
	{
		using std::swap;
		swap(other.val, val);
	}
	Z& operator+=(Z other) noexcept
	{
		val += other.val;
		return *this;
	}
	Z& operator*=(Z other) noexcept // in group theory we typically talk of a
	                                // product operator.
	{
		return (*this) += other;
	}
	friend Z operator+(Z lhs, Z rhs) noexcept
	{
		return lhs += rhs;
	}
	friend Z operator*(Z lhs, Z rhs) noexcept
	{
		return lhs *= rhs;
	}
	// Z& op( other) is the function used by cgroup.
	Z& op(Z other) { return (*this) += other; }
	Z& inverse_() noexcept
	{
		val = -val;
		return *this;
	}
	Z inverse() const noexcept
	{
		Z out(*this);
		return out.inverse_();
	}
	bool operator==(Z other) const
	{
		return val == other.val;
	}
	bool operator!=(Z other) const
	{
		return val != other.val;
	}

	// compute u such that (*this)*other = u*(*this), and store the result in
	// other. Z is abelian, therefore this function does nothing
	void commute(Z& other) const {}
	void commute_(const Z& other) {}
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

// using is_group = and_<has_op<T>, has_inverse_<T>, has_commute<T>,
// has_commute_<T>, has_comparatorequal<T>, has_comparatornotequal<T>>;
static_assert(is_group_v<Z>, "Z isn't a group?! something is very wrong");
static_assert(is_group_v<C<5>>, "C<5> isn't a group?! something is very wrong");

} // namespace groups

TEST_CASE("simple groups")
{
	using namespace groups;
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
