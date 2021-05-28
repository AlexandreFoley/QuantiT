/*
 * File: TensorGroup.h
 * Project: quantt
 * File Created: Tuesday, 1st September 2020 1:39:16 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Tuesday, 1st September 2020 1:39:17 pm
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

// TODO: replace the dynamic_cast<(const) quantity&> with a custom
// function that allow for
//      a more useful customized message when a bad cast happen.

#ifndef D56E4C12_98E1_4C9E_B0C4_5B35A5A3CD17
#define D56E4C12_98E1_4C9E_B0C4_5B35A5A3CD17

#include "Conserved/Composite/quantity_impl.h"
#include "Conserved/quantity.h"
#include "templateMeta.h"
#include <cstdint>
#include <fmt/core.h>
#include <ostream>
#include <type_traits>
#include <utility>

namespace quantt
{
// doing away with those reference like class won't be easy. First attempt was a failure.
// TODO: Do away with those reference class. They proving ever more troublesome.
class any_quantity_ref;
class any_quantity_cref;
class any_quantity;
/**
 * @brief wrapper for the polymorphic composite groups from vquantity.h that provide value semantics
 * We expect that for a given problem, the type of the underlying cgoup_impl used will be uniform.
 *
 */
class any_quantity final
{
	std::unique_ptr<vquantity> impl;

	friend any_quantity_ref;
	friend any_quantity_cref;

  public:
	any_quantity(const vquantity &other) : impl(other.clone()) {}
	// template <class... Groups, class = std::enable_if_t<conserved::all_group_v<Groups...>>>
	// any_quantity(const quantity<Groups...>& other) : impl(other.clone()) {}
	any_quantity(std::unique_ptr<vquantity> &&_impl) : impl(std::move(_impl)) {}

	/**
	 * @brief Construct a new any_quantity object. It compose the group element it receive.
	 *
	 * @tparam Groups types of the different group we're composing
	 * @tparam std::enable_if_t<groups::all_group_v<Groups...>>
	 *     verify that the arguements all satisfy the required group interface.
	 * @param grps values used to initialized the group
	 */

	template <class... Groups,
	          class = std::enable_if_t< // instantiate this template only if the following condition are satisfied:
	              !std::disjunction_v<std::is_same<Groups, any_quantity_cref>..., // refuse any reference type
	                                     std::is_same<Groups, any_quantity_ref>...,  // refuse any ref type
	                                     std::is_same<Groups,  any_quantity>...>      // no recursion.
	              and conserved::all_group_v<Groups...> > > // all element satisfy the constraint of an abelian group
	any_quantity(Groups... grps) : impl(std::make_unique<quantity<Groups...>>(grps...))
	{
		static_assert(conserved::all_group_v<Groups...> and
		                  not std::disjunction_v<std::is_same<Groups, any_quantity_cref>...,
		                                         std::is_same<Groups, any_quantity_ref>...,
		                                         std::is_same<Groups, any_quantity>...>,
		              "Don't bypass the enable if.");
	}

	/**
	 * @brief Construct a new any_quantity object, initialized with a  representation of the trivial group.
	 *
	 */
	any_quantity(); // default to a group that contain only the neutral element. avoid having an unitialized unique_ptr.
	any_quantity(const any_quantity &other) : impl(other.impl->clone()) {}
	any_quantity(any_quantity_cref other); // explicit to avoid accidental copies and ambiguous overloads
	any_quantity(any_quantity_ref other);  // explicit to avoid accidental copies and ambiguous overloads
	any_quantity(any_quantity &&) = default;

	vquantity &get();
	const vquantity &get() const;

	/**
	 * @brief swap function with reference to any_quantity. Always work.
	 *
	 * @param other
	 */
	void swap(any_quantity &other) noexcept;

	void swap(any_quantity_ref other); // only work if the underlying types ARE the same.

	/**
	 * @brief generate the neutral element of the underlying gorup.
	 *
	 * @return any_quantity
	 */
	any_quantity neutral() const;
	/**
	 * @brief assigment operators, copy the value into this.
	 * These assigment operator can change the underlying type of *this.
	 *
	 * @param other: a any_quantity, any_quantity_ref or any_quantity_cref
	 * @return any_quantity& :reference to *this
	 */
	any_quantity_ref operator=(any_quantity_cref other);
	any_quantity_ref operator=(any_quantity_ref other);
	any_quantity_ref operator=(const any_quantity &other);
	any_quantity_ref operator=(any_quantity &&other);
	/**
	 * @brief implicit conversion to the reference to const type
	 *
	 * @return any_quantity_cref: a reference to this constant object
	 */
	operator any_quantity_cref() const;
	/**
	 * @brief implicit conversion to the reference type
	 *
	 * @return any_quantity_ref : a reference to this object
	 */
	operator any_quantity_ref();

	~any_quantity() {}
	/**
	 * @brief In place group operation
	 *
	 * @param other : constant reference to another composite group of the same type.
	 *                work with any_quantity, any_quantity_ref and any_quantity_cref for input.
	 * @return any_quantity& reference to *this
	 */
	any_quantity_ref operator*=(any_quantity_cref other);
	any_quantity_ref operator+=(any_quantity_cref other);
	any_quantity_ref op(any_quantity_cref other, bool cond);
	/**
	 * @brief out of place group operation, on two composite group element of the same type
	 *
	 * @param lhs: can be a any_quantity,any_quantity_ref or any_quantity_cref
	 * @param rhs: can be a any_quantity,any_quantity_ref or any_quantity_cref
	 * @return any_quantity a new any_quantity object.
	 */
	friend any_quantity operator*(any_quantity_cref lhs, any_quantity_cref rhs);
	// friend any_quantity operator*(any_quantity_cref lhs, any_quantity&& rhs);
	friend any_quantity operator+(any_quantity_cref lhs, any_quantity_cref rhs);
	// friend any_quantity operator+(any_quantity_cref lhs, any_quantity&& rhs);

	/**
	 * @brief in place inverse
	 *
	 * @return any_quantity&
	 */
	any_quantity_ref inverse_();
	/**
	 * @brief return a new any_quantity object that is the inverse of this.
	 *
	 * @return any_quantity
	 */
	any_quantity inverse() const;

	/**
	 * @brief equality comparison operator
	 *
	 * @param lhs : works with any_quantity,any_quantity_ref and any_quantity_cref
	 * @param rhs : works with any_quantity,any_quantity_ref and any_quantity_cref
	 * @return true : same group element
	 * @return false : different group element
	 */
	friend bool operator==(any_quantity_cref lhs, any_quantity_cref rhs);
	/**
	 * @brief different comparison operator
	 *
	 * @param lhs : works with any_quantity,any_quantity_ref and any_quantity_cref
	 * @param rhs : works with any_quantity,any_quantity_ref and any_quantity_cref
	 * @return true : different group element
	 * @return false : same group element
	 */
	friend bool operator!=(any_quantity_cref lhs, any_quantity_cref rhs);
};
/**
 * @brief Constant reference type for any_quantity.
 *
 * Implements only the const method of any_quantity. consult any_quantity for methods documentation
 *
 * Made to facilitate element by element manipulation within a specialize
 * container for vquantity.
 */
class any_quantity_cref
{
	const vquantity *ref;
	friend any_quantity_ref;
	friend any_quantity;

  public:
	any_quantity_cref() = delete;
	any_quantity_cref(const vquantity &other) : ref(&other) {}
	any_quantity_cref(const any_quantity_cref &other) : ref(other.ref) {}
	any_quantity_cref(const vquantity *other) : ref(other) {}

	any_quantity neutral() const;
	const vquantity &get() const;
	any_quantity inverse() const;

	operator const vquantity &() const { return get(); }
};
/**
 * @brief Reference type for any_quantity.
 *
 * Implements all the method of any_quantity. consult any_quantity for methods documentation
 *
 * Made to facilitate element by element manipulation within a specialize
 * container for vquantity.
 */
class any_quantity_ref final
{
	vquantity *ref;
	friend any_quantity_cref;
	friend any_quantity;

  public:
	any_quantity_ref() = delete;
	any_quantity_ref(const any_quantity_ref &) = delete; // for clarity: you can't do that one. would cast away the
	                                                     // const character.
	any_quantity_ref(any_quantity_ref &other) : ref(other.ref) {}
	any_quantity_ref(vquantity &other) : ref(&other) {}
	any_quantity_ref(vquantity *other) : ref(other) {}
	operator any_quantity_cref() const;
	any_quantity_ref &operator=(any_quantity_cref other);
	any_quantity_ref &operator=(const any_quantity &other);

	any_quantity neutral() const;
	const vquantity &get() const;
	vquantity &get();

	any_quantity_ref &operator*=(any_quantity_cref other);
	any_quantity_ref &operator+=(any_quantity_cref other);
	any_quantity_ref &op(any_quantity_cref other, bool cond);
	any_quantity_ref &op(any_quantity_cref other);
	any_quantity_ref &inverse_();
	any_quantity inverse() const;
	void swap(any_quantity_ref other);

	operator const vquantity &() const { return get(); }
	operator vquantity &() { return get(); }
};

// method definitions in no particular order...
inline const vquantity &any_quantity_cref::get() const { return *ref; }
inline any_quantity any_quantity_cref::inverse() const { return any_quantity(*this).inverse_(); }
inline any_quantity_ref &any_quantity_ref::inverse_()
{
	get().inverse_();
	return *this;
}
inline any_quantity any_quantity_ref::inverse() const { return any_quantity(ref->clone()).inverse_(); }
inline any_quantity any_quantity::neutral() const { return any_quantity(impl->neutral()); }
inline any_quantity any_quantity_ref::neutral() const { return any_quantity(get().neutral()); }
inline any_quantity any_quantity_cref::neutral() const { return any_quantity(get().neutral()); }

inline void any_quantity_ref::swap(any_quantity_ref other) { ref->swap(other.get()); }
inline any_quantity::any_quantity(any_quantity_cref other) : impl(other.get().clone()) {}
inline any_quantity::any_quantity(any_quantity_ref other) : impl(other.get().clone()) {}

inline void any_quantity::swap(any_quantity &other) noexcept
{
	using std::swap;
	swap(other.impl, impl);
}
inline any_quantity_ref any_quantity::operator=(
    any_quantity &&other) // will work even when the underlying type isn't the same...
{
	swap(other);
	return *this;
}
inline any_quantity_ref any_quantity::inverse_()
{
	impl->inverse_();
	return *this;
}
inline any_quantity any_quantity::inverse() const { return any_quantity(*this).inverse_(); }
inline any_quantity_ref any_quantity::operator=(const any_quantity &other)
{
	*impl = *other.impl;
	return *this;
}
inline void any_quantity::swap(any_quantity_ref other) { impl->swap(other.get()); }
inline void swap(any_quantity &lhs, any_quantity &rhs) { lhs.swap(rhs); }
inline void swap(any_quantity_ref lhs, any_quantity_ref rhs) { lhs.swap(rhs); }
inline any_quantity::operator any_quantity_cref() const { return any_quantity_cref(*impl.get()); }
inline any_quantity::operator any_quantity_ref() { return any_quantity_ref(*impl.get()); }
inline any_quantity_ref any_quantity::operator*=(any_quantity_cref other)
{
	impl->op(other.get());
	return *this;
}
inline any_quantity operator*(any_quantity_cref lhs, any_quantity_cref rhs) { return any_quantity(lhs) *= rhs; }
inline any_quantity_ref any_quantity::op(any_quantity_cref other, bool cond)
{
	impl->op(other.get(), cond);
	return *this;
}
// any_quantity operator*(any_quantity_cref lhs, any_quantity&& rhs)
// {
// 	lhs.get().op_to(*rhs.impl);
// 	return rhs;
// }
inline any_quantity operator+(any_quantity_cref lhs, any_quantity_cref rhs) { return lhs * rhs; }
// any_quantity operator+(any_quantity_cref lhs, any_quantity&& rhs)
// {
// 	return lhs * rhs;
// }
inline any_quantity_ref any_quantity::operator=(any_quantity_cref other)
{
	impl = other.get().clone(); // allocate a new thing or copy the value? allocating a new thing allow changing the
	                            // underlying type
	// allocate a new thing, without easily changing the underlying type, we cannot have a default initialization for
	// any_quantity.
	// this make for wonky and fiddly code in many situation.
	// The only other valid option is to test the type and take a decision.
	return *this;
}
inline any_quantity_ref any_quantity::operator=(any_quantity_ref other) { return operator=(any_quantity_cref(other)); }
inline any_quantity_ref any_quantity::operator+=(any_quantity_cref other) { return (*this) *= other; }

inline bool operator!=(any_quantity_cref left, any_quantity_cref right) { return left.get() != (right.get()); }
inline bool operator==(const any_quantity_cref left, const any_quantity_cref right)
{
	return left.get() == (right.get());
}

inline const vquantity &any_quantity_ref::get() const { return *ref; }
inline vquantity &any_quantity_ref::get() { return *ref; }
inline any_quantity_ref::operator any_quantity_cref() const { return any_quantity_cref(*ref); }
inline any_quantity_ref& any_quantity_ref::operator=(any_quantity_cref other)
{
	// auto a = get() * other.get();
	// fmt::print("operator= of any_quantity_ref is very unreliable");
	ref->operator=(other.get());
	return *this;
}
inline any_quantity_ref &any_quantity_ref::operator=(const any_quantity &other)
{
	*ref = other.get();
	return *this;
}
inline any_quantity_ref &any_quantity_ref::operator*=(const any_quantity_cref other)
{
	get().op(other.get());
	return *this;
}

inline any_quantity_ref &any_quantity_ref::operator+=(const any_quantity_cref other) { return (*this) *= other; }
inline vquantity &any_quantity::get() { return *impl; }
inline const vquantity &any_quantity::get() const { return *impl; }
inline bool operator<(any_quantity_cref left, any_quantity_cref right) { return left.get().operator<(right); }
inline bool operator>(any_quantity_cref left, any_quantity_cref right) { return left.get().operator>(left); }

qtt_TEST_CASE("composite conserved")
{
	using namespace quantt;
	using namespace conserved;
	any_quantity A(C<2>(0), Z(3)); // order matters.
	any_quantity B(C<2>(1), Z(-1));
	using ccgroup = quantity<C<2>, Z>;
	auto EFF = any_quantity(ccgroup(0, 0));
	qtt_CHECK_NOTHROW(auto a = any_quantity(ccgroup(0, 0)));
	any_quantity A_copy(A);
	any_quantity B_copy(B);
	qtt_CHECK_NOTHROW(auto c = A + B);
	any_quantity D(Z(0), C<2>(1)); // D has a different underlying type.
	// the cast to void silences warnings about unused return values and
	// comparison. We know, it's ok.
	qtt_CHECK_THROWS_AS((void)(D == A),
	                    const std::bad_cast &); // A and D have different derived
	                                            // type: they're not compatible
	qtt_CHECK_THROWS_AS((void)(D != A),
	                    const std::bad_cast &); // A and D have different derived
	                                            // type: they're not compatible
	qtt_CHECK_THROWS_AS((void)(D * A),
	                    const std::bad_cast &); // A and D have different derived
	                                            // type: they're not compatible
	qtt_CHECK_THROWS_AS(D *= A,
	                    const std::bad_cast &); // A and D have different derived
	                                            // type: they're not compatible
	qtt_CHECK_THROWS_AS(D += A,
	                    const std::bad_cast &); // A and D have different derived
	                                            // type: they're not compatible
	qtt_CHECK_THROWS_AS((void)(D + A),
	                    const std::bad_cast &); // A and D have different derived
	                                            // type: they're not compatible
	any_quantity_ref A_ref(A);
	any_quantity_cref A_cref(A);
	any_quantity_cref B_cref(B);
	any_quantity_ref B_ref(B);
	// any_quantity_ref is a drop-in replacement for a reference to any_quantity
	// any_quantity_cref is a drop-in-replacement for a reference to a constant
	// any_quantity. Those two classes exists to facilitate manipulation of single
	// any_quantity located within a special container for this polymorphic type. We
	// make extensive use of the reference type within the tests specifically to
	// verify the correctness of their implementation. We advise avoiding those
	// two reference type whenever possible. const any_quantity& and any_quantity& are
	// perfectly fine for most purpose.
	qtt_CHECK_NOTHROW(A_copy = A_ref);
	qtt_CHECK_NOTHROW(A_copy = A_cref);
	// Checking that the ref type doesn't loose track of its target.
	qtt_CHECK(A_cref == A_ref);
	qtt_CHECK(A_ref == A_cref);
	qtt_CHECK(A_ref == A);
	qtt_CHECK_NOTHROW(A_ref *= B);
	qtt_CHECK(A_ref == A);
	qtt_CHECK(A_ref == A_cref);
	qtt_CHECK_NOTHROW(A_ref += B);
	qtt_CHECK(A_ref == A);
	qtt_CHECK(A_ref == A_cref);
	A_copy = A;
	qtt_CHECK(B == B_copy); // commute_ act on the the calling object only.
	qtt_CHECK(A_copy == A); // this is an abelian group.
	qtt_SUBCASE("Cast ambiguity")
	{
		// In this subcase with test different combination of any_quantity,any_quantity_ref
		// and any_quantity_cref to make sure all operation resolve correctly. Because
		// of the different type and the ways in which they are equivalent, a
		// mistake could lead to some operations failing.
		qtt_CHECK_NOTHROW(auto _t = A * B);
		qtt_CHECK_NOTHROW(auto _t = A_ref * B);
		qtt_CHECK_NOTHROW(auto _t = A_cref * B);
		qtt_CHECK_NOTHROW(auto _t = A * B_ref);
		qtt_CHECK_NOTHROW(auto _t = A_ref * B_ref);
		qtt_CHECK_NOTHROW(auto _t = A_cref * B_ref);
		qtt_CHECK_NOTHROW(auto _t = A * B_cref);
		qtt_CHECK_NOTHROW(auto _t = A_ref * B_cref);
		qtt_CHECK_NOTHROW(auto _t = A_cref * B_cref);
		qtt_CHECK_NOTHROW(auto _t = A * B_cref.inverse());     // this should use the
		                                                       // operator*(any_quantity_cref&,any_quantity&&);
		qtt_CHECK_NOTHROW(auto _t = A_ref * B.inverse());      // this should use the
		                                                       // operator*(any_quantity_cref&,any_quantity&&);
		qtt_CHECK_NOTHROW(auto _t = A_cref * B_ref.inverse()); // this should use the
		                                                       // operator*(any_quantity_cref&,any_quantity&&);
		//+ and * are completly equivalent.
		qtt_CHECK_NOTHROW(auto _t = A + B);
		qtt_CHECK_NOTHROW(auto _t = A_ref + B);
		qtt_CHECK_NOTHROW(auto _t = A_cref + B);
		qtt_CHECK_NOTHROW(auto _t = A + B_ref);
		qtt_CHECK_NOTHROW(auto _t = A_ref + B_ref);
		qtt_CHECK_NOTHROW(auto _t = A_cref + B_ref);
		qtt_CHECK_NOTHROW(auto _t = A + B_cref);
		qtt_CHECK_NOTHROW(auto _t = A_ref + B_cref);
		qtt_CHECK_NOTHROW(auto _t = A_cref + B_cref);
		qtt_CHECK_NOTHROW(auto _t = A + B_cref.inverse());     // this should use the
		                                                       // operator+(any_quantity_cref&,any_quantity&&);
		qtt_CHECK_NOTHROW(auto _t = A_ref + B.inverse());      // this should use the
		                                                       // operator+(any_quantity_cref&,any_quantity&&);
		qtt_CHECK_NOTHROW(auto _t = A_cref + B_ref.inverse()); // this should use the
		                                                       // operator+(any_quantity_cref&,any_quantity&&);
	}
	auto C = any_quantity(B_cref);
	// C == A_ref *B_cref;
	// C == B_cref *A_cref.inverse();
	// C == A_ref *B_cref *A_cref.inverse();		   //A.commute(B) should be
	// an optimisation of the formula on the right.
	qtt_CHECK(C == A_ref * B_cref * A_cref.inverse()); // A.commute(B) should be an optimisation
	                                                   // of the formula on the right.
	C = A;
	qtt_CHECK(C == B.inverse() * A * B); // A.commute_(B) should be an optimization
	                                     // of the formula on the right.
}

} // namespace quantt

#include "quantity_vector.h"

template <>
struct fmt::formatter<quantt::any_quantity_cref> : fmt::formatter<quantt::vquantity>
{
	template <class FormatContext>
	auto format(const quantt::any_quantity_cref &qt, FormatContext &ctx)
	{
		return fmt::formatter<quantt::vquantity>::format(
		    qt.get(),
		    ctx); // right now qt.format_to is only define for fmt::format_context. Should work for any output stream.
	}
};
template <>
struct fmt::formatter<quantt::any_quantity_ref> : public fmt::formatter<quantt::any_quantity_cref>
{
};
template <>
struct fmt::formatter<quantt::any_quantity> : public fmt::formatter<quantt::any_quantity_cref>
{
};

#endif /* D56E4C12_98E1_4C9E_B0C4_5B35A5A3CD17 */
