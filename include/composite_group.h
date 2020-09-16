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

// TODO: replace the dynamic_cast<(const) conc_cgroup_impl&> with a custom
// function that allow for
//      a more useful customized message when a bad cast happen.

#ifndef D56E4C12_98E1_4C9E_B0C4_5B35A5A3CD17
#define D56E4C12_98E1_4C9E_B0C4_5B35A5A3CD17

#include "cgroup_impl.h"
#include "groups_utils.h"
#include "templateMeta.h"
#include <cstdint>
#include <fmt/core.h>
#include <ostream>
#include <type_traits>
#include <utility>

namespace quantt
{

class cgroup_ref;
class cgroup_cref;
class cgroup;
/**
 * @brief wrapper for the polymorphic composite groups from cgroup_impl.h that provide value semantics
 * We expect that for a given problem, the type of the underlying cgoup_impl used will be uniform.
 * 
 */
class cgroup final
{
	std::unique_ptr<cgroup_impl> impl;

	cgroup(const cgroup_impl& other) : impl(impl->clone()) {}
	friend cgroup_ref;
	friend cgroup_cref;

public:
	cgroup(std::unique_ptr<cgroup_impl>&& _impl) : impl(std::move(_impl)) {}

	/**
	 * @brief Construct a new cgroup object. It compose the group element it receive.
	 * 
	 * @tparam Groups types of the different group we're composing
	 * @tparam std::enable_if_t<groups::all_group_v<Groups...>> 
	 *     verify that the arguements all satisfy the required group interface.
	 * @param grps values used to initialized the group
	 */
	template <class... Groups, class = std::enable_if_t<groups::all_group_v<Groups...>>>
	cgroup(Groups... grps) : impl(std::make_unique<conc_cgroup_impl<Groups...>>(grps...)) {}

	cgroup() : impl(std::make_unique<conc_cgroup_impl<groups::C<1>>>()) {} //default to a group that contain only the neutral element. avoid having an unitialized unique_ptr.
	cgroup(const cgroup& other) : impl(other.impl->clone()) {}
	explicit cgroup(cgroup_cref other); //explicit to avoid accidental copies and ambiguous overloads
	explicit cgroup(cgroup_ref other);  //explicit to avoid accidental copies and ambiguous overloads
	cgroup(cgroup&&) = default;

	/**
	 * @brief swap function with reference to cgroup. Always work.
	 * 
	 * @param other 
	 */
	void swap(cgroup& other) noexcept;

	void swap(cgroup_ref other); //only work if the underlying types ARE the same.

	/**
	 * @brief generate the neutral element of the underlying gorup.
	 * 
	 * @return cgroup 
	 */
	cgroup neutral() const;
	/**
	 * @brief assigment operators, copy the value into this.
	 * These assigment operator can change the underlying type of *this.
	 * 
	 * @param other: a cgroup, cgroup_ref or cgroup_cref
	 * @return cgroup& :reference to *this
	 */
	cgroup& operator=(cgroup_cref other);
	cgroup& operator=(cgroup_ref other);
	cgroup& operator=(const cgroup& other);
	cgroup& operator=(cgroup&& other);
	/**
	 * @brief implicit conversion to the reference to const type
	 * 
	 * @return cgroup_cref: a reference to this constant object
	 */
	operator cgroup_cref() const;
	/**
	 * @brief implicit conversion to the reference type
	 * 
	 * @return cgroup_ref : a reference to this object
	 */
	operator cgroup_ref();

	~cgroup() {}
	/**
	 * @brief In place group operation
	 * 
	 * @param other : constant reference to another composite group of the same type.
	 *                work with cgroup, cgroup_ref and cgroup_cref for input.
	 * @return cgroup& reference to *this
	 */
	cgroup& operator*=(cgroup_cref other);
	cgroup& operator+=(cgroup_cref other);
	/**
	* @brief out of place group operation, on two composite group element of the same type
	* 
	* @param lhs: can be a cgroup,cgroup_ref or cgroup_cref
	* @param rhs: can be a cgroup,cgroup_ref or cgroup_cref
	* @return cgroup a new cgroup object.
	*/
	friend cgroup operator*(cgroup_cref lhs, cgroup_cref rhs);
	friend cgroup operator*(cgroup_cref lhs, cgroup&& rhs);
	friend cgroup operator+(cgroup_cref lhs, cgroup_cref rhs);
	friend cgroup operator+(cgroup_cref lhs, cgroup&& rhs);

	/**
	 * @brief in place inverse
	 * 
	 * @return cgroup& 
	 */
	cgroup& inverse_();
	/**
	 * @brief return a new cgroup object that is the inverse of this.
	 * 
	 * @return cgroup 
	 */
	cgroup inverse() const;

	/**
	 * @brief compute z such that (*this)*other = z*(*this)
	 * store the resulting value in the other.
	 * 
	 * @param other works with cgroup_ref or cgroup.
	 */
	void commute(cgroup_ref other) const;
	void commute_(cgroup_cref other);

	/**
	 * @brief equality comparison operator
	 * 
	 * @param lhs : works with cgroup,cgroup_ref and cgroup_cref
	 * @param rhs : works with cgroup,cgroup_ref and cgroup_cref
	 * @return true : same group element
	 * @return false : different group element
	 */
	friend bool operator==(cgroup_cref lhs, cgroup_cref rhs);
	/**
	 * @brief different comparison operator
	 * 
	 * @param lhs : works with cgroup,cgroup_ref and cgroup_cref
	 * @param rhs : works with cgroup,cgroup_ref and cgroup_cref
	 * @return true : different group element
	 * @return false : same group element
	 */
	friend bool operator!=(cgroup_cref lhs, cgroup_cref rhs);
};
/**
 * @brief Constant reference type for cgroup. 
 * 
 * Implements only the const method of cgroup. consult cgroup for methods documentation
 * 
 * Made to facilitate element by element manipulation within a specialize 
 * container for cgroup_impl.
 */
class cgroup_cref
{
	const cgroup_impl* const ref;
	friend cgroup_ref;
	friend cgroup;

public:
	cgroup_cref() = delete;
	cgroup_cref(cgroup_impl& other) : ref(&other) {}
	cgroup_cref(const cgroup_cref& other) : ref(other.ref) {}

	cgroup neutral() const;
	const cgroup_impl& get() const;
	cgroup inverse() const;

	void commute(cgroup_ref other) const;
};
/**
 * @brief Reference type for cgroup. 
 * 
 * Implements all the method of cgroup. consult cgroup for methods documentation
 * 
 * Made to facilitate element by element manipulation within a specialize 
 * container for cgroup_impl.
 */
class cgroup_ref final
{
	cgroup_impl* const ref;
	friend cgroup_cref;
	friend cgroup;

public:
	cgroup_ref() = delete;
	cgroup_ref(const cgroup_ref&) = delete; // for clarity: you can't do that one. would cast away the const
	                                        // character.
	cgroup_ref(cgroup_ref& other) : ref(other.ref) {}
	cgroup_ref(cgroup_impl& other) : ref(&other) {}
	operator cgroup_cref() const;
	cgroup_ref& operator=(cgroup_cref other);

	cgroup neutral() const;
	const cgroup_impl& get() const;
	cgroup_impl& get();

	cgroup_ref& operator*=(cgroup_cref other);
	cgroup_ref& operator+=(cgroup_cref other);
	cgroup_ref& inverse_();
	cgroup inverse() const;
	void commute(cgroup_ref other) const;
	void commute_(cgroup_cref other);
	void swap(cgroup_ref other);
};

//method definitions in no particular order...
const cgroup_impl& cgroup_cref::get() const
{
	return *ref;
}
cgroup cgroup_cref::inverse() const
{
	return cgroup(*this).inverse_();
}
cgroup_ref& cgroup_ref::inverse_()
{
	get().inverse_();
	return *this;
}
cgroup cgroup_ref::inverse() const
{
	return cgroup(ref->clone()).inverse_();
}
cgroup cgroup::neutral() const
{
	return cgroup(impl->neutral());
}
cgroup cgroup_ref::neutral() const
{
	return cgroup(get().neutral());
}
cgroup cgroup_cref::neutral() const
{
	return cgroup(get().neutral());
}

void cgroup_ref::swap(cgroup_ref other)
{
	ref->swap(other.get());
}
cgroup::cgroup(cgroup_cref other) : impl(other.get().clone()) {}
cgroup::cgroup(cgroup_ref other) : impl(other.get().clone()) {}

void cgroup::swap(cgroup& other) noexcept
{
	using std::swap;
	swap(other.impl, impl);
}
cgroup& cgroup::operator=(cgroup&& other) // will work even when the underlying type isn't the same...
{
	swap(other);
	return *this;
}
cgroup& cgroup::inverse_()
{
	impl->inverse_();
	return *this;
}
cgroup cgroup::inverse() const
{
	return cgroup(*this).inverse_();
}
cgroup& cgroup::operator=(const cgroup& other)
{
	*impl = *other.impl;
	return *this;
}
void cgroup::swap(cgroup_ref other) { impl->swap(other.get()); }
void swap(cgroup& lhs, cgroup& rhs) { lhs.swap(rhs); }
void swap(cgroup_ref lhs, cgroup_ref rhs) { lhs.swap(rhs); }
cgroup::operator cgroup_cref() const { return cgroup_cref(*impl.get()); }
cgroup::operator cgroup_ref() { return cgroup_ref(*impl.get()); }
cgroup& cgroup::operator*=(cgroup_cref other)
{
	impl->op(other.get());
	return *this;
}
cgroup operator*(cgroup_cref lhs, cgroup_cref rhs)
{
	return cgroup(lhs) *= rhs;
}
cgroup operator*(cgroup_cref lhs, cgroup&& rhs)
{
	lhs.get().op_to(*rhs.impl);
	return rhs;
}
cgroup operator+(cgroup_cref lhs, cgroup_cref rhs)
{
	return lhs * rhs;
}
cgroup operator+(cgroup_cref lhs, cgroup&& rhs)
{
	return lhs * rhs;
}
cgroup& cgroup::operator=(cgroup_cref other)
{
	impl = other.get().clone();
	return *this;
}
cgroup& cgroup::operator=(cgroup_ref other)
{
	return operator=(cgroup_cref(other));
}
cgroup& cgroup::operator+=(cgroup_cref other)
{
	return (*this) *= other;
}

void cgroup::commute_(cgroup_cref other)
{
	impl->commute_(other.get());
}
void cgroup::commute(cgroup_ref other) const
{
	impl->commute(other.get());
}
bool operator!=(cgroup_cref left, cgroup_cref right)
{
	return left.get() != (right.get());
}
bool operator==(const cgroup_cref left, const cgroup_cref right)
{
	return left.get() == (right.get());
}

const cgroup_impl& cgroup_ref::get() const
{
	return *ref;
}
cgroup_impl& cgroup_ref::get()
{
	return *ref;
}
cgroup_ref::operator cgroup_cref() const
{
	return cgroup_cref(*ref);
}
cgroup_ref& cgroup_ref::operator=(cgroup_cref other)
{
	get() = other.get();
	return *this;
}
cgroup_ref& cgroup_ref::operator*=(const cgroup_cref other)
{
	get().op(other.get());
	return *this;
}

cgroup_ref& cgroup_ref::operator+=(const cgroup_cref other)
{
	return (*this) *= other;
}
void cgroup_ref::commute(cgroup_ref other) const
{
	get().commute(other.get());
}
void cgroup_ref::commute_(cgroup_cref other)
{
	get().commute_(other.get());
}

void cgroup_cref::commute(cgroup_ref other) const
{
	get().commute(other.get());
}

} // namespace quantt

#endif /* D56E4C12_98E1_4C9E_B0C4_5B35A5A3CD17 */
