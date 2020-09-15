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
class cgroup final
{
	std::unique_ptr<cgroup_impl> impl;

	cgroup(const cgroup_impl& other) : impl(impl->clone()) {}
	friend cgroup_ref;
	friend cgroup_cref;

public:
	cgroup(std::unique_ptr<cgroup_impl>&& _impl) : impl(std::move(_impl)) {}

	template <class... Groups, class = std::enable_if_t<groups::all_group_v<Groups...>>>
	cgroup(Groups... grps) : impl(std::make_unique<conc_cgroup_impl<Groups...>>(grps...)) {}

	cgroup() = default;
	cgroup(const cgroup& other) : impl(other.impl->clone()) {}
	cgroup(cgroup_cref other);
	explicit cgroup(cgroup_ref other);
	cgroup(cgroup&&) = default;
	void swap(cgroup_ref other);
	void swap(cgroup& other)
	{
		using std::swap;
		swap(other.impl, impl);
	}
	cgroup& operator=(cgroup&& other) // will work even when the underlying type
	                                  // isn't the same...
	{
		swap(other);
		return *this;
	}
	cgroup& operator=(cgroup_cref other);
	cgroup& operator=(cgroup_ref other);
	cgroup& operator=(const cgroup& other);
	operator cgroup_cref() const;
	operator cgroup_ref();

	~cgroup() {}

	cgroup& operator*=(cgroup_cref other);
	friend cgroup operator*(cgroup_cref lhs, cgroup&& rhs);

	cgroup& operator+=(cgroup_cref other);

	cgroup& inverse_()
	{
		impl->inverse_();
		return *this;
	}
	cgroup inverse() const { return cgroup(*this).inverse_(); }
	void commute(cgroup_ref other) const;
	void commute_(cgroup_cref other);
};

// class to allow easy interopt between cgroup_array and cgroup
class cgroup_cref
{
	const cgroup_impl* const ref;
	friend cgroup_ref;
	friend cgroup;

public:
	cgroup_cref() = delete;
	cgroup_cref(cgroup_impl& other) : ref(&other) {}
	cgroup_cref(const cgroup_cref& other) : ref(other.ref) {}

	const cgroup_impl& get() const { return *ref; }
	cgroup inverse() const { return cgroup(*this).inverse_(); }

	void commute(cgroup_ref other) const;
};
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

	const cgroup_impl& get() const;
	cgroup_impl& get();

	cgroup_ref& operator*=(cgroup_cref other);
	cgroup_ref& operator+=(cgroup_cref other);
	cgroup_ref& inverse_()
	{
		get().inverse_();
		return *this;
	}
	cgroup inverse() const { return cgroup(ref->clone()).inverse_(); }
	void commute(cgroup_ref other) const;
	void commute_(cgroup_cref other);
	void swap(cgroup_ref other) { ref->swap(other.get()); }
};

cgroup::cgroup(cgroup_cref other) : impl(other.get().clone()) {}
cgroup::cgroup(cgroup_ref other) : impl(other.get().clone()) {}
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
cgroup operator*(cgroup_cref lhs, cgroup_cref rhs) { return cgroup(lhs) *= rhs; }
cgroup operator*(cgroup_cref lhs, cgroup&& rhs)
{
	lhs.get().op_to(*rhs.impl);
	return rhs;
}
cgroup& cgroup::operator=(cgroup_cref other)
{
	(*impl) = (other.get());
	return *this;
}
cgroup& cgroup::operator=(cgroup_ref other) { return operator=(cgroup_cref(other)); }
cgroup& cgroup::operator+=(cgroup_cref other) { return (*this) *= other; }

cgroup operator+(cgroup_cref lhs, cgroup_cref rhs) { return cgroup(lhs) += rhs; }
void cgroup::commute_(cgroup_cref other) { impl->commute_(other.get()); }
void cgroup::commute(cgroup_ref other) const { impl->commute(other.get()); }
bool operator!=(cgroup_cref left, cgroup_cref right) { return left.get() != (right.get()); }
bool operator==(const cgroup_cref left, const cgroup_cref right) { return left.get() == (right.get()); }

const cgroup_impl& cgroup_ref::get() const { return *ref; }
cgroup_impl& cgroup_ref::get() { return *ref; }
cgroup_ref::operator cgroup_cref() const { return cgroup_cref(*ref); }
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

cgroup_ref& cgroup_ref::operator+=(const cgroup_cref other) { return (*this) *= other; }
void cgroup_ref::commute(cgroup_ref other) const { get().commute(other.get()); }
void cgroup_ref::commute_(cgroup_cref other) { get().commute_(other.get()); }

void cgroup_cref::commute(cgroup_ref other) const { get().commute(other.get()); }

} // namespace quantt

#endif /* D56E4C12_98E1_4C9E_B0C4_5B35A5A3CD17 */
