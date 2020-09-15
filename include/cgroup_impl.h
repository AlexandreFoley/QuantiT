/*
 * File: cgroup_impl.h
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

#include "groups_utils.h"
#include "templateMeta.h"
#include <memory>
#include <type_traits>

namespace quantt
{

/**
 * @brief interface type for the implementation of a cgroup
 * cgroup stands for "composite group" and is the tensor product of multiple simple groups.
 * 
 * The function that take a cgroup_impl as an argument are not expected to work
 * if the actual type of the argument isn't the same derived type as this.
 */
class cgroup_impl
{
public:
	/**
 * @brief in place implementation of the group operation.
 * 	*this = (*this)*other, where "other" is the argument givent
 * @return cgroup_impl& : a reference to the current object.
 */
	virtual cgroup_impl& op(const cgroup_impl&) = 0;
	/**
	 * @brief in place implementation of the group operation, the result is 
	 * stored in the given argument.
	 * 
	 */
	virtual void op_to(cgroup_impl&) const = 0;
	/**
	 * @brief in place computation of the inverse of *this.
	 * 
	 * @return cgroup_impl& : reference to the current object.
	 */
	virtual cgroup_impl& inverse_() = 0;
	/**
	 * @brief compute z such that (*this)*other = z*(*this)
	 * the result z is stored in the method's arguement.
	 */
	virtual void commute(cgroup_impl& other) const = 0;
	/**
	 * @brief compute z such that (*this)*other = other*z where other.
	 * z is stored in *this.
	 */
	virtual void commute_(const cgroup_impl& other) = 0;
	/**
	 * @brief create clone the object.
	 * 
	 * @return std::unique_ptr<cgroup_impl> : the clone
	 */
	virtual std::unique_ptr<cgroup_impl> clone() const = 0;

	virtual cgroup_impl& operator=(const cgroup_impl&) = 0;
	virtual bool operator==(const cgroup_impl&) const = 0;
	virtual bool operator!=(const cgroup_impl&) const = 0;
	virtual void swap(cgroup_impl&) = 0;
	virtual ~cgroup_impl() {}
};

/**
 * @brief template implementation of the concrete composite group types.
 * This template of class is used by the type cgroup, cgroup_ref and cgroup_cref
 * defined in composite_group.h
 * @tparam Groups 
 */
template <class... Groups>
class conc_cgroup_impl final : public cgroup_impl
{
	std::tuple<Groups...> val;

public:
	// has default constructor and assigment operator as well.
	conc_cgroup_impl(Groups... grp) : val(grp...) {}
	~conc_cgroup_impl() override = default;
	conc_cgroup_impl& operator=(const conc_cgroup_impl& other);

	std::unique_ptr<cgroup_impl> clone() const override;

	conc_cgroup_impl& op(const conc_cgroup_impl& other);
	cgroup_impl& op(const cgroup_impl& other) override;

	void op_to(conc_cgroup_impl& other) const;
	void op_to(cgroup_impl& other) const override;
	conc_cgroup_impl& inverse_() override;
	void commute(conc_cgroup_impl& other) const;
	void commute(cgroup_impl& other) const override;
	void commute_(const conc_cgroup_impl& other);
	void commute_(const cgroup_impl& other) override;
	cgroup_impl& operator=(const cgroup_impl& other) override;
	bool operator==(const conc_cgroup_impl& other) const;
	bool operator==(const cgroup_impl& other) const override;
	bool operator!=(const conc_cgroup_impl& other) const;
	bool operator!=(const cgroup_impl& other) const override;
	void swap(conc_cgroup_impl& other);
	void swap(cgroup_impl& other) override;
};
template <class... T>
conc_cgroup_impl<T...>& conc_cgroup_impl<T...>::operator=(const conc_cgroup_impl<T...>& other)
{
	val = other.val;
	return *this;
}
template <class... T>
conc_cgroup_impl<T...>& conc_cgroup_impl<T...>::op(const conc_cgroup_impl<T...>& other)
{
	for_each2(val, other.val, [](auto&& vl, auto&& ovl) {
		vl.op(ovl);
	});
	return *this;
}
template <class... T>
cgroup_impl& conc_cgroup_impl<T...>::op(const cgroup_impl& other)
{
	return op(dynamic_cast<const conc_cgroup_impl&>(other));
}
template <class... T>
std::unique_ptr<cgroup_impl> conc_cgroup_impl<T...>::clone() const
{
	return std::make_unique<conc_cgroup_impl<T...>>(*this);
}
template <class... T>
void conc_cgroup_impl<T...>::op_to(conc_cgroup_impl<T...>& other) const
{
	for_each2(val, other.val, [](auto&& vl, auto&& ovl) {
		ovl = groups::op(vl, ovl);
	});
}
template <class... T>
void conc_cgroup_impl<T...>::op_to(cgroup_impl& other) const
{
	return op_to(dynamic_cast<conc_cgroup_impl<T...>&>(other));
}
template <class... T>
conc_cgroup_impl<T...>& conc_cgroup_impl<T...>::inverse_()
{
	for_each(val, [](auto&& vl) {
		vl.inverse_();
	});
	return *this;
}
template <class... T>
void conc_cgroup_impl<T...>::commute(conc_cgroup_impl<T...>& other) const
{
	for_each2(val, other.val, [](auto&& vl, auto&& ovl) {
		vl.commute(ovl);
	});
}
template <class... T>
void conc_cgroup_impl<T...>::commute(cgroup_impl& other) const
{
	commute(dynamic_cast<conc_cgroup_impl&>(other));
}
template <class... T>
void conc_cgroup_impl<T...>::commute_(const conc_cgroup_impl<T...>& other)
{
	for_each2(val, other.val, [](auto&& vl, auto&& ovl) {
		vl.commute_(ovl);
	});
}
template <class... T>
void conc_cgroup_impl<T...>::commute_(const cgroup_impl& other)
{
	commute_(dynamic_cast<const conc_cgroup_impl&>(other));
}
template <class... T>
cgroup_impl& conc_cgroup_impl<T...>::operator=(const cgroup_impl& other)
{
	return (*this) = dynamic_cast<const conc_cgroup_impl<T...>&>(other);
}
template <class... T>
bool conc_cgroup_impl<T...>::operator==(const conc_cgroup_impl<T...>& other) const
{
	return val == other.val;
}
template <class... T>
bool conc_cgroup_impl<T...>::operator==(const cgroup_impl& other) const
{
	return operator==(dynamic_cast<const conc_cgroup_impl<T...>&>(other));
}
template <class... T>
bool conc_cgroup_impl<T...>::operator!=(const conc_cgroup_impl<T...>& other) const
{
	return val != other.val;
}
template <class... T>
bool conc_cgroup_impl<T...>::operator!=(const cgroup_impl& other) const
{
	return operator!=(dynamic_cast<const conc_cgroup_impl<T...>&>(other));
}
template <class... T>
void conc_cgroup_impl<T...>::swap(conc_cgroup_impl<T...>& other)
{
	using std::swap;
	swap(val, other.val);
}
template <class... T>
void conc_cgroup_impl<T...>::swap(cgroup_impl& other)
{
	swap(dynamic_cast<conc_cgroup_impl<T...>&>(other));
}
} // namespace quantt

#endif /* F2547C1C_9177_4373_9C66_8D4C8621C7CC */
