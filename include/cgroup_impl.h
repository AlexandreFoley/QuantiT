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
#include <vector>

namespace quantt
{

/**
 * @brief forward declaration of the iterator type used for type erased container
 * 
 */
struct cgroup_iterator;
struct const_cgroup_iterator;
class cgroup_vector_impl;

/**
 * @brief interface type for the implementation of a cgroup
 * cgroup stands for "composite group" and is the tensor product of multiple simple groups.
 * 
 * The function that take a cgroup_impl as an argument are not expected to work
 * if the actual type of the argument isn't the same derived type as this.
 */
class cgroup_impl
{
	friend struct cgroup_iterator;
	friend struct const_cgroup_iterator;
	/**
	 * @brief compute the distance between two pointer to a derived type.
	 *        for use by the iterators cgroup_iterator and const_cgroup_iterator.
	 * 
	 * @param rhs 
	 * @return std::ptrdiff_t 
	 */
	virtual std::ptrdiff_t ptr_diff(const cgroup_impl* const rhs) const = 0;
	virtual cgroup_impl* ptr_add(std::ptrdiff_t rhs) const = 0;

public:
	/**
	* @brief in place implementation of the group operation.
	*        *this = (*this)*other, where "other" is the argument givent
	* @return cgroup_impl& : a reference to the current object.
	*/
	virtual cgroup_impl&
	op(const cgroup_impl&) = 0;
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

	virtual std::unique_ptr<cgroup_impl> clone() const = 0;
	virtual std::unique_ptr<cgroup_vector_impl> make_vector(size_t cnt) const = 0;

	/**
	 * @brief create the neutral element of whatever underlying type
	 * 
	 * @return std::unique_ptr<cgroup_impl> : the neutral element
	 */
	virtual std::unique_ptr<cgroup_impl> neutral() const = 0;

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
public:
	// has default constructor and assigment operator as well.
	conc_cgroup_impl(Groups... grp) : val(grp...) {}
	conc_cgroup_impl() = default;
	conc_cgroup_impl(const conc_cgroup_impl&) = default;
	conc_cgroup_impl(conc_cgroup_impl&&) = default;
	~conc_cgroup_impl() override {};
	conc_cgroup_impl& operator=(conc_cgroup_impl other) noexcept;

	std::unique_ptr<cgroup_impl> clone() const override;
	std::unique_ptr<cgroup_impl> neutral()const override;

	    conc_cgroup_impl&
	    op(const conc_cgroup_impl& other);
	cgroup_impl& op(const cgroup_impl& other) override;

	void op_to(conc_cgroup_impl& other) const;
	void op_to(cgroup_impl& other) const override;
	conc_cgroup_impl& inverse_() override;
	cgroup_impl& operator=(const cgroup_impl& other) override;
	bool operator==(const conc_cgroup_impl& other) const;
	bool operator==(const cgroup_impl& other) const override;
	bool operator!=(const conc_cgroup_impl& other) const;
	bool operator!=(const cgroup_impl& other) const override;
	void swap(conc_cgroup_impl& other);
	void swap(cgroup_impl& other) override;

	std::unique_ptr<cgroup_vector_impl> make_vector(size_t cnt) const override;

private:
	/**
	 * @brief implementation of the polymophic pointer difference
	 * 
	 * this is the way to implement this, basically no matter what.
	 * 
	 */
	std::tuple<Groups...> val;
	std::ptrdiff_t ptr_diff(const cgroup_impl* const rhs) const override
	{
		return this - static_cast<const conc_cgroup_impl* const>(rhs);
	}

	/**
	 * @brief implementation of the polymophic pointer addition-assignment
	 * 
	 * this is the way to implement this, basically no matter what.
	 * 
	 */
	conc_cgroup_impl* ptr_add(std::ptrdiff_t rhs) const override
	{
		return const_cast<conc_cgroup_impl*>(this) + rhs;
	}
};

template<class... T>
conc_cgroup_impl<T...>& conc_cgroup_impl<T...>::operator=(conc_cgroup_impl other) noexcept
{
	using std::swap;
	swap(val,other.val);
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

template<class... T>
std::unique_ptr<cgroup_impl> conc_cgroup_impl<T...>::neutral() const
{
	return std::make_unique<conc_cgroup_impl<T...> >();
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

template <class>
struct is_conc_cgroup_impl : public std::false_type
{
};
template <class... S>
struct is_conc_cgroup_impl<conc_cgroup_impl<S...>> : public groups::all_group<S...>
{
};

} // namespace quantt

#endif /* F2547C1C_9177_4373_9C66_8D4C8621C7CC */
