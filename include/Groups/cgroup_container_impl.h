/*
 * File: cgroup_container_impl.h
 * Project: quantt
 * File Created: Monday, 28th September 2020 1:46:36 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Monday, 28th September 2020 1:46:36 pm
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#ifndef BBF1F73E_87CC_4C69_9CCC_5D2526535A4F
#define BBF1F73E_87CC_4C69_9CCC_5D2526535A4F

#include "boost/stl_interfaces/iterator_interface.hpp"
#include "composite_group.h"

namespace quantt
{

class blocklist
{
};
class Qtensor
{
};

struct virt_ptr_aritmetic
{
	virtual std::ptrdiff_t ptr_diff(const cgroup_impl* lhs, const cgroup_impl* rhs) const = 0;
	virtual cgroup_impl* ptr_add(cgroup_impl* ptr, std::ptrdiff_t n) const = 0;
	virtual const cgroup_impl* ptr_add(const cgroup_impl* ptr, std::ptrdiff_t n) const = 0;
};

struct const_cgroup_iterator : public boost::stl_interfaces::iterator_interface<const_cgroup_iterator, std::random_access_iterator_tag, cgroup, cgroup_cref, const cgroup_impl*>
{
private:
	const cgroup_impl* it;
	const virt_ptr_aritmetic* ar;
	friend boost::stl_interfaces::access;

	const cgroup_impl*& base_reference() noexcept { return it; }
	const cgroup_impl* base_reference() const noexcept { return it; }

public:
	constexpr const_cgroup_iterator() : it(nullptr), ar(nullptr) {}
	constexpr const_cgroup_iterator(const cgroup_impl* _it, const virt_ptr_aritmetic* _ar) : it(_it), ar(_ar) {}
	using base_type = boost::stl_interfaces::iterator_interface<const_cgroup_iterator, std::random_access_iterator_tag, cgroup, cgroup_cref, const cgroup_impl*>;
	cgroup_cref operator*()
	{
		return cgroup_cref(it);
	}
	base_type::difference_type operator-(const_cgroup_iterator rhs)
	{
		return ar->ptr_diff(it, rhs.it);
	}
	const_cgroup_iterator& operator+=(base_type::difference_type n)
	{
		it = ar->ptr_add(it, n);
		return *this;
	}
	const cgroup_impl* base() const
	{
		return it;
	}
	const virt_ptr_aritmetic* vt() const
	{
		return ar;
	}
};
struct cgroup_iterator : public boost::stl_interfaces::iterator_interface<cgroup_iterator, std::random_access_iterator_tag, cgroup, cgroup_ref, cgroup_impl*>
{
private:
	cgroup_impl* it;
	const virt_ptr_aritmetic* ar;
	friend boost::stl_interfaces::access;

	cgroup_impl*& base_reference() noexcept { return it; }
	cgroup_impl* base_reference() const noexcept { return it; }

public:
	constexpr cgroup_iterator() : it(nullptr), ar(nullptr) {}
	constexpr cgroup_iterator(cgroup_impl* _it, const virt_ptr_aritmetic* _ar) : it(_it), ar(_ar) {}
	using base_type = boost::stl_interfaces::iterator_interface<cgroup_iterator, std::random_access_iterator_tag, cgroup, cgroup_ref, cgroup_impl*>;
	cgroup_ref operator*() const
	{
		return cgroup_ref(it);
	}
	base_type::difference_type operator-(cgroup_iterator rhs)
	{
		return ar->ptr_diff(it, rhs.it);
	}
	cgroup_iterator& operator+=(base_type::difference_type n)
	{
		it = ar->ptr_add(it, n);
		return *this;
	}
	cgroup_impl* base()
	{
		return it;
	}
	operator const_cgroup_iterator()
	{
		return const_cgroup_iterator(it, ar);
	}
	const virt_ptr_aritmetic* vt()
	{
		return ar;
	}
};
//so that different vector type have a different base type
/**
 * @brief polymorphic (type-erased?) container of cgroup.
 * All the element in the container are of the same concrete type, but the concrete type doesn't change the type of the container.
 * Implement most of the interface of std::vector (some parts are templates that
 * can't be replaced with run time polymorphism). 
 * 
 * While iterating on the polymorphic container is possible, and useful for prototyping,
 * doing so will incur a significant cost due to virtual calls.
 * **For perfomance critical operations on the whole collection, it is necessary to
 * implement the loop as polymorphic function that will resolve in O(1) polymorphic call/dynamic_cast.**
 * 
 * The visitor pattern could also be used to accomplish efficient iteration on the vector, but a visitor cannot be 
 * made to work with all the types in an open set of types.
 * 
 * Note: I can't help but feel there's one too many inderection level necessary here (pointer to conc_vect.data, data is a pointer to the actual data)
 *       There's almost certainly a design without this extra indirection and the same interface.
 */
class cgroup_vector_impl
{
public:
	using iterator = cgroup_iterator;
	using const_iterator = const_cgroup_iterator;
	using reverse_iterator = std::reverse_iterator<iterator>;
	using const_reverse_iterator = std::reverse_iterator<const_iterator>;

	virtual blocklist identify_blocks(const Qtensor&) const = 0;
	virtual std::unique_ptr<cgroup_vector_impl> clone() const = 0;
	/**
	 * @brief accessors.
	 * 	can be implemented as virtual, by exploiting the covariant return type.
	 * 
	 * @return cgroup_impl& 
	 */
	virtual cgroup_impl& operator[](size_t) = 0;
	virtual const cgroup_impl& operator[](size_t) const = 0;
	virtual cgroup_impl& at(size_t) = 0;
	virtual const cgroup_impl& at(size_t) const = 0;

	virtual cgroup_impl& front() = 0;
	virtual const cgroup_impl& front() const = 0;
	virtual cgroup_impl& back() = 0;
	virtual const cgroup_impl& back() const = 0;
	virtual cgroup_impl* data() = 0;
	virtual const cgroup_impl* data() const = 0;
	// capacity
	[[nodiscard]] virtual bool empty() const = 0;
	[[nodiscard]] virtual size_t size() const = 0;
	[[nodiscard]] virtual size_t max_size() const = 0;
	virtual void reserve(size_t) = 0;
	[[nodiscard]] virtual size_t capacity() const = 0;
	virtual void shrink_to_fit() = 0;
	//modifiers
	virtual void clear() = 0;
	virtual iterator insert(const_iterator pos, const cgroup_impl& Val) = 0;
	virtual iterator insert(const_iterator pos, size_t count, const cgroup_impl& Val) = 0;
	virtual iterator insert(const_iterator pos, const_iterator first, const_iterator last) = 0;
	virtual iterator insert(const_iterator pos, const_reverse_iterator first, const_reverse_iterator last) = 0;
	virtual iterator erase(const_iterator pos) = 0;
	virtual iterator erase(const_iterator first, const_iterator last) = 0;
	virtual void push_back(const cgroup_impl& value) = 0;
	virtual void pop_back() = 0;
	virtual void resize(size_t count) = 0;
	virtual void resize(size_t count, const cgroup_impl& val) = 0;
	virtual void swap(cgroup_vector_impl& other) = 0;

private:
	/**
	 * @brief virtual implementation of the begin and end functions
	 * 
	 * necessary to use a different function name to allow override by child class
	 * while allowing those child class to have optimized begin and end functions 
	 * (that return an incompatible type.).
	 * 
	 * @return iterator 
	 */
	virtual iterator begin_impl() = 0;
	virtual iterator end_impl() = 0;
	virtual const_iterator cbegin_impl() const = 0;
	virtual const_iterator cend_impl() const = 0;

public:
	iterator begin()
	{
		return begin_impl();
	}
	iterator end()
	{
		return end_impl();
	}
	const_iterator cbegin() const
	{
		return cbegin_impl();
	}
	const_iterator cend() const
	{
		return cend_impl();
	}

	reverse_iterator rbegin()
	{
		return reverse_iterator(end());
	}
	reverse_iterator rend()
	{
		return reverse_iterator(begin());
	}
	const_iterator begin() const
	{
		return cbegin();
	};
	const_iterator end() const
	{
		return cend();
	};
	const_reverse_iterator crbegin() const
	{
		return const_reverse_iterator(cend());
	}
	const_reverse_iterator crend() const
	{
		return const_reverse_iterator(cbegin());
	}
	const_reverse_iterator rbegin() const
	{
		return (crbegin());
	}
	const_reverse_iterator rend() const
	{
		return (crend());
	}
};

template <class S, class Allocator = std::allocator<S>, class = std::enable_if_t<std::is_base_of_v<cgroup_impl, S>>>
class conc_cgroup_vector_impl final : public cgroup_vector_impl, public std::vector<S, Allocator>
{
public:
	using iterator = typename std::vector<S>::iterator;
	using const_iterator = typename std::vector<S>::const_iterator;
	using reverse_iterator = typename std::vector<S>::reverse_iterator;
	using const_reverse_iterator = typename std::vector<S>::const_reverse_iterator;
	using value_type = typename std::vector<S>::value_type;
	using allocator_type = typename std::vector<S>::allocator_type;
	using size_type = typename std::vector<S>::size_type;
	using difference_type = typename std::vector<S>::difference_type;
	using reference = typename std::vector<S>::reference;
	using const_reference = typename std::vector<S>::const_reference;
	using pointer = typename std::vector<S>::pointer;
	using const_pointer = typename std::vector<S>::const_pointer;

	using std::vector<S, Allocator>::vector; //so we have all of vector's constructors as well as the rest of its interface.
	conc_cgroup_vector_impl(const std::vector<S, Allocator>& other) : conc_cgroup_vector_impl(static_cast<const conc_cgroup_vector_impl&>(other)) {}
	conc_cgroup_vector_impl(std::vector<S, Allocator>&& other) : conc_cgroup_vector_impl(static_cast<conc_cgroup_vector_impl&&>(other)) {}

	blocklist identify_blocks(const Qtensor&) const override
	{
		//dummy implementation so that the template instantiation is concrete.
		return blocklist();
	}
	//since both parent class define those functions, we have to disambiguate which to use or override.
	S& operator[](size_t n) override
	{
		return std::vector<S>::operator[](n);
	}
	const S& operator[](size_t n) const override
	{
		return std::vector<S>::operator[](n);
	}
	S& at(size_t n) override
	{
		return std::vector<S>::at(n);
	}
	const S& at(size_t n) const override
	{
		return std::vector<S>::at(n);
	}
	// iterators
	using std::vector<S>::begin;   //ok
	using std::vector<S>::end;     //ok
	using std::vector<S>::cend;    //ok
	using std::vector<S>::cbegin;  //ok
	using std::vector<S>::rbegin;  //ok
	using std::vector<S>::rend;    //ok
	using std::vector<S>::crbegin; //ok
	using std::vector<S>::crend;   //ok
	S& front() override
	{
		return std::vector<S>::front(); //covriant -> virtualize
	}
	const S& front() const override
	{
		return std::vector<S>::front(); //covriant -> virtualize
	}
	S& back() override
	{
		return std::vector<S>::back(); //covariant -> virtualize
	}
	const S& back() const override
	{
		return std::vector<S>::back(); //covariant -> virtualize
	}
	S* data() override
	{
		return std::vector<S>::data();
	} //covariant -> virtualize
	const S* data() const override
	{
		return std::vector<S>::data();
	} //covariant -> virtualize
	// capacity
	bool empty() const override
	{
		return std::vector<S>::empty(); //virtualize
	}
	size_t size() const override
	{
		return std::vector<S>::size(); //virtualize
	}
	size_t max_size() const override
	{
		return std::vector<S>::max_size(); //virtualize
	}
	void reserve(size_t n) override
	{
		return std::vector<S>::reserve(n); //virtualize
	}
	size_t capacity() const override
	{
		return std::vector<S>::capacity(); //virtualize
	}
	void shrink_to_fit() override
	{
		return std::vector<S>::shrink_to_fit(); //virtualize
	}                                           //modifiers
	void clear() override
	{
		return std::vector<S>::clear(); //virtualize
	}
	struct ptr_aritmetic_t : virt_ptr_aritmetic
	{
		std::ptrdiff_t ptr_diff(const cgroup_impl* lhs, const cgroup_impl* rhs) const override
		{
			return static_cast<const S*>(lhs) - static_cast<const S*>(rhs);
		}
		const cgroup_impl* ptr_add(const cgroup_impl* ptr, std::ptrdiff_t n) const override
		{
			return static_cast<const S*>(ptr) + n;
		}
		cgroup_impl* ptr_add(cgroup_impl* ptr, std::ptrdiff_t n) const override
		{
			return static_cast<S*>(ptr) + n;
		}
	};
	constexpr static ptr_aritmetic_t ar = ptr_aritmetic_t();

	using std::vector<S>::insert;
	cgroup_iterator insert(const_cgroup_iterator pos, const cgroup_impl& Val) override
	{
		return cgroup_iterator(insert(to_S_iterator(pos), dynamic_cast<const S&>(Val)).base(), &ar);
	}
	cgroup_iterator insert(const_cgroup_iterator pos, size_t count, const cgroup_impl& Val) override
	{
		return cgroup_iterator(insert(to_S_iterator(pos), count, dynamic_cast<const S&>(Val)).base(), &ar);
	}
	cgroup_iterator insert(const_cgroup_iterator pos, const_cgroup_iterator first, const_cgroup_iterator last) override
	{
		return cgroup_iterator(insert(to_S_iterator(pos), to_S_iterator(first), to_S_iterator(last)).base(), &ar);
	}
	cgroup_iterator insert(const_cgroup_iterator pos, cgroup_vector_impl::const_reverse_iterator first, cgroup_vector_impl::const_reverse_iterator last) override
	{
		return cgroup_iterator(insert(to_S_iterator(pos), to_S_iterator(first), to_S_iterator(last)).base(), &ar);
	}
	using std::vector<S>::emplace; // can't virtualize (template)
	using std::vector<S>::erase;   //virtual proxy
	cgroup_iterator erase(const_cgroup_iterator pos) override
	{
		return cgroup_iterator(erase(to_S_iterator(pos)).base(), &ar);
	}
	cgroup_iterator erase(const_cgroup_iterator first, const_cgroup_iterator last) override
	{
		return cgroup_iterator(erase(to_S_iterator(first), to_S_iterator(last)).base(), &ar);
	}
	using std::vector<S>::push_back; // virtual proxy
	void push_back(const cgroup_impl& val)
	{
		return push_back(dynamic_cast<const S&>(val));
	}
	using std::vector<S>::emplace_back; // can't virtualize (template)
	void pop_back() override
	{
		return std::vector<S>::pop_back();
	}
	void resize(size_t count) override
	{
		return std::vector<S>::resize(count); //virtualize}
	}
	void resize(size_t count, const cgroup_impl& val) override
	{
		return std::vector<S>::resize(count, dynamic_cast<const S&>(val));
	}
	using std::vector<S>::resize;
	using std::vector<S>::swap; //virtualize
	void swap(cgroup_vector_impl& other) override
	{
		std::vector<S>::swap(dynamic_cast<conc_cgroup_vector_impl&>(other));
	}

private:
	typename cgroup_vector_impl::iterator begin_impl() override
	{
		return cgroup_vector_impl::iterator(begin().base(), &ar);
	}
	typename cgroup_vector_impl::iterator end_impl() override
	{
		return cgroup_vector_impl::iterator(end().base(), &ar);
	}

	typename cgroup_vector_impl::const_iterator cbegin_impl() const override
	{
		return cgroup_vector_impl::const_iterator(cbegin().base(), &ar);
	}
	typename cgroup_vector_impl::const_iterator cend_impl() const override
	{
		return cgroup_vector_impl::const_iterator(cend().base(), &ar);
	}
	static const_iterator to_S_iterator(const_cgroup_iterator in)
	{
		if (in.vt() != &ar)
			throw std::bad_cast();
		const S* _ptr = static_cast<const S*>(in.base());
		return const_iterator(_ptr);
	}
	static reverse_iterator to_S_iterator(std::reverse_iterator<cgroup_iterator> in)
	{
		return reverse_iterator(to_S_iterator(in.base())); //base() of reverse_iterator<base_iterator> is of type base_iterator
	}
	static const_reverse_iterator to_S_iterator(std::reverse_iterator<const_cgroup_iterator> in)
	{
		return const_reverse_iterator(to_S_iterator(in.base())); //base() of reverse_iterator<base_iterator> is of type base_iterator
	}
	static iterator to_S_iterator(cgroup_iterator in)
	{
		if (in.vt() != &ar)
			throw std::bad_cast();
		return iterator(static_cast<S*>(in.base()));
	}

	std::unique_ptr<cgroup_vector_impl> clone() const override
	{
		return std::make_unique<conc_cgroup_vector_impl>(*this);
	}
};

template <class... T>
std::unique_ptr<cgroup_vector_impl> conc_cgroup_impl<T...>::make_vector(size_t cnt) const
{
	return std::make_unique<conc_cgroup_vector_impl<conc_cgroup_impl<T...>>>(cnt, *this);
}

} // namespace quantt

#endif /* BBF1F73E_87CC_4C69_9CCC_5D2526535A4F */
