/*
 * File: flat_map.h
 * Project: quantt
 * File Created: Thursday, 15th October 2020 1:03:53 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Thursday, 15th October 2020 1:03:53 pm
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#ifndef D7E9786D_BD4E_41BF_A6C5_4E902E127A7D
#define D7E9786D_BD4E_41BF_A6C5_4E902E127A7D

#include "doctest/doctest_proxy.h"
#include <algorithm>
#include <cassert>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <functional>
#include <vector>
namespace quantt
{
template <class Key, class Value, class Comp_less = std::less<Key>, class Allocator = std::allocator<std::pair<const Key, Value>>, template <class...> class Array = std::vector>
class flat_map
{
public:
	using key_type = Key;
	using mapped_type = Value;
	using value_type = std::pair<Key, Value>;

	using content_t = Array<std::pair<Key, Value>, Allocator>;

	// currently giving access to keys in a mutable form... that's bad.
	// can't have the key be stored as const because of inplace sorting.
	// this means creating a proxy iterator >< to ensure user cannot screw the ordering.
	using size_type = typename content_t::size_type;
	using difference_type = typename content_t::difference_type;
	using key_compare = Comp_less;
	using allocator_type = Allocator;
	using reference = value_type&;
	using const_reference = const value_type&;
	using pointer = typename Allocator::pointer;
	using const_pointer = typename Allocator::const_pointer;
	class iterator : public content_t::iterator //algorithm specifcally for iterator of this type will no accept iterator of a content_t
	{
	public:
		using content_t::iterator::iterator;
		explicit iterator(const typename content_t::iterator& in) : content_t::iterator(in) {}
		iterator operator+(std::ptrdiff_t in) { return iterator(content_t::iterator::operator+(in)); }
		iterator operator-(std::ptrdiff_t in) { return iterator(content_t::iterator::operator-(in)); }
	}; //require random access.
	struct const_iterator : public content_t::const_iterator
	{
	public:
		using content_t::const_iterator::const_iterator;
		explicit const_iterator(const typename content_t::const_iterator& in) : content_t::const_iterator(in) {}
		explicit const_iterator(const typename content_t::iterator& in) : content_t::const_iterator(in) {}
		const_iterator(const iterator& in) : content_t::const_iterator(in) {}
		const_iterator operator+(std::ptrdiff_t in) { return const_iterator(content_t::const_iterator::operator+(in)); }
		const_iterator operator-(std::ptrdiff_t in) { return const_iterator(content_t::const_iterator::operator-(in)); }
	}; //require random access.
	using reverse_iterator = std::reverse_iterator<iterator>;
	using const_reverse_iterator = std::reverse_iterator<const_iterator>;
	class value_compare
	{
		friend class flat_map;

	protected:
		key_compare comp;

	public:
		value_compare(Comp_less _c) : comp(std::move(_c)) {}
		value_compare() : comp() {}
		bool operator()(const value_type& a, const value_type& b)
		{
			return comp(a.first, b.first);
		}
		bool operator()(const value_type& a, const key_type& b)
		{
			return comp(a.first, b);
		}
		bool operator()(const key_type& a, const value_type& b)
		{
			return comp(a, b.first);
		}
		bool operator()(const key_type& a, const key_type& b)
		{
			return comp(a, b);
		}
	};
	//constructors
	// flat_map() : comp(), content(){};
	explicit flat_map(size_type capacity, const key_compare& _comp = key_compare(),
	                  const allocator_type& _alloc = allocator_type())
	    : comp(_comp), content(capacity, _alloc) { content.resize(0); }
	explicit flat_map(const key_compare& _comp, const allocator_type& _alloc = allocator_type())
	    : comp(_comp), content(_alloc) {}
	flat_map(const allocator_type& _alloc = allocator_type())
	    : comp(), content(_alloc) {}
	flat_map(content_t&& in) : comp(), content(std::move(in))
	{
		sort(content.begin(), content.end());
	}
	flat_map(const content_t& in) : comp(), content((in))
	{
		sort(content.begin(), content.end());
	}
	template <class InputIt>
	flat_map(InputIt first, InputIt last, const key_compare& _comp = key_compare(),
	         const allocator_type& _alloc = allocator_type())
	    : comp(_comp), content(first, last, _alloc)
	{
		sort(content.begin(), content.end());
		auto l = filter_unique(content.begin(), content.end());
		content.resize(std::distance(content.begin(), l));
	}
	template <class InputIt>
	flat_map(InputIt first, InputIt last, const allocator_type& _alloc)
	    : comp(), content(first, last, _alloc)
	{
		sort(content.begin(), content.end());
		auto l = filter_unique(content.begin(), content.end());
		content.resize(std::distance(content.begin(), l));
	}
	flat_map(const flat_map& other) : comp(other.comp), content(other.content) {}
	flat_map(const flat_map& other, const allocator_type& alloc) : comp(other.comp), content(other.content, alloc) {}
	flat_map(flat_map&& other) : comp(std::move(other.comp)), content(std::move(other.content)) {}
	flat_map(flat_map&& other, const allocator_type& alloc) : comp(std::move(other.comp)), content(std::move(other.content), alloc) {}
	flat_map(std::initializer_list<value_type> init) : comp(), content(std::move(init))
	{
		sort(content.begin(), content.end());
		auto l = filter_unique(content.begin(), content.end());
		content.resize(std::distance(content.begin(), l));
	}
	flat_map(std::initializer_list<value_type> init, const value_compare& _comp,
	         const allocator_type& alloc) : comp(_comp), content(std::move(init), alloc)
	{
		sort(content.begin(), content.end());
		auto l = filter_unique(content.begin(), content.end());
		content.resize(std::distance(content.begin(), l));
	}
	flat_map(std::initializer_list<value_type> init, const allocator_type& alloc)
	    : comp(), content(std::move(init), alloc)
	{
		sort(content.begin(), content.end());
		auto l = filter_unique(content.begin(), content.end());
		content.resize(std::distance(content.begin(), l));
	}

	//assigment
	flat_map& operator=(const flat_map& other)
	{
		content = other.content;
		comp = other.compu;
		return *this;
	}
	flat_map& operator=(flat_map&& other) noexcept
	{
		content = std::move(other.content);
		comp = std::move(other.comp);
		return *this;
	}
	flat_map& operator=(std::initializer_list<value_type> ilist)
	{
		content.resize(0);
		content.insert(ilist.begin(), ilist.end());
		return *this;
	}

	allocator_type get_allocator() const noexcept
	{
		return content.get_allocator();
	}
	//element access
	mapped_type& at(const key_type& key)
	{
		auto it = std::lower_bound(content.begin(), content.end(), key, comp);
		auto val = comp(key, *it);
		if (it == content.end() or val)
			//we should enter here only if *it != key or it is at the end
			throw std::out_of_range("key absent from flat_map.");
		return it->second;
	}
	const mapped_type& at(const key_type& key) const
	{
		return const_cast<flat_map*>(this)->at(key);
	}
	mapped_type& operator[](const key_type key)
	{
		auto it = std::lower_bound(content.begin(), content.end(), key, comp);
		if (it == content.end() or comp(key, *it))
			it = content.insert(it, std::make_pair(key, mapped_type()));
		return it->second;
		//no const version of this one, because it inserts an element if the key isn't already present.
	}
	//iterators
	iterator begin()
	{
		return iterator(content.begin());
	}
	iterator end()
	{
		return iterator(content.end());
	}
	const_iterator cbegin() const
	{
		return const_iterator(content.cbegin());
	}
	const_iterator begin() const
	{
		return const_iterator(cbegin());
	}
	const_iterator cend() const
	{
		return const_iterator(content.cend());
	}
	const_iterator end() const
	{
		return cend();
	}
	reverse_iterator rbegin()
	{
		return reverse_iterator(end());
	}
	reverse_iterator rend()
	{
		return reverse_iterator(begin());
	}
	const_reverse_iterator crbegin() const
	{
		return const_reverse_iterator(cend());
	}
	const_reverse_iterator rbegin() const
	{
		return crbegin();
	}
	const_reverse_iterator crend() const
	{
		return const_reverse_iterator(cbegin());
	}
	const_reverse_iterator rend() const
	{
		return crend();
	}
	//capacity
	bool empty() const
	{
		return content.empty();
	}
	size_type size() const
	{
		return content.size();
	}
	size_type capacity() const
	{
		return content.capacity();
	}
	size_type max_size() const
	{
		return content.max_size();
	}
	void reserve(size_type new_cap)
	{
		content.reserve(new_cap);
	}
	void shrink_to_fit()
	{
		content.shrink_to_fit();
	}
	//modifiers
	void clear()
	{
		content.clear();
	}
	std::pair<iterator, bool> insert(const value_type& value)
	{
		bool success = false; // true if the value was actually inserted.
		auto it = std::lower_bound(begin(), end(), value.first, comp);
		if (it->first != value.first)
		{
			it = content.insert(it, value);
			success = true;
		}
		return std::make_pair(it, success);
	}
	std::pair<iterator, bool> insert(value_type&& value)
	{
		bool success = false; // true if the value was actually inserted.
		auto it = std::lower_bound(begin(), end(), value.first, comp);
		if (it->first != value.first)
		{
			it = content.insert(it, std::move(value));
			success = true;
		}
		return std::make_pair(it, success);
	}
	template <class P, class = std::enable_if_t<std::is_constructible_v<value_type, P&&>>>
	std::pair<iterator, bool> insert(P&& value)
	{
		static_assert(std::is_constructible_v<value_type, P&&>, "Don't try to bypass the enable if");
		return insert(value_type(value));
	}
	iterator insert(const_iterator hint, const value_type& value)
	{
		auto [first, last] = use_hint(hint, value.first);
		iterator it = std::lower_bound(first, last, value, comp); //that's not a big saving if the hint is really good. except if the element is near the ends
		if (it->first != value.first)
		{
			it = content.insert(it, value);
		}
		return it;
	}
	iterator insert(const_iterator hint, value_type&& value)
	{
		auto [first, last] = use_hint(hint, value.first);
		iterator it = std::lower_bound(first, last, value, comp); //that's not a big saving if the hint is really good. except if the element is near the ends
		if (it->first != value.first)
		{
			it = content.insert(it, std::move(value));
		}
		return it;
	}
	template <class P, class = std::enable_if_t<std::is_constructible_v<value_type, P&&>>>
	iterator insert(const_iterator hint, P&& value)
	{
		static_assert(std::is_constructible_v<value_type, P&&>, "Don't try to bypass the enable if");
		return insert(hint, value_type(std::forward<P>(value)));
	}
	template <class InputIt>
	void insert(InputIt first, InputIt last)
	{
		insert(first, last, [](auto&, const auto&) {});
	}
	template <class InputIt, class Collision, class = std::enable_if_t<!std::is_convertible_v<InputIt, const_iterator>>>
	void insert(InputIt first, InputIt last, Collision&& collision)
	{
		static_assert(!std::is_convertible_v<InputIt, const_iterator>, "Don't try to bypass the enable if");
		content.insert(content.end(), first, last);
		sort();
		auto l = filter_unique(content.begin(), content.end(), collision); //pack the unique element at the begining, mainting order
		content.resize(l - begin());                                       //l is the new end pointer.
	}
	//specialisation for the case of a inserting an ordered array, we can take some small shortcut in that case.
	//should be templated on the iterator such that it accept iterator to ordered array with a value_type convertible to this value_type.
	template <class Collision>
	void insert(
	    const_iterator first, const_iterator last, Collision&& collision)
	{
		auto look = begin();
		{
			auto n1 = std::distance(first, last); //n1 is the number of element to be copied.
			//in first approxiamtion, it's the number of element in the input
			//detect collisions between content and the input iterators.
			for (auto i = first; i != last; ++i)
			{
				look = std::lower_bound(look, end(), *i, comp);
				if (look == end())
					break;
				if (!comp(*i, *look))
				{
					--n1; //correct it by finding all the collisions.
					      // collision(look->second, i->second);
				}
			}
			content.resize(content.size() + n1); //potentially invalidates iterators. could be more time efficient if we need to reallocate, at the cost of holding the previous memory space  abit longer
			//Copy the stuff from the input that has a larger key than the largest key in this.
			look = end() - n1; //the pointer past the end of the original, to be moved, elements
		}                      //pop n1 from the stack
		auto move_to_end = end();
		while (move_to_end != look and look != begin() and last != first)
		{
			decltype(first - last) found_collision;
			//part A: copy element in [first,last[ larger than *(look-1) at move_to_end.
			{
				auto la = std::lower_bound(first, last, *(look - 1), comp);
				found_collision = la != last and !comp(*(look - 1), *la);
				std::copy_backward(la + found_collision, last, move_to_end);
				move_to_end -= last - (la + found_collision);
				last = la;
			} //pop la from the stack
			if (found_collision)
				collision(move_to_end->second, last->second); // if we move the collision treatment here, to allow parallelism.
			//part B: move the elements in [begin,look[ that are greater than *(last-1) to move_to_end
			if (last == first) // we've copied everything, there's nothing to move.
			{
				break;
			}
			{
				auto lb = std::lower_bound(begin(), look, *(last - 1), comp);
				std::move_backward(lb, look, move_to_end);
				move_to_end -= look - lb;
				look = lb;
			} //pop lb from the stack
			found_collision = !comp(*(last - 1), *move_to_end);
			if (found_collision)
				collision(move_to_end->second, (last - 1)->second); //if we move collision treatment here, to allow parallelism
			last -= found_collision;
		}
		assert((last - first) == 0 or (move_to_end - begin()) == (last - first)); //we've copied everything, or we still have some stuff to copy and the room necessary
		std::copy_backward(first, last, move_to_end);
	}

	template <class M>
	std::pair<iterator, bool> insert_or_assign(const key_type& k, M&& obj)
	{
		bool success = false; // true if the value was actually inserted.
		auto it = std::lower_bound(begin(), end(), k, comp);
		if (comp(k, *it))
		{
			it = content.emplace(it, k, std::forward<M>(obj));
			success = true;
		}
		else
		{
			*it = value_type(k, std::forward<M>(obj));
		}
		return std::make_pair(it, success);
	}
	template <class M>
	std::pair<iterator, bool> insert_or_assign(key_type&& k, M&& obj)
	{
		bool success = false; // true if the value was actually inserted.
		auto it = std::lower_bound(begin(), end(), k, comp);
		if (comp(k, *it))
		{
			it = content.emplace(it, std::move(k), std::forward<M>(obj));
			success = true;
		}
		else
		{
			*it = value_type(std::move(k), std::forward<M>(obj));
		}
		return std::make_pair(it, success);
	}
	template <class M>
	iterator insert_or_assign(const_iterator hint, const key_type& k, M&& obj)
	{

		auto [first, last] = use_hint(hint, k);
		iterator it = std::lower_bound(first, last, k, comp); //that's not a big saving if the hint is really good. except if the element is near the ends
		if (comp(k, *it))
		{
			it = content.emplace(it, k, std::forward<M>(obj));
		}
		else
		{
			*it = value_type(k, std::forward<M>(obj));
		}

		return it;
	}
	template <class M>
	iterator insert_or_assign(const_iterator hint, key_type&& k, M&& obj)
	{
		auto [first, last] = use_hint(hint, k);
		iterator it = std::lower_bound(first, last, k, comp); //that's not a big saving if the hint is really good. except if the element is near the ends
		if (comp(k, *it))
		{
			it = content.emplace(it, std::move(k), std::forward<M>(obj));
		}
		else
		{
			*it = value_type(std::move(k), std::forward<M>(obj));
		}
		return it;
	}
	template <class... Args>
	std::pair<iterator, bool> emplace(Args&&... args)
	{
		value_type val(std::forward<Args>(args)...);
		bool success = false;
		auto it = std::lower_bound(begin(), end(), val, comp);
		if (comp(val, *it))
		{
			it = content.emplace(it, std::move(val));
			success = true;
		}
		return std::make_pair(it, success);
	}
	template <class... Args>
	std::pair<iterator, bool> emplace(const_iterator hint, Args&&... args)
	{
		value_type val(std::forward<Args>(args)...);
		bool success = false;
		auto [first, last] = use_hint(hint, val.first);
		iterator it = std::lower_bound(first, last, val, comp); //that's not a big saving if the hint is really good. except if the element is near the ends
		if (comp(val, *it))
		{
			it = content.emplace(it, std::move(val));
			success = true;
		}
		return std::make_pair(it, success);
	}

	template <class... Args>
	std::pair<iterator, bool> try_emplace(const key_type& k, Args&&... args)
	{
		iterator it = std::lower_bound(begin(), end(), k, comp);
		bool success = false;
		if (comp(k, *it))
		{
			it = content.emplace(it, k, std::forward<Args>(args)...);
			success = true;
		}
		return std::make_pair(it, success);
	}
	template <class... Args>
	std::pair<iterator, bool> try_emplace(key_type&& k, Args&&... args)
	{
		iterator it = std::lower_bound(begin(), end(), k, comp);
		bool success = false;
		if (comp(k, *it))
		{
			it = content.emplace(it, std::move(k), std::forward<Args>(args)...);
			success = true;
		}
		return std::make_pair(it, success);
	}
	template <class... Args>
	std::pair<iterator, bool> try_emplace(const_iterator hint, const key_type& k, Args&&... args)
	{
		auto [first, last] = use_hint(hint, k);
		iterator it = std::lower_bound(first, last, k, comp);
		bool success = false;
		if (comp(k, *it))
		{
			it = content.emplace(it, k, std::forward<Args>(args)...);
			success = true;
		}
		return std::make_pair(it, success);
	}
	template <class... Args>
	std::pair<iterator, bool> try_emplace(const_iterator hint, key_type&& k, Args&&... args)
	{
		auto [first, last] = use_hint(hint, k);
		iterator it = std::lower_bound(first, last, k, comp);
		bool success = false;
		if (comp(k, *it))
		{
			it = content.emplace(it, std::move(k), std::forward<Args>(args)...);
			success = true;
		}
		return std::make_pair(it, success);
	}

	iterator erase(const_iterator pos)
	{
		return iterator(content.erase(pos));
	}
	iterator erase(iterator pos)
	{
		return iterator(content.erase(pos));
	}
	void erase(const_iterator first, const_iterator last)
	{
		content.erase(first, last);
	}
	size_type erase(const key_type& k)
	{
		auto it = std::lower_bound(begin(), end(), k, comp);
		if (comp(*it, k))
		{
			return 0;
		}
		content.erase(it);
		return 1;
	}
	void swap(flat_map& other) noexcept
	{
		using std::swap;
		swap(content, other.content);
		swap(comp, other.comp);
	}
	template <class C2, class Collision>
	void merge(const flat_map<key_type, mapped_type, C2, allocator_type>& source, Collision collision)
	{
		insert(source.begin(), source.end(), collision);
	}

	template <class C2, class Collision>
	void merge(flat_map<key_type, mapped_type, C2, allocator_type>&& source, Collision collision)
	{
		if (source.capacity() >= source.size() + size()) //reuse source's resources if it avoid doing an allocation.
		{
			swap(content, source.content); //if the source has enough capacity, we'll keep it instead.
			insert(source.begin(), source.end(), [collision](auto& a, auto& b) {std::swap(a,b);collision(a,b); });
		}
		else
		{
			insert(source.begin(), source.end(), collision);
		}
	}
	template <class C2>
	void merge(const flat_map<key_type, mapped_type, C2, allocator_type>& source)
	{
		insert(source.begin(), source.end());
	}

	template <class C2>
	void merge(flat_map<key_type, mapped_type, C2, allocator_type>&& source)
	{
		if (source.capacity() >= source.size() + size())
		{
			swap(content, source.content); //if the source has enough capacity, we'll keep it instead.
			using std::swap;
			insert(source.begin(), source.end(), swap);
		}
		else
		{
			insert(source.begin(), source.end());
		}
	}

	iterator find(const key_type& k)
	{
		auto it = std::lower_bound(begin(), end(), k, comp);
		if (comp(*it, k))
			return end();
		return it;
	}
	const_iterator find(const key_type& k) const
	{
		return const_cast<flat_map*>(this)->find(k);
	}
	size_type count(const key_type& k) const
	{
		return find(k) != end();
	}
	bool contains(const key_type& k) const
	{
		return count(k);
	}
	std::pair<iterator, iterator> equal_range(const key_type& k)
	{
		auto it = find(k);
		return std::make_pair(it, it == end() ? end() : it + 1);
	}
	std::pair<const_iterator, const_iterator> equal_range(const key_type& k) const
	{
		return std::make_from_tuple<std::pair<const_iterator, const_iterator>>(const_cast<flat_map*>(this)->equal_range(k));
	}
	iterator lower_bound(const key_type& k)
	{
		auto it = std::lower_bound(begin(), end(), k, comp);
		if (comp(k, *it))
			it = end();
		return it;
	}
	const_iterator lower_bound(const key_type& k) const
	{
		auto it = std::lower_bound(cbegin(), cend(), k, comp);
		if (comp(k, *it))
			it = end();
		return it;
	}
	iterator upper_bound(const key_type& k)
	{
		auto it = std::upper_bound(begin(), end(), k, comp);
		if (comp(*it, k))
			it = end();
		return it;
	}
	const_iterator upper_bound(const key_type& k) const
	{
		auto it = std::upper_bound(begin(), end(), k, comp);
		if (comp(*it, k))
			it = end();
		return it;
	}

	key_compare key_comp() const
	{
		return comp.comp;
	}

	value_compare value_comp() const
	{
		return comp;
	}

	/**
	* @brief sort the flat_map. So long as you only manipulate the content of the map using it's methods, you do not need to reorder it.
	* 
	*  for the block tensor class, some manipulation of the index could be necessary.
	*  If the transformation on the index changes the ordering, call sort afterward. Or face the consequences.
	*  All the method of this class are UB if it is not sorted.
	* 
	* @param first iterator to the first element of the range to sort
	* @param last iterator to the element after the range to sort
	*
	*/
	void sort(iterator first, iterator last)
	{
		std::sort(first, last, comp);
	}
	void sort()
	{
		std::sort(begin(), end(), comp);
	}

protected:
	value_compare comp;

private:
	content_t content;
	void sort(typename content_t::iterator first, typename content_t::iterator last)
	{
		std::sort(first, last, comp);
	}
	// void remove_collisions(iterator first = begin(), iterator last = end())
	// { //assume a sorted array
	// 	auto beg = first while (first != last)
	// 	{
	// 	}
	// }
	std::pair<const_iterator, const_iterator> use_hint(const_iterator hint, key_type& key) const
	{
		const_iterator first = comp(*hint, key) ? hint : begin();
		const_iterator last = comp(key, *hint) ? end() : hint + 1;
		return std::make_pair(first, last);
	}
	std::pair<iterator, iterator> use_hint(const_iterator hint, key_type& key)
	{
		iterator first = comp(*hint, key) ? const_cast<value_type*>(hint.base()) : begin();
		iterator last = comp(key, *hint) ? const_cast<value_type*>(hint.base() + 1) : end();
		return std::make_pair(first, last);
	}
	typename content_t::iterator filter_unique(typename content_t::iterator first, typename content_t::iterator last)
	{
		return filter_unique(first, last, [](auto&, auto&) {});
	}
	template <class Collision>
	typename content_t::iterator filter_unique(typename content_t::iterator first, typename content_t::iterator last, Collision collision)
	{
		using std::swap;
		auto l = first;
		auto second = std::upper_bound(first, last, *l, comp);
		while (first != last)
		{
			std::for_each(first + 1, second, [collision, l](auto& x) { collision(*l, x); });
			++l;
			swap(*l, *second); //when there's only one element of a given value it swaps with self.
			first = second;
			second = std::upper_bound(first, last, *l, comp);
		}
		return l;
	}
};

template <class key, class val, class C1, class C2, class alloc1, class alloc2, template <class...> class array1, template <class...> class array2>
bool operator==(const flat_map<key, val, C1, alloc1, array1>& a, const flat_map<key, val, C2, alloc2, array2>& b)
{
	if (&a == &b)
		return true;
	if (a.size() != b.size())
		return false;
	auto it_a = a.begin();
	auto it_b = b.begin();
	bool out = true;
	while (out and it_a != a.end())
	{
		out &= *it_a == *it_b;
		++it_a;
		++it_b;
	}
	return out;
}

qtt_TEST_CASE("flat_map")
{
	flat_map<int, double> a{{40, 1.1}, {20, 1.2}, {30, 1.3}, {10, 1.4}};
	qtt_REQUIRE(a == flat_map<int, double>({{10, 1.4}, {20, 1.2}, {30, 1.3}, {40, 1.1}}));

	// qtt_CHECK(false == true);
	qtt_SUBCASE("insert a flat_map before, in and at the end, with collisions")
	{
		int collisions = 0;
		flat_map<int, double> b{{1, 6e10}, {10, 6e10}, {50, 1e10}, {30, 1}, {33, 1e12}, {70, 1e11}};
		flat_map<int, double> result{{1, 6e10}, {10, 1.4}, {20, 1.2}, {30, 1.3}, {33, 1e12}, {40, 1.1}, {50, 1e10}, {70, 1e11}};
		a.insert(b.begin(), b.end(), [&collisions](auto&&, auto&&) { ++collisions; });
		qtt_CHECK(a == result);
		qtt_CHECK(collisions == 2);
	}
	qtt_SUBCASE("insert a flat_map in middle and at the end, with collisions")
	{
		int collisions = 0;
		flat_map<int, double> b{{10, 6e10}, {50, 1e10}, {33, 1e12}, {40, 5.5}, {70, 1e11}};
		flat_map<int, double> result{{10, 1.4}, {20, 1.2}, {30, 1.3}, {33, 1e12}, {40, 1.1}, {50, 1e10}, {70, 1e11}};
		a.insert(b.begin(), b.end(), [&collisions](auto&&, auto&&) { ++collisions; });
		qtt_CHECK(a == result);
		qtt_CHECK(collisions == 2);
	}
	qtt_SUBCASE("insert a flat_map in middle, with collisions")
	{
		int collisions = 0;
		flat_map<int, double> b{{10, 6e10}, {33, 1e12}, {31, 1e12}, {40, 5.5}};
		flat_map<int, double> result{{10, 1.4}, {20, 1.2}, {31, 1e12}, {30, 1.3}, {33, 1e12}, {40, 1.1}};
		a.insert(b.begin(), b.end(), [&collisions](auto&&, auto&&) { ++collisions; });
		qtt_CHECK(a == result);
		qtt_CHECK(collisions == 2);
	}
	qtt_SUBCASE("insert a flat_map before, in middle and at the end, no collisions")
	{
		int collisions = 0;
		flat_map<int, double> b{{1, 6e10}, {50, 1e10}, {33, 1e12}, {70, 1e11}};
		flat_map<int, double> result{{1, 6e10}, {10, 1.4}, {20, 1.2}, {30, 1.3}, {33, 1e12}, {40, 1.1}, {50, 1e10}, {70, 1e11}};
		a.insert(b.begin(), b.end(), [&collisions](auto&&, auto&&) { ++collisions; });
		qtt_CHECK(a == result);
		qtt_CHECK(collisions == 0);
	}
	qtt_SUBCASE("insert a flat_map in middle and at the end, no collisions")
	{
		int collisions = 0;
		flat_map<int, double> b{{50, 1e10}, {33, 1e12}, {70, 1e11}};
		flat_map<int, double> result{{10, 1.4}, {20, 1.2}, {30, 1.3}, {33, 1e12}, {40, 1.1}, {50, 1e10}, {70, 1e11}};
		a.insert(b.begin(), b.end(), [&collisions](auto&&, auto&&) { ++collisions; });
		qtt_CHECK(a == result);
		qtt_CHECK(collisions == 0);
	}
	qtt_SUBCASE("insert a flat_map in middle, no collisions")
	{
		int collisions = 0;
		flat_map<int, double> b{{33, 1e12}, {31, 1e12}};
		flat_map<int, double> result{{10, 1.4}, {20, 1.2}, {31, 1e12}, {30, 1.3}, {33, 1e12}, {40, 1.1}};
		a.insert(b.begin(), b.end(), [&collisions](auto&&, auto&&) { ++collisions; });
		qtt_CHECK(a == result);
		qtt_CHECK(collisions == 0);
	}
	qtt_SUBCASE("insert unsorted sequence")
	{
		int collisions = 0;
		std::vector<std::pair<int, double>> b{{1, 6e10}, {1, 6e10}, {10, 6e10}, {50, 1e10}, {50, 1e10}, {50, 1e10}, {30, 1}, {33, 1e12}, {70, 1e11}};
		flat_map<int, double> result{{10, 1.4}, {20, 1.2}, {30, 1.3}, {33, 1e12}, {40, 1.1}, {1, 6e10}, {50, 1e10}, {70, 1e11}};
		a.insert(b.begin(), b.end(), [&collisions](auto&&, auto&&) { ++collisions; });
		qtt_CHECK(a == result);
		qtt_CHECK(collisions == 5);
	}
}
qtt_TEST_CASE("accessors")
{
	flat_map<int, int> A;
	A[0] = 5;
	A[0] = 6;
	qtt_SUBCASE("first access")
	{
		qtt_REQUIRE_NOTHROW(A.at(0));
		qtt_CHECK(A[0] == 6);
	}
}

} // namespace quantt
#endif /* D7E9786D_BD4E_41BF_A6C5_4E902E127A7D */
