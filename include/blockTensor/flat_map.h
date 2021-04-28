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
#include <boost/stl_interfaces/iterator_interface.hpp>
#include <cassert>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <functional>
#include <iterator>
#include <numeric>
#include <vector>

namespace quantt
{
template <class Key, class Value, class Comp_less = std::less<Key>,
          class Allocator = std::allocator<std::pair<Key, Value>>, template <class...> class Array = std::vector>
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
	using reference = typename content_t::reference;
	using const_reference = typename content_t::const_reference;
	using pointer = typename content_t::pointer;
	using const_pointer = typename content_t::const_pointer;
	using iterator_tag = typename std::iterator_traits<typename content_t::iterator>::iterator_category;
	using const_iterator_tag = typename std::iterator_traits<typename content_t::const_iterator>::iterator_category;
	static_assert(std::is_same_v<iterator_tag, std::random_access_iterator_tag>,
	              "The container type must be random access");
	static_assert(std::is_same_v<const_iterator_tag, std::random_access_iterator_tag>,
	              "The container type must be random access");
	class const_iterator;
	class iterator : public boost::stl_interfaces::iterator_interface<iterator, iterator_tag, value_type, reference, pointer,
	                                                                  difference_type>
	// algorithm specifically for iterator of this type will not accept iterator of a content_t
	{
		using it_type = typename content_t::iterator;
		it_type it;
		friend const_iterator;

	  public:
		explicit constexpr iterator(const typename content_t::iterator &in) : it(in) {}
		constexpr iterator() : it() {}
		constexpr iterator(const iterator &in) : it(in.it) {}
		constexpr iterator &operator+=(std::ptrdiff_t in)
		{
			it += in;
			return *this;
		}
		// constexpr difference_type operator-(iterator in) { return it - in.it; }
		constexpr value_type *base() { return it.base(); }
		// reference operator*() const {return *it;}
		// bool operator==( const iterator& in) {return it == in.it;}

	  private:
		friend boost::stl_interfaces::access;
		friend flat_map;
		constexpr it_type &base_reference() noexcept { return it; }
		constexpr it_type base_reference() const noexcept { return it; }
		operator it_type() const {return base_reference(); }
	}; // require random access.
	class const_iterator
	    : public boost::stl_interfaces::iterator_interface<const_iterator, const_iterator_tag, const value_type,
	                                                       const_reference,const_pointer, difference_type>
	// algorithm specifcally for iterator of this type will no accept iterator of a content_t
	{
		using it_type = typename content_t::const_iterator;
		it_type it;

	  public:
		explicit constexpr const_iterator(const typename content_t::iterator &in) : it(in) {}
		explicit constexpr const_iterator(const typename content_t::const_iterator &in) : it(in) {}
		constexpr const_iterator() : it() {}
		constexpr const_iterator(const iterator &in) : it(in.it) {}
		constexpr const_iterator(const const_iterator &in) : it(in.it) {}
		constexpr const_iterator &operator+=(std::ptrdiff_t in)
		{
			it += in;
			return *this;
		}
		// constexpr difference_type operator-(const_iterator in) { return it - in.it; }
		constexpr const value_type *base() { return it.base(); }
		// const_reference operator*() const {return *it;}
		// bool operator==( const const_iterator& in) {return it == in.it;}
	  private:
		friend boost::stl_interfaces::access;
		friend flat_map;
		constexpr it_type &base_reference() noexcept { return it; }
		constexpr it_type base_reference() const noexcept { return it; }
		operator it_type() const {return base_reference(); }
	}; // require random access.
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
		bool operator()(const value_type &a, const value_type &b) const { return comp(a.first, b.first); }
		bool operator()(const value_type &a, const key_type &b) const { return comp(a.first, b); }
		bool operator()(const key_type &a, const value_type &b) const { return comp(a, b.first); }
		bool operator()(const key_type &a, const key_type &b) const { return comp(a, b); }
	};
	// constructors
	// flat_map() : comp(), content(){};
	explicit flat_map(size_type capacity, const key_compare &_comp = key_compare(),
	                  const allocator_type &_alloc = allocator_type())
	    : comp(_comp), content(capacity, _alloc)
	{
		content.resize(0);
	}
	explicit flat_map(const key_compare &_comp, const allocator_type &_alloc = allocator_type())
	    : comp(_comp), content(_alloc)
	{
	}
	flat_map(const allocator_type &_alloc = allocator_type()) : comp(), content(_alloc) {}
	flat_map(content_t &&in) : comp(), content(std::move(in)) { sort(content.begin(), content.end()); }
	flat_map(const content_t &in) : comp(), content((in)) { sort(content.begin(), content.end()); }
	template <class InputIt>
	flat_map(InputIt first, InputIt last, const key_compare &_comp = key_compare(),
	         const allocator_type &_alloc = allocator_type())
	    : comp(_comp), content(first, last, _alloc)
	{
		sort(content.begin(), content.end());
		auto l = filter_unique(content.begin(), content.end());
		content.resize(std::distance(content.begin(), l));
	}
	template <class InputIt>
	flat_map(InputIt first, InputIt last, const allocator_type &_alloc) : comp(), content(first, last, _alloc)
	{
		sort(content.begin(), content.end());
		auto l = filter_unique(content.begin(), content.end());
		content.resize(std::distance(content.begin(), l));
	}
	flat_map(const flat_map &other) : comp(other.comp), content(other.content) {}
	flat_map(const flat_map &other, const allocator_type &alloc) : comp(other.comp), content(other.content, alloc) {}
	flat_map(flat_map &&other) : comp(std::move(other.comp)), content(std::move(other.content)) {}
	flat_map(flat_map &&other, const allocator_type &alloc)
	    : comp(std::move(other.comp)), content(std::move(other.content), alloc)
	{
	}
	flat_map(std::initializer_list<value_type> init) : comp(), content(std::move(init))
	{
		sort(content.begin(), content.end());
		auto l = filter_unique(content.begin(), content.end());
		content.resize(std::distance(content.begin(), l));
	}
	flat_map(std::initializer_list<value_type> init, const value_compare &_comp, const allocator_type &alloc)
	    : comp(_comp), content(std::move(init), alloc)
	{
		sort(content.begin(), content.end());
		auto l = filter_unique(content.begin(), content.end());
		content.resize(std::distance(content.begin(), l));
	}
	flat_map(std::initializer_list<value_type> init, const allocator_type &alloc)
	    : comp(), content(std::move(init), alloc)
	{
		sort(content.begin(), content.end());
		auto l = filter_unique(content.begin(), content.end());
		content.resize(std::distance(content.begin(), l));
	}

	// assigment
	flat_map &operator=(const flat_map &other)
	{
		content = other.content;
		comp = other.comp;
		return *this;
	}
	flat_map &operator=(flat_map &&other) noexcept
	{
		content = std::move(other.content);
		comp = std::move(other.comp);
		return *this;
	}
	/**
	 * @brief set the content of the flat_map to that of the arguement. Does not sort if input isn't already sorted.
	 * 
	 * TODO: add a call to sort?
	 * 
	 * @param ilist 
	 * @return flat_map& 
	 */
	flat_map &operator=(std::initializer_list<value_type> ilist)
	{
		content.resize(0);
		content.insert(ilist.begin(), ilist.end());
		return *this;
	}

	allocator_type get_allocator() const noexcept { return content.get_allocator(); }
	// element access
	mapped_type &at(const key_type &key)
	{
		auto it = std::lower_bound(content.begin(), content.end(), key, comp);
		if (it == content.end() or comp(key, *it))
			// we should enter here only if *it != key or it is at the end
			throw std::out_of_range("key absent from flat_map.");
		return it->second;
	}
	const mapped_type &at(const key_type &key) const { return const_cast<flat_map *>(this)->at(key); }
	mapped_type &operator[](const key_type key)
	{
		auto it = std::lower_bound(content.begin(), content.end(), key, comp);
		if (it == content.end() or comp(key, *it))
			it = content.insert(it, std::make_pair(key, mapped_type()));
		return it->second;
		// no const version of this one, because it inserts an element if the key isn't already present.
	}
	// iterators
	iterator begin() { return iterator(content.begin()); }
	iterator end() { return iterator(content.end()); }
	const_iterator cbegin() const { return const_iterator(content.cbegin()); }
	const_iterator begin() const { return const_iterator(cbegin()); }
	const_iterator cend() const { return const_iterator(content.cend()); }
	const_iterator end() const { return cend(); }
	reverse_iterator rbegin() { return reverse_iterator(end()); }
	reverse_iterator rend() { return reverse_iterator(begin()); }
	const_reverse_iterator crbegin() const { return const_reverse_iterator(cend()); }
	const_reverse_iterator rbegin() const { return crbegin(); }
	const_reverse_iterator crend() const { return const_reverse_iterator(cbegin()); }
	const_reverse_iterator rend() const { return crend(); }
	// capacity
	bool empty() const { return content.empty(); }
	size_type size() const { return content.size(); }
	size_type capacity() const { return content.capacity(); }
	size_type max_size() const { return content.max_size(); }
	void reserve(size_type new_cap) { content.reserve(new_cap); }
	void shrink_to_fit() { content.shrink_to_fit(); }
	// modifiers
	void clear() { content.clear(); }
	std::pair<iterator, bool> insert(const value_type &value)
	{
		bool success = false; // true if the value was actually inserted.
		auto it = std::lower_bound(begin(), end(), value.first, comp);
		if (it->first != value.first)
		{
			it = iterator(content.insert(it.base_reference(), value));
			success = true;
		}
		return std::make_pair(it, success);
	}
	std::pair<iterator, bool> insert(value_type &&value)
	{
		bool success = false; // true if the value was actually inserted.
		auto it = std::lower_bound(begin(), end(), value.first, comp);
		if (it->first != value.first)
		{
			it = iterator(content.insert(it.base_reference(), std::move(value)));
			success = true;
		}
		return std::make_pair(it, success);
	}
	template <class... P, class = std::enable_if_t<std::is_constructible_v<value_type, P&&...>>>
	std::pair<iterator, bool> insert(P&&... value)
	{
		static_assert(std::is_constructible_v<value_type, P &&...>, "Don't try to bypass the enable if");
		return insert(value_type(std::forward<P>(value)...));
	}
	iterator insert(const_iterator hint, const value_type &value)
	{
		auto [first, last] = use_hint(hint, value.first);
		auto it = std::lower_bound(
		    first, last, value,
		    comp); // that's not a big saving if the hint is really good. except if the element is near the ends
		if (it == end() or it->first != value.first)
		{
			return iterator(content.insert(it.base_reference(), value));
		}
		return erase(it,it); //convert the const_iterator in an iterator
	}
	iterator insert(const_iterator hint, value_type &value)
	{
		auto [first, last] = use_hint(hint, value.first);
		auto it = std::lower_bound(
		    first, last, value,
		    comp); // that's not a big saving if the hint is really good. except if the element is near the ends
		if (it == end() or it->first != value.first)
		{
			return iterator(content.insert(it.base_reference(), std::move(value)));
		}
		return erase(it,it);
	}
	template <class... P, class = std::enable_if_t<std::is_constructible_v<value_type, P&&...>>>
	iterator insert(const_iterator hint, P&& ... value)
	{
		static_assert(std::is_constructible_v<value_type, P&&...>, "Don't try to bypass the enable if");
		return insert(hint, value_type(std::forward<P>(value)...));
	}
	template <class InputIt>
	void insert(InputIt first, InputIt last)
	{
		insert(
		    first, last, [](auto &, const auto &) {}, [](auto &) {});
	}
	template <class InputIt, class Collision, class = std::enable_if_t<!std::is_convertible_v<InputIt, const_iterator>>>
	void insert(InputIt first, InputIt last, Collision &&collision)
	{
		static_assert(!std::is_convertible_v<InputIt, const_iterator>, "Don't try to bypass the enable if");
		content.insert(content.end(), const_iterator(first).base_reference(),const_iterator(last).base_reference());
		sort();
		auto l = filter_unique(
		    content.begin(), content.end(),
		    std::forward<Collision>(collision)); // pack the unique element at the begining, mainting order
		content.resize(l - content.begin());             // l is the new end pointer.
	}
	template <class Collision>
	void insert(const_iterator first, const_iterator last, Collision &&collision)
	{
		insert(first, last, std::forward<Collision>(collision), [](auto &&x) {});
	}
	// specialisation for the case of a inserting an ordered array, we can take some small shortcut in that case.
	// should be templated on the iterator such that it accept iterator to ordered array with a value_type convertible
	// to this value_type.
	template <class Collision, class NoCollision>
	void insert(const_iterator first, const_iterator last, Collision &&collision, NoCollision &&nocollision)
	{
		auto look = begin();
		{
			auto n1 = std::distance(first, last); // n1 is the number of element to be copied.
			// in first approxiamtion, it's the number of element in the input
			// detect collisions between content and the input iterators.
			for (auto i = first; i != last; ++i)
			{
				look = std::lower_bound(look, end(), *i, comp);
				if (look == end())
					break;
				if (!comp(*i, *look))
				{
					--n1; // correct it by counting all the collisions.
				}
			}
			content.resize(content.size() +
			               n1); // potentially invalidates iterators. could be more time efficient if we need to
			                    // reallocate, at the cost of holding the previous memory space  abit longer
			// Copy the stuff from the input that has a larger key than the largest key in this.
			look = end() - n1;    // the pointer past the end of the original, to be moved, elements
		}                         // pop n1 from the stack
		auto move_to_end = end(); // for copy backward
		// potential optimization: if move_to_end==look at this point, all new elements are colliding.
		while (look != begin() and last != first)
		{
			decltype(first - last) found_collision;
			// part A: copy element in [first,last[ larger than *(look-1) at move_to_end.
			{
				auto la = std::lower_bound(first, last, *(look - 1), comp);
				found_collision = la != last and !comp(*(look - 1), *la);
				std::copy_backward(la + found_collision, last, move_to_end);
				auto old_move_to_end = move_to_end;
				move_to_end -= last - (la + found_collision);
				std::for_each(move_to_end, old_move_to_end, [nocollision](auto &&x) { nocollision(std::get<1>(x)); });
				last = la;
			} // pop la from the stack
			// part B: move the elements in [begin,look[ that are greater than *(last-1) to move_to_end
			if (last == first)
				break; // we've copied everything, there's nothing to move.
			{
				auto lb = std::lower_bound(begin(), look, *(last - 1), comp);
				if (move_to_end != look)
					std::move_backward(lb, look,
					                   move_to_end); // we don't want to do anything here if move_to_end == look. plus
					                                 // move_backward behavior is undefined in that case.
				if (found_collision)
					collision((move_to_end - 1)->second,
					          last->second); // if we move the collision treatment here, we allow parallelism.
				move_to_end -= look - lb;
				look = lb;
			} // pop lb from the stack
			found_collision = !comp(*(last - 1), *move_to_end);
			if (found_collision)
				collision(move_to_end->second,
				          (last - 1)->second); // if we move collision treatment here, to allow parallelism
			last -= found_collision;
		}
		assert(
		    (last - first) == 0 or
		    (move_to_end - begin()) ==
		        (last - first)); // we've copied everything, or we still have some stuff to copy and the room necessary
		std::copy_backward(first, last, move_to_end);
		std::for_each(begin(), move_to_end, [nocollision](auto &&x) { nocollision(std::get<1>(x)); });
	}

	template <class M>
	std::pair<iterator, bool> insert_or_assign(const key_type &k, M &&obj)
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
	std::pair<iterator, bool> insert_or_assign(key_type &&k, M &&obj)
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
	iterator insert_or_assign(const_iterator hint, const key_type &k, M &&obj)
	{

		auto [first, last] = use_hint(hint, k);
		iterator it = std::lower_bound(
		    first, last, k,
		    comp); // that's not a big saving if the hint is really good. except if the element is near the ends
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
	iterator insert_or_assign(const_iterator hint, key_type &&k, M &&obj)
	{
		auto [first, last] = use_hint(hint, k);
		iterator it = std::lower_bound(
		    first, last, k,
		    comp); // that's not a big saving if the hint is really good. except if the element is near the ends
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
	std::pair<iterator, bool> emplace(Args &&... args)
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
	std::pair<iterator, bool> emplace(const_iterator hint, Args &&... args)
	{
		value_type val(std::forward<Args>(args)...);
		bool success = false;
		auto [first, last] = use_hint(hint, val.first);
		iterator it = std::lower_bound(
		    first, last, val,
		    comp); // that's not a big saving if the hint is really good. except if the element is near the ends
		if (comp(val, *it))
		{
			it = content.emplace(it, std::move(val));
			success = true;
		}
		return std::make_pair(it, success);
	}

	template <class... Args>
	std::pair<iterator, bool> try_emplace(const key_type &k, Args &&... args)
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
	std::pair<iterator, bool> try_emplace(key_type &&k, Args &&... args)
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
	std::pair<iterator, bool> try_emplace(const_iterator hint, const key_type &k, Args &&... args)
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
	std::pair<iterator, bool> try_emplace(const_iterator hint, key_type &&k, Args &&... args)
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

	iterator erase(const_iterator pos) { return iterator(content.erase(pos.base_reference())); }
	iterator erase(iterator pos) { return iterator(content.erase(pos.base_reference())); }
	iterator erase(const_iterator first, const_iterator last) { return iterator(content.erase(first.base_reference(), last.base_reference())); }
	size_type erase(const key_type &k)
	{
		auto it = std::lower_bound(begin(), end(), k, comp);
		if (comp(*it, k))
		{
			return 0;
		}
		content.erase(it);
		return 1;
	}
	void swap(flat_map &other) noexcept
	{
		using std::swap;
		swap(content, other.content);
		swap(comp, other.comp);
	}
	template <class C2, class Collision>
	void merge(const flat_map<key_type, mapped_type, C2, allocator_type> &source, Collision &&collision)
	{
		merge(source, std::forward<Collision>(collision), [](auto &&) {});
	}
	template <class C2, class Collision, class NoCollision>
	void merge(const flat_map<key_type, mapped_type, C2, allocator_type> &source, Collision &&collision,
	           NoCollision &&nocollision)
	{
		insert(source.begin(), source.end(), std::forward<Collision>(collision),
		       std::forward<NoCollision>(nocollision));
	}

	template <class C2, class Collision>
	void merge(flat_map<key_type, mapped_type, C2, allocator_type> &&source, Collision &&collision)
	{
		merge(std::move(source), std::forward<Collision>(collision), [](auto &&) {});
	}
	template <class C2, class Collision, class NoCollision>
	void merge(flat_map<key_type, mapped_type, C2, allocator_type> &&source, Collision &&collision,
	           NoCollision &&nocollision)
	{
		if (source.capacity() >= source.size() + size()) // reuse source's resources if it avoid doing an allocation.
		{
			std::swap(content, source.content); // if the source has enough capacity, we'll keep it instead.
			insert(
			    source.begin(), source.end(),
			    [collision](auto &a, auto &b) {
				    // b is const in the context that call this lambda function.
				    // But in this present context, we know it's actually a reference to a non-const value.
				    // Casting away const in order to change a value should almost never be used.
				    auto &c = const_cast<std::remove_const_t<std::remove_reference_t<decltype(b)>> &>(b);
				    std::swap(a, c);
				    collision(a, c);
			    },
			    std::forward<NoCollision>(nocollision));
		}
		else
		{
			insert(source.begin(), source.end(), std::forward<Collision>(collision),
			       std::forward<NoCollision>(nocollision));
		}
	}
	template <class C2>
	void merge(const flat_map<key_type, mapped_type, C2, allocator_type> &source)
	{
		insert(source.begin(), source.end());
	}

	template <class C2>
	void merge(flat_map<key_type, mapped_type, C2, allocator_type> &&source)
	{
		if (source.capacity() >= source.size() + size())
		{
			swap(content, source.content); // if the source has enough capacity, we'll keep it instead.
			insert(source.begin(), source.end(), [](auto &a, const auto &b) {
				// b is const in the context that call this lambda function.
				// But in this present context, we know it's actually a reference to a non-const value.
				// Casting away const in order to change a value should almost never be used.
				auto &c = const_cast<std::remove_const_t<std::remove_reference_t<decltype(b)>> &>(b);
				using std::swap;
				swap(a, c);
			});
		}
		else
		{
			insert(source.begin(), source.end());
		}
	}

	iterator find(const key_type &k)
	{
		auto it = std::lower_bound(begin(), end(), k, comp);
		if (comp(*it, k))
			return end();
		return it;
	}
	const_iterator find(const key_type &k) const { return const_cast<flat_map *>(this)->find(k); }
	size_type count(const key_type &k) const { return find(k) != end(); }
	bool contains(const key_type &k) const { return count(k); }
	std::pair<iterator, iterator> equal_range(const key_type &k)
	{
		auto it = find(k);
		return std::make_pair(it, it == end() ? end() : it + 1);
	}
	std::pair<const_iterator, const_iterator> equal_range(const key_type &k) const
	{
		return std::make_from_tuple<std::pair<const_iterator, const_iterator>>(
		    const_cast<flat_map *>(this)->equal_range(k));
	}
	iterator lower_bound(const key_type &k)
	{
		auto it = std::lower_bound(begin(), end(), k, comp);
		if (comp(k, *it))
			it = end();
		return it;
	}
	const_iterator lower_bound(const key_type &k) const
	{
		auto it = std::lower_bound(cbegin(), cend(), k, comp);
		if (comp(k, *it))
			it = end();
		return it;
	}
	iterator upper_bound(const key_type &k)
	{
		auto it = std::upper_bound(begin(), end(), k, comp);
		if (comp(*it, k))
			it = end();
		return it;
	}
	const_iterator upper_bound(const key_type &k) const
	{
		auto it = std::upper_bound(begin(), end(), k, comp);
		if (comp(*it, k))
			it = end();
		return it;
	}

	key_compare key_comp() const { return comp.comp; }

	value_compare value_comp() const { return comp; }

	/**
	 * @brief sort the flat_map. So long as you only manipulate the content of the map using it's methods, you do not
	 * need to reorder it.
	 *
	 *  for the block tensor class, some manipulation of the index could be necessary.
	 *  If the transformation on the index changes the ordering, call sort afterward. Or face the consequences.
	 *  All the method of this class are UB if it is not sorted.
	 *
	 * @param first iterator to the first element of the range to sort
	 * @param last iterator to the element after the range to sort
	 *
	 */
	void sort(iterator first, iterator last) { std::sort(first, last, comp); }
	void sort() { std::sort(begin(), end(), comp); }

  protected:
	value_compare comp;

  private:
	content_t content;
	void sort(typename content_t::iterator first, typename content_t::iterator last) { std::sort(first, last, comp); }
	// void remove_collisions(iterator first = begin(), iterator last = end())
	// { //assume a sorted array
	// 	auto beg = first while (first != last)
	// 	{
	// 	}
	// }
	std::pair<const_iterator, const_iterator> use_hint(const_iterator hint,const key_type &key) const
	{
		const_iterator first = comp(*hint, key) ? hint : cbegin();
		const_iterator last = comp(key, *hint) ? cend() : hint + 1;
		return std::make_pair(first, last);
	}
	std::pair<iterator, iterator> use_hint(iterator hint,const key_type &key)
	{
		iterator first = comp(*hint, key) ? hint : begin();
		iterator last = comp(key, *hint) ? hint + 1 : end();
		return std::make_pair(first, last);
	}
	typename content_t::iterator filter_unique(typename content_t::iterator first, typename content_t::iterator last)
	{
		return filter_unique(first, last, [](auto &, auto &) {});
	}
	template <class Collision>
	typename content_t::iterator filter_unique(typename content_t::iterator first, typename content_t::iterator last,
	                                           Collision &&collision)
	{ // no collision get applied to all elements, instead of just the new element inserted.
		using std::swap;
		auto l = first;
		auto second = std::upper_bound(first, last, *l, comp);
		while (first != last)
		{
			std::for_each(first + 1, second, [collision, l](auto &x) { collision(*l, x); });
			++l;
			if (second == last)
				break;
			swap(*l, *second); // when there's only one element (i.e. no collision) of a given value it swaps with self.
			first = second;
			second = std::upper_bound(first, last, *l, comp);
		}
		return l;
	}
};

template <class key, class val, class C1, class C2, class alloc1, class alloc2, template <class...> class array1,
          template <class...> class array2>
bool operator==(const flat_map<key, val, C1, alloc1, array1> &a, const flat_map<key, val, C2, alloc2, array2> &b)
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
	{
		auto pre_ordered = flat_map<int, double>(
		    {{10, 1.4},
		     {20, 1.2},
		     {30, 1.3},
		     {40, 1.1}}); // allocation of the pre-ordered flatmap trigger a report from valgrind (invalid read)
		qtt_REQUIRE(a == pre_ordered);
	}

	// qtt_CHECK(false == true);
	qtt_SUBCASE("insert a flat_map before, in and at the end, with collisions")
	{
		int collisions = 0;
		flat_map<int, double> b{{1, 6e10}, {10, 6e10}, {50, 1e10}, {30, 1}, {33, 1e12}, {70, 1e11}};
		flat_map<int, double> result{{1, 6e10},  {10, 1.4}, {20, 1.2},  {30, 1.3},
		                             {33, 1e12}, {40, 1.1}, {50, 1e10}, {70, 1e11}};
		a.insert(b.begin(), b.end(), [&collisions](auto &&, auto &&) { ++collisions; });
		qtt_CHECK(a == result);
		qtt_CHECK(collisions == 2);
	}
	qtt_SUBCASE("insert a flat_map in middle and at the end, with collisions")
	{
		int collisions = 0;
		flat_map<int, double> b{{10, 6e10}, {50, 1e10}, {33, 1e12}, {40, 5.5}, {70, 1e11}};
		flat_map<int, double> result{{10, 1.4}, {20, 1.2}, {30, 1.3}, {33, 1e12}, {40, 1.1}, {50, 1e10}, {70, 1e11}};
		a.insert(b.begin(), b.end(), [&collisions](auto &&, auto &&) { ++collisions; });
		qtt_CHECK(a == result);
		qtt_CHECK(collisions == 2);
	}
	qtt_SUBCASE("insert a flat_map in middle, with collisions")
	{
		int collisions = 0;
		flat_map<int, double> b{{10, 6e10}, {33, 1e12}, {31, 1e12}, {40, 5.5}};
		flat_map<int, double> result{{10, 1.4}, {20, 1.2}, {31, 1e12}, {30, 1.3}, {33, 1e12}, {40, 1.1}};
		a.insert(b.begin(), b.end(), [&collisions](auto &&, auto &&) { ++collisions; });
		qtt_CHECK(a == result);
		qtt_CHECK(collisions == 2);
	}
	qtt_SUBCASE("insert a flat_map before, in middle and at the end, no collisions")
	{
		int collisions = 0;
		flat_map<int, double> b{{1, 6e10}, {50, 1e10}, {33, 1e12}, {70, 1e11}};
		flat_map<int, double> result{{1, 6e10},  {10, 1.4}, {20, 1.2},  {30, 1.3},
		                             {33, 1e12}, {40, 1.1}, {50, 1e10}, {70, 1e11}};
		a.insert(b.begin(), b.end(), [&collisions](auto &&, auto &&) { ++collisions; });
		qtt_CHECK(a == result);
		qtt_CHECK(collisions == 0);
	}
	qtt_SUBCASE("insert a flat_map in middle and at the end, no collisions")
	{
		int collisions = 0;
		flat_map<int, double> b{{50, 1e10}, {33, 1e12}, {70, 1e11}};
		flat_map<int, double> result{{10, 1.4}, {20, 1.2}, {30, 1.3}, {33, 1e12}, {40, 1.1}, {50, 1e10}, {70, 1e11}};
		a.insert(b.begin(), b.end(), [&collisions](auto &&, auto &&) { ++collisions; });
		qtt_CHECK(a == result);
		qtt_CHECK(collisions == 0);
	}
	qtt_SUBCASE("insert a flat_map in middle, no collisions")
	{
		int collisions = 0;
		flat_map<int, double> b{{33, 1e12}, {31, 1e12}};
		flat_map<int, double> result{{10, 1.4}, {20, 1.2}, {31, 1e12}, {30, 1.3}, {33, 1e12}, {40, 1.1}};
		a.insert(b.begin(), b.end(), [&collisions](auto &&, auto &&) { ++collisions; });
		qtt_CHECK(a == result);
		qtt_CHECK(collisions == 0);
	}
	qtt_SUBCASE("insert unsorted sequence")
	{
		int collisions = 0;
		std::vector<std::pair<int, double>> b{{1, 6e10},  {1, 6e10}, {10, 6e10}, {50, 1e10}, {50, 1e10},
		                                      {50, 1e10}, {30, 1},   {33, 1e12}, {70, 1e11}};
		flat_map<int, double> result{{10, 1.4}, {20, 1.2}, {30, 1.3},  {33, 1e12},
		                             {40, 1.1}, {1, 6e10}, {50, 1e10}, {70, 1e11}};
		qtt_REQUIRE(!std::is_convertible_v<decltype(b)::iterator, decltype(a)::const_iterator>);
		// fmt::print("a pre {}\n",a);
		a.insert(b.begin(), b.end(), [&collisions](auto &&, auto &&) { ++collisions; });
		qtt_CHECK(a == result);
		// fmt::print("a post {}\n",a);
		// fmt::print("result {}\n",result);
		qtt_CHECK(collisions == 5);
	}
	qtt_SUBCASE("merge sequences, collision moment test")
	{
		flat_map<int, double> a{{10, 100}, {20, 100}, {30, 100}, {33, 100}, {40, 100}, {1, 100}, {50, 100}, {70, 100}};
		flat_map<int, double> b{{10, 20}, {20, 20}, {30, 20}, {33, 20}, {40, 20}, {1, 20}, {50, 20}, {70, 20}};
		flat_map<int, double> result{{10, 120}, {20, 120}, {30, 120}, {33, 120},
		                             {40, 120}, {1, 120},  {50, 120}, {70, 120}};
		a.merge(b, [](auto &&x, auto &&y) { x += y; });
		qtt_CHECK(a == result);
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
