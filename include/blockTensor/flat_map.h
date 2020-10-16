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

#include <algorithm>
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

private:
	using content_t = Array<std::pair<Key, Value>, Allocator>;

public:
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
	using iterator = typename content_t::iterator;             //require random access.
	using const_iterator = typename content_t::const_iterator; //require random access.
	using reverse_iterator = typename content_t::reverse_iterator;
	using const_reverse_iterator = typename content_t::const_reverse_iterator;
	class value_compare
	{
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
	flat_map() : comp(), content(){};
	explicit flat_map(size_type capacity, const key_compare& _comp = key_compare(),
	                  const allocator& _alloc = allocator_type())
	    : comp(_comp), content(capacity, _alloc) { content.resize(0); }
	explicit flat_map(const key_compare& _comp, const allocator_type& _alloc = allocator_type())
	    : comp(_comp), content(_alloc) {}
	explicit flat_map(allocator_type& _alloc = allocator_type())
	    : comp(), content(_alloc) {}
	template <class InputIt>
	flat_map(InputIt first, InputIt last, const key_compare& _comp = key_compare(),
	         const allocator_type& _alloc = allocator_type())
	    : comp(_comp), content(first, last, _alloc) {}
	template <class InputIt>
	flat_map(InputIt first, InputIt last, const allocator_type& _alloc)
	    : comp(), content(first, last, _alloc) { sort(); }
	flat_map(const flat_map& other) : comp(other.comp), content(other.content) {}
	flat_map(const flat_map& other, const allocator_type& alloc) : comp(other.comp), content(other.content, alloc) {}
	flat_map(flat_map&& other) : comp(std::move(other.comp)), content(std::move(other.content)) {}
	flat_map(flat_map&& other, const allocator_type& alloc) : comp(std::move(other.comp)), content(std::move(other.content), alloc) {}
	flat_map(std::initializer_list<value_type> init, const value_compare& _comp = value_compare(),
	         const allocator_type& alloc = allocator_type()) : comp(_comp), content(init, alloc){sort()} flat_map(std::initializer_list<value_type> init, const allocator_type& alloc)
	    : comp(), content(init, alloc){sort()}

	                  //assigment
	                  flat_map
	                  & operator=(const flat_map& other)
	{
		content = other.content;
		comp = other.comp;
	}
	flat_map& operator=(flat_map&& other) noexcept
	{
		content = std::move(other.content);
		comp = std::move(other.comp);
	}
	flat_map& operator=(std::initializer_list<value_type> ilist)
	{
		content = ilist;
		sort();
	}

	allocator_type get_allocator() const noexcept
	{
		return content.get_allocator();
	}
	//element access
	mapped_type& at(const key_type& key)
	{
		auto it = std::lower_bound(content.begin(), content.end(), key, comp);
		if (it == content.end() or it->first != key)
			throw std::out_of_range("key absent from sorted array.");
		return it->second;
	}
	const mapped_type& at(const key_type& key) const
	{
		return const_cast<flat_map*>(this)->at(key);
	}
	mapped_type& operator[](const key_type key)
	{
		auto it = std::lower_bound(content.begin(), content.end(), key, comp);
		if (it == content.end() or it->first != key)
			it = content.insert(it, value_type());
		return it->second;
		//no const version of this one, because it inserts an element if the key isn't already present.
	}
	//iterators
	iterator begin()
	{
		return content.begin();
	}
	iterator end()
	{
		return content.end();
	}
	const_iterator cbegin() const
	{
		return content.cbegin();
	}
	const_iterator begin() const
	{
		return cbegin();
	}
	const_iterator cend() const
	{
		return content.cend();
	}
	const_iterator end() const
	{
		return cend();
	}
	reverse_iterator rbegin()
	{
		return content.rbegin();
	}
	reverse_iterator rend()
	{
		return content.rend();
	}
	const_reverse_iterator crbegin() const
	{
		return content.crbegin();
	}
	const_reverse_iterator rbegin() const
	{
		return crbegin();
	}
	const_reverse_iterator crend() const
	{
		return content.crend();
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
		if (it->first == value.first)
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
		if (it->first == value.first)
		{
			it = content.insert(it, std::move(value));
			success = true;
		}
		return std::make_pair(it, success);
	}
	template <class P, class = std::enable_if_t<std::is_constructible_v<value_type, P&&>>>
	std::pair<iterator, bool> instert(P&& value)
	{
		static_assert(std::is_constructible_v<value_type, P&&>, "Don't try to bypass the enable if");
		return insert(value_type(value));
	}
	iterator insert(const_iterator hint, const value_type& value)
	{
		int hint_bool = comp(*hint, value);
		iterator first = hint_bool? hint.base() : begin();
		iterator last = hint_bool? end() : hint;
		iterator it = std::lower_bound(first, last, value, comp); //that's not a big saving if the hint is really good. except if the element is near the ends
		if (it->first == value.first)
		{
			it = content.insert(it, value);
		}
		return it;
	}
	iterator insert(const_iterator hint, value_type&& value)
	{
		int hint_bool = comp(*hint, value);
		iterator first = hint_bool? hint.base() : begin();
		iterator last = hint_bool? end() : hint;
		iterator it = std::lower_bound(first, last, value, comp); //that's not a big saving if the hint is really good. except if the element is near the ends
		if (it->first == value.first)
		{
			it = content.insert(it, std::move(value));
		}
		return it;
	}
	template <class P, class = std::enable_if_t<std::is_constructible_v<value_type, P&&>>>
	iterator insert(const_iterator hint, P&& value)
	{
		static_assert(std::is_constructible_v<value_type, P&&>, "Don't try to bypass the enable if");
		return insert(hint,value_type(value));
	}
	template<class InputIt, class Collision, class = std::enable_if_t<!std::is_convertible_v<InputIt,const_iterator> > >
	void insert(InputIt first, InputIt last, Collision&& collision = [](auto&a,const auto&b){})
	{
		static_assert(!std::is_convertible_v<InputIt,const_iterator>,"Don't try to bypass the enable if");
		auto s2 = std::distance(first,last);
		auto s1 = size();
		content.insert(end(),first,last);
		sort(end()-s2,end());//sort the new stuff.
		auto i = end() - s2;
		while( i != begin() and s2 >0)
		{	
			i = std::lower_bound(begin(),i,content[s1+s2],comp);
			if (i -> first == content[s1+s2].first)
			{
				collision(i->second,content[s1+s2].second);//do something about the collision, nothing by default;will be useful to do the addition as an insertion.
				std::swap(content[s1+s2],*end());//move the colliding element of the inserted span to the end
				content.pop_back();//move the end back by one, making the colliding element out of scope.
			}
			--s2;
		}
		//find the repeats. should be feasable in O(n)
		sort();//sort everything.
	}
	//specialisation for the case of a inserting an ordered array, we can take some small shortcut in that case.
	template<class Collision>
	void insert(const_iterator first, const_iterator last, Collision&& collision = [](auto&a, const auto&b){})
	{
		auto n1 = std::distance(first,last);
		auto look = begin();
		//detect collisions between content and the input iterators.
		for(auto i = first; i!= last;++i)
		{
			look = std::lower_bound(look,end(),*i,comp);
			if (look == end()) break;
			if (look->first == i->first)
			{
				--n1;
				collision(look->second,i->second);
			}
		}
		content.resize(content.size()+n1);
		look = end() - n1;
		for(auto i = last; i!= first;--i)
		{
			auto new_look = std::lower_bound(begin(),look,*i,comp);
			if (new_look->first != i->first)
			{
				std::copy_backward(new_look,new_look+n1,look+n1);//move everything greater than what we have to insert to the end
				--new_look;//move new_look to be at the last element of the left array, instead of just past it.
				auto j = std::lower_bound(first,i,*new_look,comp);//find out how many element we can insert in the space we just created.
				size_t collision_at_j = (j->first == new_look->first ); //check for collision on this new edge
				std::copy_backward(j+collision_at_j,i,new_look+n1+1);//copy the stuff from the inserting array that is greater than what we have to the left.
				n1 -= i-j+collision_at_j;//update the number of element that remain to copy.
				i = j-1;// we have copied everything we had to in [j,i]
			}
			look = new_look+1;
		}//will require thorough testing.
	}

protected:
	value_compare comp;

	void sort(iterator first = content.begin(), iterator last = content.end())
	{
		std::sort(first, last, comp);
	}

private:
	content_t content;
};
} // namespace quantt
#endif /* D7E9786D_BD4E_41BF_A6C5_4E902E127A7D */
