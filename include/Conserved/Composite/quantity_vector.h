/*
 * File: any_quantity_vector.h
 * Project: quantt
 * File Created: Friday, 18th September 2020 4:07:41 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Friday, 18th September 2020 4:07:41 pm
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#ifndef D34E37A6_732F_4F45_9171_6B931CC1F812
#define D34E37A6_732F_4F45_9171_6B931CC1F812

#include "Conserved/Composite/quantity_vector_impl.h"
#include "Conserved/Composite/quantity.h"
#include "Conserved/quantity.h"
#include <vector>

#include "doctest/doctest_proxy.h"

namespace quantt
{

class any_quantity_vector final
{
	std::unique_ptr<vquantity_vector> ptr;

public:
	using iterator = vquantity_vector::iterator;
	using const_iterator = vquantity_vector::const_iterator;
	using reverse_iterator = std::reverse_iterator<iterator>;
	using const_reverse_iterator = std::reverse_iterator<const_iterator>;
	using value_type = any_quantity;
	using size_type = size_t;
	using difference_type = std::ptrdiff_t;
	using reference = any_quantity_ref;
	using const_reference = any_quantity_cref;
	using pointer = typename iterator::pointer;
	using const_pointer = typename const_iterator::pointer;

	any_quantity_vector(std::unique_ptr<vquantity_vector>&& _ptr) : ptr(std::move(_ptr)) {}
	any_quantity_vector() = default;
	any_quantity_vector(const any_quantity_vector& other) : ptr(other.ptr->clone()) {}
	any_quantity_vector(any_quantity_vector&& other) : ptr(std::move(other.ptr)) {}
	any_quantity_vector(size_t cnt, any_quantity_cref val) : ptr(val.get().make_vector(cnt)) {}
	/**
	 * @brief Construct a new any_quantity vector object from an initializer list of any_quantity
	 * @param list all the any_quantity must have the same concrete type, or an exception will be raised
	 */
	any_quantity_vector(std::initializer_list<any_quantity> list) : ptr(list.begin()->get().make_vector(0))
	{
		ptr->reserve(list.end() - list.begin());
		for (const_reference a : list)
		{
			push_back(a);
		}
	}
	any_quantity_vector(const vquantity_vector& other) : ptr(other.clone()) {}
	template <class Conc_cgroup, class = std::enable_if_t<is_conc_cgroup_impl<Conc_cgroup>::value>>
	any_quantity_vector(std::initializer_list<Conc_cgroup> list) : ptr(std::make_unique<quantity_vector<Conc_cgroup>>(list)) {}
	template <class Conc_cgroup, class Allocator, class = std::enable_if_t<is_conc_cgroup_impl<Conc_cgroup>::value>>
	any_quantity_vector(const quantity_vector<Conc_cgroup, Allocator>& other) : ptr(other.clone()) {}
	template <class Conc_cgroup, class Allocator, class = std::enable_if_t<is_conc_cgroup_impl<Conc_cgroup>::value>>
	any_quantity_vector(quantity_vector<Conc_cgroup, Allocator>&& other) : ptr(std::make_unique<quantity_vector<Conc_cgroup, Allocator>>())
	{
		*ptr = std::move(other);
	}

	any_quantity_vector(std::initializer_list<any_quantity_cref> list) : ptr(list.begin()->get().make_vector(0))
	{
		ptr->reserve(list.end() - list.begin());
		for (const_reference a : list)
		{
			push_back(a);
		}
	}
	any_quantity_vector(std::initializer_list<any_quantity_ref> list) : ptr(list.begin()->get().make_vector(0))
	{
		ptr->reserve(list.end() - list.begin());
		for (const_reference a : list)
		{
			push_back(a);
		}
	}

	any_quantity_vector& operator=(any_quantity_vector other)
	{
		ptr = std::move(other.ptr);
		return *this;
	}

	reference operator[](size_t n)
	{
		return any_quantity_ref((*ptr)[n]);
	}
	const_reference operator[](size_t n) const
	{
		return (*ptr)[n];
	}
	reference at(size_t n)
	{
		return ptr->at(n);
	}
	const_reference at(size_t n) const
	{
		return ptr->at(n);
	}

	reference front()
	{
		return ptr->front();
	}
	const_reference front() const
	{
		return ptr->front();
	}
	reference back()
	{
		return ptr->back();
	}
	const_reference back() const
	{
		return ptr->back();
	}
	pointer data()
	{
		return (ptr->data());
	}
	const_pointer data() const
	{
		return (ptr->data());
	}
	// capacity
	[[nodiscard]] bool empty() const
	{
		return ptr->empty();
	}
	[[nodiscard]] size_t size() const
	{
		return ptr->size();
	}
	[[nodiscard]] size_t max_size() const
	{
		return ptr->max_size();
	}
	void reserve(size_t n)
	{
		ptr->reserve(n);
	}
	[[nodiscard]] size_t capacity() const
	{
		return ptr->capacity();
	}
	void shrink_to_fit()
	{
		ptr->shrink_to_fit();
	}
	//modifiers
	void clear()
	{
		ptr->clear();
	}
	iterator insert(const_iterator pos, const_reference Val)
	{
		return ptr->insert(pos, Val.get());
	}
	iterator insert(const_iterator pos, size_t count, const_reference Val)
	{
		return ptr->insert(pos, count, Val.get());
	}
	iterator insert(const_iterator pos, const_iterator first, const_iterator last)
	{
		return ptr->insert(pos, first, last);
	}
	iterator insert(const_iterator pos, const_reverse_iterator first, const_reverse_iterator last)
	{
		return ptr->insert(pos, first, last);
	}
	iterator erase(const_iterator pos)
	{
		return ptr->erase(pos);
	}
	iterator erase(const_iterator first, const_iterator last)
	{
		return ptr->erase(first, last);
	}
	void push_back(const_reference value)
	{
		ptr->push_back(value.get());
	}
	void pop_back()
	{
		ptr->pop_back();
	}
	void resize(size_t count)
	{
		ptr->resize(count);
	}
	void resize(size_t count, const_reference val)
	{
		ptr->resize(count, val.get());
	}
	void swap(any_quantity_vector& other)
	{
		using std::swap;
		swap(ptr, other.ptr);
	}
	iterator begin()
	{
		return ptr->begin();
	}
	iterator end()
	{
		return ptr->end();
	}
	const_iterator cbegin() const
	{
		return ptr->cbegin();
	}
	const_iterator cend() const
	{
		return ptr->cend();
	}

	reverse_iterator rbegin()
	{
		return ptr->rbegin();
	}
	reverse_iterator rend()
	{
		return ptr->rend();
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
		return ptr->crbegin();
	}
	const_reverse_iterator crend() const
	{
		return ptr->crend();
	}
	const_reverse_iterator rbegin() const
	{
		return (crbegin());
	}
	const_reverse_iterator rend() const
	{
		return (crend());
	}

	any_quantity_vector permute(const int64_t* permute_begin, const int64_t* permute_end, const std::vector<int64_t> repetition) const
	{
		return any_quantity_vector(ptr->virtual_permute(permute_begin, permute_end, repetition));
	}
};

inline void swap(any_quantity_vector& a, any_quantity_vector& b)
{
	a.swap(b);
}

qtt_TEST_CASE("concrete any_quantity container implementation")
{
	using ccgroup = quantity<conserved::C<2>, conserved::Z>;
	quantity_vector<ccgroup> t1{ccgroup(1, 1), ccgroup(0, 0), ccgroup(1, -1)};
	const quantity_vector<ccgroup> t2{ccgroup(1, 1), ccgroup(0, 0), ccgroup(1, -1)};
	qtt_REQUIRE(t1.size() > 0);
	qtt_REQUIRE(t2.size() > 0);
	qtt_SUBCASE("Access operator[]")
	{
		auto& a = t1[0];
		auto& b = t2[0];
		qtt_CHECK(std::is_same_v<decltype(a), ccgroup&>);
		qtt_CHECK(std::is_same_v<decltype(b), const ccgroup&>);
	}
	qtt_SUBCASE("Access method at")
	{
		qtt_REQUIRE_NOTHROW(t1.at(0));
		qtt_REQUIRE_NOTHROW(t2.at(0));
		auto& a = t1.at(0);
		auto& b = t2.at(0);
		qtt_CHECK(std::is_same_v<decltype(a), ccgroup&>);
		qtt_CHECK(std::is_same_v<decltype(b), const ccgroup&>);
		qtt_CHECK_THROWS_AS(t2.at(t2.size()), std::out_of_range);
		qtt_CHECK_THROWS_AS(t1.at(t1.size()), std::out_of_range);
	}
	qtt_SUBCASE("front and back and data")
	{
		auto& f1 = t1.front();
		auto& b1 = t1.back();
		auto& f2 = t2.front();
		auto& b2 = t2.back();
		auto d1 = t1.data();
		auto d2 = t2.data();
		qtt_CHECK(std::is_same_v<decltype(f1), ccgroup&>);
		qtt_CHECK(std::is_same_v<decltype(b1), ccgroup&>);
		qtt_CHECK(std::is_same_v<decltype(f2), const ccgroup&>);
		qtt_CHECK(std::is_same_v<decltype(b2), const ccgroup&>);
		qtt_CHECK(std::is_same_v<decltype(d1), ccgroup*>);
		qtt_CHECK(std::is_same_v<decltype(d2), const ccgroup*>);
	}
	qtt_SUBCASE("capacity")
	{
		auto s = t1.size();
		auto c = t1.capacity();
		qtt_CHECK_FALSE(t1.empty());
		qtt_CHECK(t1.max_size() > 1000);
		t1.reserve(c + 5);
		qtt_CHECK(t1.capacity() >= (c + 5));
		qtt_CHECK(t1.size() == s);
		t1.shrink_to_fit();
		qtt_CHECK(t1.capacity() >= s);
		qtt_WARN_MESSAGE(t1.capacity() == s, "shrink_to_fit doesn't fully shrink the vector. This behavior is allowed by the standard for std::vector");
	}
	qtt_SUBCASE("modifiers")
	{
		t1.emplace(t1.cbegin(), 3, -2);
		qtt_CHECK(t1[0] == ccgroup(3, -2));
		t1.emplace_back(0, 10);
		qtt_CHECK(t1[t1.size() - 1] == ccgroup(0, 10));
		auto s = t1.size();
		t1.pop_back();
		qtt_CHECK(t1.size() == (s - 1));
		auto c = t1.capacity();
		t1.clear();
		qtt_CHECK(t1.capacity() == c);
		qtt_CHECK(t1.size() == 0);
		quantity_vector<ccgroup> t3{ccgroup(1, 1), ccgroup(0, 0), ccgroup(1, -1)};
		auto data1 = t1.data();
		auto data3 = t3.data();
		swap(t1, t3);
		qtt_CHECK(data1 == t3.data());
		qtt_CHECK(data3 == t1.data());
	}
}
qtt_TEST_CASE("polymorphic any_quantity container with value semantic")
{
	using ccgroup = quantity<conserved::C<2>, conserved::Z>;
	quantity_vector<ccgroup> conc_t1{ccgroup(1, 1), ccgroup(0, 0), ccgroup(1, -1)};
	const quantity_vector<ccgroup> conc_t2{ccgroup(1, 1), ccgroup(0, 0), ccgroup(1, -1)};
	any_quantity_vector t1{any_quantity(conserved::C<2>(1), conserved::Z(1)), any_quantity(ccgroup(0, 0)), any_quantity(ccgroup(1, -1))};
	const any_quantity_vector t2{any_quantity(conserved::C<2>(1), conserved::Z(1)), any_quantity(ccgroup(0, 0)), any_quantity(ccgroup(1, -1))};

	qtt_REQUIRE(t1.size() > 0);
	qtt_REQUIRE(t2.size() > 0);
	qtt_SUBCASE("Access operator[]")
	{
		auto a = t1[0]; //makes a reference
		auto b = t2[0]; //makes a reference
		//if you want a copy, you have to be specific about the return type.
		// any_quantity c = t1[1]; //make a copy of the value.
		qtt_CHECK(std::is_same_v<decltype(a), any_quantity_ref>);
		qtt_CHECK(std::is_same_v<decltype(b), any_quantity_cref>);
	}
	qtt_SUBCASE("Access method at")
	{
		qtt_REQUIRE_NOTHROW(t1.at(0));
		qtt_REQUIRE_NOTHROW(t2.at(0));
		auto a = t1.at(0); //makes a reference
		auto b = t2.at(0); //makes a reference
		qtt_CHECK(std::is_same_v<decltype(a), any_quantity_ref>);
		qtt_CHECK(std::is_same_v<decltype(b), any_quantity_cref>);
		qtt_CHECK_THROWS_AS(t2.at(t2.size()), std::out_of_range);
		qtt_CHECK_THROWS_AS(t1.at(t1.size()), std::out_of_range);
	}
	qtt_SUBCASE("front and back and data")
	{
		auto f1 = t1.front(); //makes a reference
		auto b1 = t1.back();  //makes a reference
		auto f2 = t2.front(); //makes a reference
		auto b2 = t2.back();  //makes a reference
		auto d1 = t1.data();  //makes a reference
		auto d2 = t2.data();  //makes a reference
		qtt_CHECK(std::is_same_v<decltype(f1), any_quantity_ref>);
		qtt_CHECK(std::is_same_v<decltype(b1), any_quantity_ref>);
		qtt_CHECK(std::is_same_v<decltype(f2), any_quantity_cref>);
		qtt_CHECK(std::is_same_v<decltype(b2), any_quantity_cref>);
		qtt_CHECK(std::is_same_v<decltype(d1), typename any_quantity_vector::pointer>);
		qtt_CHECK(std::is_same_v<decltype(d2), typename any_quantity_vector::const_pointer>);
	}
	qtt_SUBCASE("capacity")
	{
		auto s = t1.size();
		auto c = t1.capacity();
		qtt_CHECK_FALSE(t1.empty());
		qtt_CHECK(t1.max_size() > 1000);
		t1.reserve(c + 5);
		qtt_CHECK(t1.capacity() >= (c + 5));
		qtt_CHECK(t1.size() == s);
		t1.shrink_to_fit();
		qtt_CHECK(t1.capacity() >= s);
		qtt_WARN_MESSAGE(t1.capacity() == s, "shrink_to_fit doesn't fully (or at all) shrink the vector. This behavior is allowed by the standard for std::vector");
	}
	qtt_SUBCASE("modifiers")
	{
		auto s = t1.size();
		t1.pop_back();
		qtt_CHECK(t1.size() == (s - 1));
		auto c = t1.capacity();
		t1.clear();
		qtt_CHECK(t1.capacity() == c);
		qtt_CHECK(t1.size() == 0);
		t1.push_back(ccgroup(1,1));
		qtt_CHECK_NOTHROW(t1.push_back(ccgroup(1, 1)));
		qtt_CHECK_THROWS_AS(t1.push_back(any_quantity(conserved::C<2>(1), conserved::C<4>(1))), std::bad_cast);
		any_quantity_vector t3{any_quantity(ccgroup(1, 1)), any_quantity(ccgroup(0, 0)), any_quantity(ccgroup(1, -1))};
		qtt_SUBCASE("insert, pop_back and erase")
		{
			auto it = t1.begin();
			qtt_CHECK_THROWS_AS(t1.insert(t1.begin(), any_quantity(conserved::C<5>(0))), std::bad_cast); // the any_quantity has the wrong concrete type for the container.
			qtt_CHECK_NOTHROW(it = t1.insert(t1.begin(), any_quantity(ccgroup(1, -1))));
			qtt_CHECK(t1[0] == any_quantity(ccgroup(1, -1))); //The insertion happen *before* the position of the given iterator. In this case the inserted value should become the 0th element
			qtt_CHECK(it == t1.begin());                      //the return iterator points to the first (or only) element inserted.
			//current implementation incorrect because the end() iterator does not point to a valid object... iterator comparator needs to be implemented in a way that doesn't rely on the vtable.
			qtt_CHECK_NOTHROW(it = t1.insert(t1.end(), 3, ccgroup(1, 5))); //insert 3 ccgroup(1,5) before the end;
			for (; it != t1.end();++it)
			{
				qtt_CHECK(*it == ccgroup(1, 5));
			}
			any_quantity_vector t4{any_quantity(conserved::C<2>(1)), conserved::C<2>(0), conserved::C<2>(-1)};
			qtt_CHECK_THROWS_AS(it = t1.insert(t1.begin() + 2, t4.begin(), t4.end()), std::bad_cast); //t4 is incompatible with t1.
			qtt_CHECK_NOTHROW(it = t1.insert(t1.begin() + 2, t3.begin(), t3.end()));
			int i = 0;
			for(auto end = it+(t3.end()-t3.begin()); it != end;++it )
			{
				qtt_CHECK(*it == t3[i++]);
			}
			qtt_CHECK_NOTHROW(it = t1.insert(t1.begin() + 2, t3.rbegin(), t3.rend())); //insertion by reverse iterator.
			i = t3.size()-1;
			for(auto end = it+(t3.end()-t3.begin()); it != end;++it )
			{
				qtt_CHECK(*it == t3[i--]);
			}
			any_quantity last = *(t1.end() - 2);
			qtt_CHECK_NOTHROW(t1.erase(t1.end() - 1));
			qtt_CHECK(last == *(t1.end() - 1));
			auto size = t1.size();
			qtt_CHECK_NOTHROW(t1.erase(t1.begin(), t1.begin() + 3));
			qtt_CHECK(size - 3 == t1.size());
			i = t1.size();
			qtt_CHECK_NOTHROW(t1.pop_back());
			qtt_CHECK(i - 1 == t1.size());
		}
		auto size = t1.size();
		qtt_CHECK_NOTHROW(t1.resize(size + 2));
		qtt_CHECK(t1.back() == ccgroup());
		qtt_CHECK(t1.size() == size + 2);
		qtt_CHECK_NOTHROW(t1.resize(size));
		qtt_CHECK(size == t1.size());
		qtt_CHECK_NOTHROW(t1.resize(size + 2, ccgroup(1, 10)));
		qtt_CHECK(t1.back() == ccgroup(1, 10));
		qtt_CHECK(t1.size() == size + 2);
		auto data1 = t1.data();
		auto data3 = t3.data();
		swap(t1, t3);
		qtt_CHECK(data1 == t3.data());
		qtt_CHECK(data3 == t1.data());
	}
}



} // namespace quantt
#endif /* D34E37A6_732F_4F45_9171_6B931CC1F812 */
