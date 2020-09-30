/*
 * File: cgroup_vector.h
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

#include "composite_group.h"
#include "Groups/groups.h"
#include <vector>
#include "Groups/cgroup_container_impl.h"

#include "doctest/cond_doctest.h"

namespace quantt
{


class cgroup_vector final
{
	std::unique_ptr<cgroup_vector_impl> ptr;

public:
	using iterator = cgroup_iterator;
	using const_iterator = const_cgroup_iterator;
	using reverse_iterator = std::reverse_iterator<iterator>;
	using const_reverse_iterator = std::reverse_iterator<const_iterator>;
	using value_type = cgroup;
	using size_type = size_t;
	using difference_type = std::ptrdiff_t;
	using reference = cgroup_ref;
	using const_reference = cgroup_cref;
	using pointer = typename iterator::pointer;
	using const_pointer = typename const_iterator::pointer;

	cgroup_vector(std::unique_ptr<cgroup_vector_impl>&& _ptr) : ptr(std::move(_ptr)) {}
	cgroup_vector() = default;
	cgroup_vector(const cgroup_vector& other) : ptr(other.ptr->clone()) {}
	cgroup_vector(cgroup_vector&& other) : ptr(std::move(other.ptr)) {}
	cgroup_vector(size_t cnt, cgroup_cref val) : ptr(val.get().make_vector(cnt)) {}
	/**
	 * @brief Construct a new cgroup vector object from an initializer list of cgroup
	 * @param list all the cgroup must have the same concrete type, or an exception will be raised
	 */
	cgroup_vector(std::initializer_list<cgroup> list) : ptr(list.begin()->get().make_vector(0))
	{
		ptr->reserve(list.end() - list.begin());
		for (const_reference a : list)
		{
			push_back(a);
		}
	}
	cgroup_vector(const cgroup_vector_impl& other) : ptr(other.clone()) {}
	template <class Conc_cgroup, class = std::enable_if_t<is_conc_cgroup_impl<Conc_cgroup>::value>>
	cgroup_vector(std::initializer_list<Conc_cgroup> list) : ptr(std::make_unique<conc_cgroup_vector_impl<Conc_cgroup>>(list)) {}
	template <class Conc_cgroup, class Allocator, class = std::enable_if_t<is_conc_cgroup_impl<Conc_cgroup>::value>>
	cgroup_vector(const conc_cgroup_vector_impl<Conc_cgroup, Allocator>& other) : ptr(other.clone()) {}
	template <class Conc_cgroup, class Allocator, class = std::enable_if_t<is_conc_cgroup_impl<Conc_cgroup>::value>>
	cgroup_vector(conc_cgroup_vector_impl<Conc_cgroup, Allocator>&& other) : ptr(std::make_unique<conc_cgroup_vector_impl<Conc_cgroup, Allocator>>())
	{
		*ptr = std::move(other);
	}

	cgroup_vector(std::initializer_list<cgroup_cref> list) : ptr(list.begin()->get().make_vector(0))
	{
		ptr->reserve(list.end() - list.begin());
		for (const_reference a : list)
		{
			push_back(a);
		}
	}
	cgroup_vector(std::initializer_list<cgroup_ref> list) : ptr(list.begin()->get().make_vector(0))
	{
		ptr->reserve(list.end() - list.begin());
		for (const_reference a : list)
		{
			push_back(a);
		}
	}

	cgroup_vector& operator=(cgroup_vector other)
	{
		ptr = std::move(other.ptr);
		return *this;
	}

	reference operator[](size_t n)
	{
		return cgroup_ref((*ptr)[n]);
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
	void swap(cgroup_vector& other)
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
};
void swap(cgroup_vector& a, cgroup_vector& b)
{
	a.swap(b);
}


TEST_CASE("concrete cgroup container implementation")
{
	using ccgroup = conc_cgroup_impl<groups::C<2>, groups::Z>;
	conc_cgroup_vector_impl<ccgroup> t1{ccgroup(1, 1), ccgroup(0, 0), ccgroup(1, -1)};
	const conc_cgroup_vector_impl<ccgroup> t2{ccgroup(1, 1), ccgroup(0, 0), ccgroup(1, -1)};
	REQUIRE(t1.size() > 0);
	REQUIRE(t2.size() > 0);
	SUBCASE("Access operator[]")
	{
		auto& a = t1[0];
		auto& b = t2[0];
		CHECK(std::is_same_v<decltype(a), ccgroup&>);
		CHECK(std::is_same_v<decltype(b), const ccgroup&>);
	}
	SUBCASE("Access method at")
	{
		REQUIRE_NOTHROW(t1.at(0));
		REQUIRE_NOTHROW(t2.at(0));
		auto& a = t1.at(0);
		auto& b = t2.at(0);
		CHECK(std::is_same_v<decltype(a), ccgroup&>);
		CHECK(std::is_same_v<decltype(b), const ccgroup&>);
		CHECK_THROWS_AS(t2.at(t2.size()), std::out_of_range);
		CHECK_THROWS_AS(t1.at(t1.size()), std::out_of_range);
	}
	SUBCASE("front and back and data")
	{
		auto& f1 = t1.front();
		auto& b1 = t1.back();
		auto& f2 = t2.front();
		auto& b2 = t2.back();
		auto d1 = t1.data();
		auto d2 = t2.data();
		CHECK(std::is_same_v<decltype(f1), ccgroup&>);
		CHECK(std::is_same_v<decltype(b1), ccgroup&>);
		CHECK(std::is_same_v<decltype(f2), const ccgroup&>);
		CHECK(std::is_same_v<decltype(b2), const ccgroup&>);
		CHECK(std::is_same_v<decltype(d1), ccgroup*>);
		CHECK(std::is_same_v<decltype(d2), const ccgroup*>);
	}
	SUBCASE("capacity")
	{
		auto s = t1.size();
		auto c = t1.capacity();
		CHECK_FALSE(t1.empty());
		CHECK(t1.max_size() > 1000);
		t1.reserve(c + 5);
		CHECK(t1.capacity() >= (c + 5));
		CHECK(t1.size() == s);
		t1.shrink_to_fit();
		CHECK(t1.capacity() >= s);
		WARN_MESSAGE(t1.capacity() == s, "shrink_to_fit doesn't fully shrink the vector. This behavior is allowed by the standard for std::vector");
	}
	SUBCASE("modifiers")
	{
		t1.emplace(t1.cbegin(), 3, -2);
		CHECK(t1[0] == ccgroup(3, -2));
		t1.emplace_back(0, 10);
		CHECK(t1[t1.size() - 1] == ccgroup(0, 10));
		auto s = t1.size();
		t1.pop_back();
		CHECK(t1.size() == (s - 1));
		auto c = t1.capacity();
		t1.clear();
		CHECK(t1.capacity() == c);
		CHECK(t1.size() == 0);
		conc_cgroup_vector_impl<ccgroup> t3{ccgroup(1, 1), ccgroup(0, 0), ccgroup(1, -1)};
		auto data1 = t1.data();
		auto data3 = t3.data();
		swap(t1, t3);
		CHECK(data1 == t3.data());
		CHECK(data3 == t1.data());
	}
}
TEST_CASE("polymorphic cgroup container with value semantic")
{
	using ccgroup = conc_cgroup_impl<groups::C<2>, groups::Z>;
	conc_cgroup_vector_impl<ccgroup> conc_t1{ccgroup(1, 1), ccgroup(0, 0), ccgroup(1, -1)};
	const conc_cgroup_vector_impl<ccgroup> conc_t2{ccgroup(1, 1), ccgroup(0, 0), ccgroup(1, -1)};

	cgroup_vector t1{cgroup(groups::C<2>(1), groups::Z(1)), cgroup(ccgroup(0, 0)), cgroup(ccgroup(1, -1))};
	const cgroup_vector t2{cgroup(groups::C<2>(1), groups::Z(1)), cgroup(ccgroup(0, 0)), cgroup(ccgroup(1, -1))};

	REQUIRE(t1.size() > 0);
	REQUIRE(t2.size() > 0);
	SUBCASE("Access operator[]")
	{
		auto a = t1[0]; //makes a reference
		auto b = t2[0]; //makes a reference
		//if you want a copy, you have to be specific about the return type.
		// cgroup c = t1[1]; //make a copy of the value.
		CHECK(std::is_same_v<decltype(a), cgroup_ref>);
		CHECK(std::is_same_v<decltype(b), cgroup_cref>);
	}
	SUBCASE("Access method at")
	{
		REQUIRE_NOTHROW(t1.at(0));
		REQUIRE_NOTHROW(t2.at(0));
		auto a = t1.at(0); //makes a reference
		auto b = t2.at(0); //makes a reference
		CHECK(std::is_same_v<decltype(a), cgroup_ref>);
		CHECK(std::is_same_v<decltype(b), cgroup_cref>);
		CHECK_THROWS_AS(t2.at(t2.size()), std::out_of_range);
		CHECK_THROWS_AS(t1.at(t1.size()), std::out_of_range);
	}
	SUBCASE("front and back and data")
	{
		auto f1 = t1.front(); //makes a reference
		auto b1 = t1.back();  //makes a reference
		auto f2 = t2.front(); //makes a reference
		auto b2 = t2.back();  //makes a reference
		auto d1 = t1.data();  //makes a reference
		auto d2 = t2.data();  //makes a reference
		CHECK(std::is_same_v<decltype(f1), cgroup_ref>);
		CHECK(std::is_same_v<decltype(b1), cgroup_ref>);
		CHECK(std::is_same_v<decltype(f2), cgroup_cref>);
		CHECK(std::is_same_v<decltype(b2), cgroup_cref>);
		CHECK(std::is_same_v<decltype(d1), typename cgroup_vector::pointer>);
		CHECK(std::is_same_v<decltype(d2), typename cgroup_vector::const_pointer>);
	}
	SUBCASE("capacity")
	{
		auto s = t1.size();
		auto c = t1.capacity();
		CHECK_FALSE(t1.empty());
		CHECK(t1.max_size() > 1000);
		t1.reserve(c + 5);
		CHECK(t1.capacity() >= (c + 5));
		CHECK(t1.size() == s);
		t1.shrink_to_fit();
		CHECK(t1.capacity() >= s);
		WARN_MESSAGE(t1.capacity() == s, "shrink_to_fit doesn't fully (or at all) shrink the vector. This behavior is allowed by the standard for std::vector");
	}
	SUBCASE("modifiers")
	{
		auto s = t1.size();
		t1.pop_back();
		CHECK(t1.size() == (s - 1));
		auto c = t1.capacity();
		t1.clear();
		CHECK(t1.capacity() == c);
		CHECK(t1.size() == 0);
		t1.push_back(ccgroup(1,1));
		CHECK_NOTHROW(t1.push_back( ccgroup(1,1)));
		CHECK_THROWS_AS(t1.push_back(cgroup(groups::C<2>(1),groups::C<4>(1))), std::bad_cast);
		cgroup_vector t3{cgroup(ccgroup(1, 1)), cgroup(ccgroup(0, 0)), cgroup(ccgroup(1, -1))};
		SUBCASE("insert, pop_back and erase")
		{
			auto it = t1.begin();
			CHECK_THROWS_AS(t1.insert(t1.begin(),cgroup(groups::C<5>(0))),std::bad_cast);// the cgroup has the wrong concrete type for the container.
			CHECK_NOTHROW(it = t1.insert(t1.begin(),cgroup(ccgroup(1,-1))));
			CHECK(t1[0] == cgroup(ccgroup(1,-1)));//The insertion happen *before* the position of the given iterator. In this case the inserted value should become the 0th element
			CHECK(it == t1.begin());//the return iterator points to the first (or only) element inserted.
			//current implementation incorrect because the end() iterator does not point to a valid object... iterator comparator needs to be implemented in a way that doesn't rely on the vtable.
			CHECK_NOTHROW(it = t1.insert(t1.end(), 3,ccgroup(1,5) )); //insert 3 ccgroup(1,5) before the end;
			for (; it != t1.end();++it)
			{
				CHECK(*it == ccgroup(1,5) );
			}
			cgroup_vector t4{cgroup(groups::C<2>(1)), groups::C<2>(0), groups::C<2>(-1)};
			CHECK_THROWS_AS(it = t1.insert(t1.begin()+2, t4.begin(),t4.end()),std::bad_cast); //t4 is incompatible with t1.
			CHECK_NOTHROW(it = t1.insert(t1.begin()+2, t3.begin(),t3.end()));
			int i = 0;
			for(auto end = it+(t3.end()-t3.begin()); it != end;++it )
			{
				CHECK(*it == t3[i++]);
			}
			CHECK_NOTHROW(it = t1.insert(t1.begin()+2, t3.rbegin(),t3.rend())); //insertion by reverse iterator.
			i = t3.size()-1;
			for(auto end = it+(t3.end()-t3.begin()); it != end;++it )
			{
				CHECK(*it == t3[i--]);
			}
			cgroup last = *(t1.end()-2);
			CHECK_NOTHROW(t1.erase(t1.end()-1));
			CHECK(last == *(t1.end()-1));
			auto size = t1.size();
			CHECK_NOTHROW( t1.erase(t1.begin(),t1.begin()+3));
			CHECK(size-3 == t1.size());
			i = t1.size();
			CHECK_NOTHROW(t1.pop_back());
			CHECK(i-1 == t1.size());
		}
		auto size = t1.size();
		CHECK_NOTHROW(t1.resize(size+2));
		CHECK(t1.back() == ccgroup());
		CHECK(t1.size() == size + 2);
		CHECK_NOTHROW(t1.resize(size));
		CHECK(size == t1.size());
		CHECK_NOTHROW(t1.resize(size+2,ccgroup(1,10)));
		CHECK(t1.back() == ccgroup(1,10));
		CHECK(t1.size() == size+2);
		auto data1 = t1.data();
		auto data3 = t3.data();
		swap(t1, t3);
		CHECK(data1 == t3.data());
		CHECK(data3 == t1.data());
	}
}



} // namespace quantt
#endif /* D34E37A6_732F_4F45_9171_6B931CC1F812 */
