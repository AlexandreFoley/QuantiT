/*
 * File: MPT.h
 * Project: QuanTT
 * File Created: Thursday, 23rd July 2020 9:38:33 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Thursday, 23rd July 2020 10:46:02 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */


#ifndef ADA5A359_8ACF_448D_91BC_09C085F510CC
#define ADA5A359_8ACF_448D_91BC_09C085F510CC

#include <vector>
#include <torch/torch.h>
#include "property.h"
#include <type_traits>
#include <algorithm>

//doctest always last. its' macro must work and conflict with pytorch's.
#include "doctest_redef.h" // makes the redefinition appear without compiler warnings.
// we don't use pytorch's macro so its fine to redefine them.
#include "doctest.h"

namespace quantt{
// matrix product tensors base type. require concrete derived class to implement MPT empty_copy(const S&)
template <class S>
class vector_lift
{
	std::vector<torch::Tensor> tensors;
	
	public:
	//dependent types
	using Tens = torch::Tensor;
	using VTens = std::vector<Tens>;
	using size_type = VTens::size_type;
	using iterator = VTens::iterator;
	using const_iterator = VTens::const_iterator;
	using reverse_iterator = VTens::reverse_iterator;
	using const_reverse_iterator = VTens::const_reverse_iterator;
	using const_reference = VTens::const_reference;
	using reference = VTens::reference;
	//constructors
	vector_lift():tensors(){}
	vector_lift(size_type size):tensors(size){}
	vector_lift(size_type size, const Tens& val):tensors(size,val){}
	template <class T>
	friend class vector_lift;
	template<class T> // can copy and move a vector_lift no matter the derived class.
	vector_lift(const vector_lift<T> & other ):tensors(other.tensors){}
	template<class T> 
	vector_lift(vector_lift<T>&& other) noexcept :tensors(std::move(other.tensors))  {}
	vector_lift(VTens& other): tensors(other) {}
	vector_lift(std::initializer_list<Tens> initl):tensors(initl) {}
	virtual ~vector_lift() {}
	

	explicit operator S()
	{
		S out;
		vector_lift* view = &out;
		view->tensors = this->tensors;
		return out;
	}

	void swap(vector_lift& other) noexcept
	{
		this->tensors.swap(other.tensors);
	}
	friend void swap(vector_lift& lhs, vector_lift& rhs) noexcept
	{
		lhs.swap(rhs); 
	}
	vector_lift& operator=(vector_lift other) noexcept
	{
		swap(other);
		return *this;
	}
	// interface function from std::vector	
	//elements access
	reference at(size_t i) {return tensors.at( i);}
	const_reference at(size_t i) const {return tensors.at( i);}
	reference operator[](size_t i) {return tensors[i];}
	const_reference operator[](size_t i) const {return tensors[i];}
	reference front() {return tensors.front();}
	const_reference front() const {return tensors.front();}
	reference back() {return tensors.back();}
	const_reference back() const {return tensors.back();}
	Tens* data() noexcept {return tensors.data();}
	const Tens* data() const noexcept {return tensors.data();}
	//iterators
	iterator begin() noexcept {return tensors.begin();}
	const_iterator begin() const noexcept {return tensors.begin();}
	const_iterator cbegin() const noexcept {return tensors.cbegin();}
	iterator end() noexcept {return tensors.end();}
	const_iterator end() const noexcept {return tensors.end();}
	const_iterator cend() const noexcept {return tensors.cend();}
	reverse_iterator rbegin() noexcept {return tensors.rbegin();}
	const_reverse_iterator rbegin() const noexcept {return tensors.rbegin();}
	const_reverse_iterator crbegin() const noexcept {return tensors.crbegin();}
	reverse_iterator rend() noexcept {return tensors.rend();}
	const_reverse_iterator rend() const noexcept {return tensors.rend();}
	const_reverse_iterator crend() const noexcept {return tensors.crend();}
	//capacity
	[[nodiscard]] auto empty() const noexcept {return tensors.empty();}
	[[nodiscard]] auto size() const noexcept {return tensors.size();}
	[[nodiscard]] auto max_size() const noexcept {return tensors.max_size();}
	void reserve(size_t new_cap)  { tensors.reserve(new_cap);}
	[[nodiscard]] auto capacity() const noexcept {return tensors.capacity();}
	void shrink_to_fit()  { tensors.shrink_to_fit();}
	//modifiers
	void clear() noexcept {tensors.clear();}
	iterator insert(const_iterator pos, const Tens& val) {return tensors.insert(pos,val);}
	iterator insert(const_iterator pos, Tens&& val) {return tensors.insert(pos,val);}
	iterator insert(const_iterator pos,size_type count, Tens&& val) {return tensors.insert(pos,count,val);}
	template<class InputIT>
	iterator insert(const_iterator pos,InputIT first, InputIT last) {return tensors.insert(pos,first,last);}
	iterator insert(const_iterator pos,std::initializer_list<Tens> list) {return tensors.insert(pos,list);}
	template<class... Args>
	iterator emplace(const_iterator pos, Args&&... args) {return tensors.emplace(pos,std::forward<Args>(args)... );}
	iterator erase(const_iterator pos) {return tensors.erase(pos);}
	iterator erase(const_iterator first,const_iterator last) {return tensors.erase(first,last);}
	void push_back(const Tens& val) {tensors.push_back(val);}
	void push_back(Tens&& val) {tensors.push_back(val);}
	template<class... Args>
	auto emplace_back(Args&&... args) {return tensors.emplace_back(std::forward<Args>(args)...);}
	void pop_back() {tensors.pop_back();}
	void resize(size_type count) {tensors.resize(count);}
	void resize(size_type count, const Tens& value) {tensors.resize(count,value);}

	//stuff about the tensors

	S to(const torch::TensorOptions& options={}, bool non_blocking=false,bool copy = false,c10::optional<c10::MemoryFormat> memory_format=c10::nullopt) const
	{// those function can't really exist outside the derived class if empty_copy is private, which it is for MPT, MPS and MPO.
		S out = empty_copy(static_cast<S const &>( *this) );
		auto out_it = out.begin();
		std::for_each(this->cbegin(),this->cend(),[&out_it,&options,non_blocking,copy,memory_format](const auto& atensor)
		{
			*(out_it++) = atensor.to(options,non_blocking,copy,memory_format);
		});
		return out;
	}
	S to(torch::Device device, torch::ScalarType dtype, bool non_blocking=false,bool copy = false,c10::optional<c10::MemoryFormat> memory_format=c10::nullopt) const
	{
		S out = empty_copy(static_cast<S const &>( *this) );
		auto out_it = out.begin();
		std::for_each(this->cbegin(),this->cend(),[&out_it,device,dtype,non_blocking,copy,memory_format](const auto& atensor)
		{
			*(out_it++) = atensor.to(device,dtype,non_blocking,copy,memory_format);
		});
		return out;
	}
	S to(torch::ScalarType dtype, bool non_blocking=false,bool copy = false,c10::optional<c10::MemoryFormat> memory_format=c10::nullopt) const
	{
		S out = empty_copy(static_cast<S const &>( *this) );
		auto out_it = out.begin();
		std::for_each(this->cbegin(),this->cend(),[&out_it,dtype,non_blocking,copy,memory_format](const auto& atensor)
		{
			*(out_it++) = atensor.to(dtype,non_blocking,copy,memory_format);
		});
		return out;
	}
	S to(caffe2::TypeMeta type_meta, bool non_blocking=false,bool copy = false,c10::optional<c10::MemoryFormat> memory_format=c10::nullopt) const
	{
		S out = empty_copy(static_cast<S const &>( *this) );
		auto out_it = out.begin();
		std::for_each(this->cbegin(),this->cend(),[&out_it,type_meta,non_blocking,copy,memory_format](const auto& atensor)
		{
			*(out_it++) = atensor.to(type_meta,non_blocking,copy,memory_format);
		});
		return out;
	}
	S to(const torch::Tensor& other, bool non_blocking=false,bool copy = false,c10::optional<c10::MemoryFormat> memory_format=c10::nullopt) const
	{
		S out = empty_copy(static_cast<S const &>( *this) );
		auto out_it = out.begin();
		std::for_each(this->cbegin(),this->cend(),[&out_it,&other,non_blocking,copy,memory_format](const auto& atensor)
		{
			*(out_it++) = atensor.to(other,non_blocking,copy,memory_format);
		});
		return out;
	}
	//inplace version of to, will resolve to any equivalent out-of-place equivalent.
	template<class... Args>
	void inplace_to(Args&& ... args)
	{
		auto other = this->to(std::forward<Args>(args)...);
		swap(other);
	}
};
class MPT final : public vector_lift<MPT>
{
	public:
	
	MPT():vector_lift<MPT>(){}
	MPT(size_type size):vector_lift<MPT>(size){}
	MPT(size_type size, const Tens& val):vector_lift<MPT>(size,val){}
	MPT(const MPT & other ):vector_lift<MPT>(other){}
	MPT(MPT&& other) noexcept :vector_lift<MPT>(std::move(other))  {}
	MPT(std::initializer_list<Tens> initl):vector_lift<MPT>(initl) {}
	virtual ~MPT() {}

	void swap(MPT& other)
	{
		vector_lift<MPT>::swap(other);
	}

	friend void swap(MPT& lhs, MPT & rhs)
	{
		lhs.swap(rhs);
	}

	private:
	static MPT empty_copy(const MPT& in)
	{
		return MPT(in.size());
	}
};

class MPS final: public vector_lift<MPS> // specialization for rank 3 tensors, addionnaly can have an orthogonality center.
{
	public:
	property<size_t,MPS> orthogonality_center; // read only for users
	/**
	 * Check that all the tensors are rank three, and that the size of the bond dimension of neighbouring tensors matches.
	 */
	void check_ranks() const;
	static void check_one(const Tens& tens);
	MPS():vector_lift<MPS>(),orthogonality_center(){}
	MPS(size_type size):vector_lift<MPS>(size){}
	MPS(const MPS & other ):vector_lift(other),orthogonality_center(other.orthogonality_center) {}
	MPS(MPS&& other) noexcept :vector_lift(std::move(other)),orthogonality_center(other.orthogonality_center)  {}
	
	MPS(size_type size, const Tens& val,size_t oc):vector_lift(size,val),orthogonality_center(oc) {check_one(val);}
	
	MPS(const MPT & other,size_t oc=0 ):vector_lift<MPS>(other),orthogonality_center(oc) {check_ranks();}
	MPS(MPT&& other,size_t oc=0) noexcept :vector_lift<MPS>(std::move(other)),orthogonality_center(oc)  {check_ranks();}
	
	virtual ~MPS() {}

	/**
	 * explicit conversion to a MPT. discards the position of the orthogonality center and lift the MPS constraints.
	 * Be careful, this function does not create a copy of the underlying tensors.
	 */
	explicit operator MPT()
	{// careful! the underlying data is shared.
		return MPT(vector_lift<MPT>(static_cast<vector_lift<MPS>& >(*this) ));
	}

	/**
	 * move the orthogonality center to the position i on the chain.
	 */
	void move_oc(int i);

	private:
	size_t& oc = orthogonality_center.value; // private direct access to the value variable.
	static MPS empty_copy(const MPS& in)
	{
		return MPS(in.size(),in.oc);
	}
};


TEST_CASE("MPS basic manipulation")
{
	SUBCASE("Construction"){
		MPS A({torch::rand({1,4,3}),torch::rand({3,4,1})});

		REQUIRE( A.size() == 2);
		REQUIRE( A.capacity() >= 2);

		CHECK( A.orthogonality_center == 0 );
		auto size_0 = std::vector{1L,4L,3L};
		auto size_1 = std::vector{3L,4L,1L};
		CHECK(  A[0].sizes() == size_0  );
		CHECK(  A[1].sizes() == size_1  );
		size_0 = A[0].sizes().vec(); //copy the current size of A[0];
		SUBCASE("moving orthogonality center"){
			A.move_oc(1);
			CHECK( A.orthogonality_center == 1 );
			CHECK(  A[0].sizes() == size_0  );
			CHECK(  A[1].sizes() == size_1  );
		}
		SUBCASE("conversion to MPT")
		{
		auto B = MPT(A);
		}
	}

}


class MPO final: public vector_lift<MPO> // specialization for rank 4 tensors
{
	public:
	
	void check_ranks() const;
	static void check_one(const Tens& tens);
	
	MPO():vector_lift<MPO>(){}
	MPO(size_type size):vector_lift<MPO>(size){}
	MPO(const MPO & other ):vector_lift<MPO>(other){}
	MPO(MPO&& other) noexcept :vector_lift<MPO>(std::move(other))  {}
	MPO(std::initializer_list<Tens> initl):vector_lift<MPO>(initl) {}
	
	MPO(size_type size, const Tens& val):vector_lift<MPO>(size,val){check_one(val);}
	
	MPO(const MPT & other,size_t oc=0 ):vector_lift<MPO>(other) {check_ranks();}
	MPO(MPT&& other,size_t oc=0) noexcept :vector_lift<MPO>(std::move(other))  {check_ranks();}
	
	virtual ~MPO() {}

	void swap(MPO& other)
	{
		vector_lift<MPO>::swap(other);
	}

	explicit operator MPT()
	{// careful! the underlying data is shared.
		return MPT(vector_lift<MPT>(static_cast<vector_lift<MPO>& >(*this) ));
	}
	
	friend void swap(MPO& lhs, MPO & rhs)
	{
		lhs.swap(rhs);
	}

	private:
	static MPO empty_copy(const MPO& in)
	{
		return MPT(in.size());
	}

};

}//namespace QuanTT
#endif /* ADA5A359_8ACF_448D_91BC_09C085F510CC */
