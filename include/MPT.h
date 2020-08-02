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
// matrix product tensors. any rank.
class MPT
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
	MPT():tensors(){}
	MPT(size_type size):tensors(size){}
	MPT(size_type size, const Tens& val):tensors(size,val){}
	MPT(const MPT & other ):tensors(other.tensors){}
	MPT(MPT&& other) noexcept :tensors(std::move(other.tensors))  {}
	MPT(std::initializer_list<Tens> initl):tensors(initl) {}
	virtual ~MPT() {}
	
	void swap(MPT& other) noexcept
	{
		this->tensors.swap(other.tensors);
	}
	friend void swap(MPT& lhs, MPT& rhs) noexcept
	{
		lhs.swap(rhs); 
		// this is gonna work to swap a MPT with a derived class, might lead to strange behavior...
		// this, perhaps, is justification enough to make sure implicit conversion between the matrix product classes are not possible.
	}
	MPT& operator=(MPT other) noexcept
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

	MPT to(const torch::TensorOptions& options={}, bool non_blocking=false,bool copy = false,c10::optional<c10::MemoryFormat> memory_format=c10::nullopt) const
	{
		MPT out(this->size());
		auto out_it = out.begin();
		std::for_each(this->cbegin(),this->cend(),[&out_it,&options,non_blocking,copy,memory_format](const auto& atensor)
		{
			*(out_it++) = atensor.to(options,non_blocking,copy,memory_format);
		});
		return out;
	}
	MPT to(torch::Device device, torch::ScalarType dtype, bool non_blocking=false,bool copy = false,c10::optional<c10::MemoryFormat> memory_format=c10::nullopt) const
	{
		MPT out(this->size());
		auto out_it = out.begin();
		std::for_each(this->cbegin(),this->cend(),[&out_it,device,dtype,non_blocking,copy,memory_format](const auto& atensor)
		{
			*(out_it++) = atensor.to(device,dtype,non_blocking,copy,memory_format);
		});
		return out;
	}
	MPT to(torch::ScalarType dtype, bool non_blocking=false,bool copy = false,c10::optional<c10::MemoryFormat> memory_format=c10::nullopt) const
	{
		MPT out(this->size());
		auto out_it = out.begin();
		std::for_each(this->cbegin(),this->cend(),[&out_it,dtype,non_blocking,copy,memory_format](const auto& atensor)
		{
			*(out_it++) = atensor.to(dtype,non_blocking,copy,memory_format);
		});
		return out;
	}
	MPT to(caffe2::TypeMeta type_meta, bool non_blocking=false,bool copy = false,c10::optional<c10::MemoryFormat> memory_format=c10::nullopt) const
	{
		MPT out(this->size());
		auto out_it = out.begin();
		std::for_each(this->cbegin(),this->cend(),[&out_it,type_meta,non_blocking,copy,memory_format](const auto& atensor)
		{
			*(out_it++) = atensor.to(type_meta,non_blocking,copy,memory_format);
		});
		return out;
	}
	MPT to(const torch::Tensor& other, bool non_blocking=false,bool copy = false,c10::optional<c10::MemoryFormat> memory_format=c10::nullopt) const
	{
		MPT out(this->size());
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


class MPS : public MPT // specialization for rank 3 tensors, addionnaly can have an orthogonality center.
{
	public:
	property<size_t,MPS> orthogonality_center; // read only for users
	/**
	 * Check that the rank of all the tensor is three, and that the size of the bond dimension of neighbour tensors matches.
	 */
	void check_ranks();
	MPS():MPT(),orthogonality_center(){}
	MPS(size_type size):MPT(size){}
	MPS(size_type size, const Tens& val,size_t oc):MPT(size,val),orthogonality_center(oc) {check_ranks();}
	MPS(const MPT & other,size_t oc=0 ):MPT(other),orthogonality_center(oc) {check_ranks();}
	MPS(const MPS & other ):MPT(other),orthogonality_center(other.orthogonality_center) {}
	MPS(MPT&& other,size_t oc=0) noexcept :MPT(std::move(other)),orthogonality_center(oc)  {check_ranks();}
	MPS(MPS&& other) noexcept :MPT(std::move(other)),orthogonality_center(other.orthogonality_center)  {}
	virtual ~MPS() {}

	/**
	 * move the orthogonality center to the position i on the chain.
	 */
	void move_oc(int i);

	private:
		size_t& oc = orthogonality_center.value; // private direct access to the value variable.
		void check_ranks() const; //make sure all the ranks are MPS compatible
};

TEST_CASE("MPS basic manipulation")
{
	MPS({torch::rand({1,4,3}),torch::rand({3,4,1})});
}


class MPO : public MPT // specialization for rank 4 tensors
{

};

}//namespace QuanTT
#endif /* ADA5A359_8ACF_448D_91BC_09C085F510CC */
