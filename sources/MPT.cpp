/*
 * File: MPT.cpp
 * Project: QuanTT
 * File Created: Thursday, 23rd July 2020 10:32:10 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Thursday, 23rd July 2020 10:46:12 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#include "MPT.h"

#include <exception>

namespace quantt{
	
void MPS::move_oc(int i)
	{
		if (not ( i>=0 and i<size())) throw std::invalid_argument(" Proposed orthogonality center falls outside the MPS");
		auto dims = (*this)[orthogonality_center].sizes();
		auto prod = [&dims](size_t start,size_t end)
		{
			std::remove_const_t<std::remove_reference_t< decltype(dims[0])> > out = 1; 
			for (auto i = start; i< end;++i)
			{
				out *= dims[i];
			}
			return out;
		};

	while (i < orthogonality_center)
	{	
		//move right
		auto& curr_oc = (*this)[orthogonality_center];
		auto& next_oc = (*this)[orthogonality_center-1];
		dims = curr_oc.sizes();
		auto reshaped = curr_oc.reshape({dims[0],prod(1,dims.size())});
		auto [u,d,v] = torch::svd(reshaped);
		curr_oc = v.t().reshape(dims); // needs testing. svd documentation makes no mention of complex numbers case.
		// testing shows that v is only transposed in the complex number case as well.
		auto ud = torch::matmul(u,torch::diag(d));
		next_oc = torch::tensordot(next_oc,ud,{2},{0});
		--oc;
	}

	while (i > orthogonality_center)
	{
		// move left
		auto& curr_oc = (*this)[orthogonality_center];
		auto& next_oc = (*this)[orthogonality_center+1];
		dims = curr_oc.sizes();
		auto reshaped = curr_oc.reshape({prod(0,dims.size()-1),dims[dims.size()-1]});
		auto [u,d,v] = torch::svd(reshaped);
		curr_oc = u.reshape(dims);
		auto dv = torch::tensordot(torch::diag(d),v,{1},{1});
		next_oc = torch::tensordot(dv,next_oc,{1},{0});

		++oc;
	}
	// otherwise we're already there, do nothing.
	}

bool MPS::check_one(const Tens & tens)
{
	auto sizes = tens.sizes();
	return (( sizes.size() == 3 and sizes[0] == sizes[2] ));
}

bool MPS::check_ranks() const
{
	auto prev_bond = operator[](0).sizes()[0];
	bool all_rank_3 = std::all_of(begin(),end(),[&prev_bond](const Tens & el)
	{
		using std::swap;
		auto bond = el.sizes()[2];
		swap(bond,prev_bond);
		return el.sizes().size() == 3 and bond == el.sizes()[0];
	});
	return all_rank_3;
}

bool MPO::check_one(const Tens & tens)
{
	auto sizes = tens.sizes();
	// if (!( sizes.size() == 4 and sizes[0] == sizes[2] )) throw std::invalid_argument("The input tensor must have rank 4 and equal bond dimensions (dims 0 and 2).");
	return  ( sizes.size() == 4 and sizes[0] == sizes[2] );
}

bool MPO::check_ranks() const
{
	auto prev_bond = operator[](0).sizes()[0];
	bool all_rank_4 = std::all_of(begin(),end(),[&prev_bond](const Tens & el)
	{
		using std::swap;
		auto bond = el.sizes()[2];
		swap(bond,prev_bond);
		return el.sizes().size() == 4 and bond == el.sizes()[0];
	});
	// assert(all_rank_4);// a MPS must have only rank 3 tensors
	return all_rank_4;
}


torch::Tensor contract(const MPS &a, const MPS &b, const MPO &obs)
{
	assert(a.size() == b.size());
	auto E = torch::ones({1,1,1});
	for (size_t i = 0; i < a.size(); ++i)
	{
		E = torch::tensordot(E, a[i], {0}, {0});
		E = torch::tensordot(E, obs[i], {0, 2}, {0, 3});
		E = torch::tensordot(E, b[i], {0, 2}, {0, 1});
	}
	auto L = torch::ones({1,1,1});
	return torch::tensordot(E, L, {0, 1, 2}, {0, 1, 2});
}

torch::Tensor contract(const MPS& a, const MPS& b)
{
	assert(a.size() == b.size());
	auto E = torch::ones({1,1});
	for (size_t i = 0; i < a.size(); ++i)
	{
		E = torch::tensordot(E, a[i], {0}, {0});
		E = torch::tensordot(E, b[i].conj(), {0, 1}, {0, 1});
	}
	auto L = torch::ones({1,1});
	return torch::tensordot(E, L, {0, 1}, {0, 1});
}

}//namespace QuanTT