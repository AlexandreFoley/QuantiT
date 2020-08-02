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
		assert( i>=0 and i<size());
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

void MPS::check_ranks()
{
	bool all_rank_3 = std::all_of(begin(),end(),[](const Tens & el)
	{
		return el.sizes().size() == 3;
	});
	assert(all_rank_3);// a MPS must have only rank 3 tensors
}

}//namespace QuanTT