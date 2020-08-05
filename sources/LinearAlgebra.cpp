/*
 * File: LinearAlgebra.cpp
 * Project: quantt
 * File Created: Wednesday, 5th August 2020 11:39:13 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Wednesday, 5th August 2020 3:34:30 pm
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#include "LinearAlgebra.h"
namespace quantt
{

auto compute_last_index(torch::Tensor d,torch::Scalar tol,torch::Scalar pow)
{

	//make no asumption regarding the batched nature of the tensors.
	// not trying to optimize based on underlying type of the tensors. might be necessary, but will require significant efforts.
	using namespace torch::indexing;
	auto Last_dim_size = d.sizes()[d.sizes().size()-1];
	auto last_index = Last_dim_size-1;
	auto trunc_val = d.index({Ellipsis,last_index}).abs().pow(pow);// will have to test to make sure we're not doing the abs and pow in place in d
	while (last_index > 0)
	{
		if (  (trunc_val > tol).any().item().to<bool>() )
			break;
		
		--last_index;
		trunc_val += d.index({Ellipsis,last_index}).abs().pow(pow);//Again, abs and pow must not be in place...
	}
	return last_index;
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> truncate(torch::Tensor u,torch::Tensor d, torch::Tensor v, torch::Scalar tol=0, torch::Scalar pow=2)
{
	//make no asumption regarding the batched nature of the tensors.
	// not trying to optimize based on underlying type of the tensors. might be necessary, but will require significant efforts.
	using namespace torch::indexing;
	auto last_index = compute_last_index(d,tol,pow);
	return std::make_tuple(u.index({Ellipsis,last_index}),d.index({Ellipsis,last_index}),v.index({Ellipsis,last_index}));
}

std::tuple<torch::Tensor,torch::Tensor> truncate(torch::Tensor u,torch::Tensor e, torch::Scalar tol=0, torch::Scalar pow=1)
{
	using namespace torch::indexing;
	auto last_index = compute_last_index(e,tol,pow);
	return std::make_tuple(u.index({Ellipsis,last_index}),e.index({Ellipsis,last_index}));
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> svd(torch::Tensor A, size_t split)
{
	auto dims = A.sizes();
	auto prod = [&dims](size_t start,size_t end)
	{
		std::remove_const_t<std::remove_reference_t< decltype(dims[0])> > out = 1; 
		for (auto i = start; i< end;++i)
		{
			out *= dims[i];
		}
		return out;
	};
	
	auto rA = A.reshape({prod(0,split),prod(split+1,dims.size()-1)});
	auto [u,d,v] = A.svd();
	//u = u.reshape({})
	
}


}//namespace quantt