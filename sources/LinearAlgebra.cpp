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

#include <exception>
#include "LinearAlgebra.h"
#include "dimension_manip.h"
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

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> truncate(torch::Tensor u,torch::Tensor d, torch::Tensor v, torch::Scalar tol, torch::Scalar pow)
{
	//make no asumption regarding the batched nature of the tensors.
	// not trying to optimize based on underlying type of the tensors. might be necessary, but will require significant efforts.
	using namespace torch::indexing;
	auto last_index = compute_last_index(d,tol,pow);
	return std::make_tuple(u.index({Ellipsis,Slice(None,last_index)}),d.index({Ellipsis,Slice(None,last_index) }),v.index({Ellipsis,Slice(None,last_index)} ));
}


std::tuple<torch::Tensor,torch::Tensor> truncate(std::tuple<torch::Tensor,torch::Tensor> tensors, torch::Scalar tol, torch::Scalar pow)
{	
	auto& [d,u] = tensors;
	return truncate(u,d,tol,pow);
}
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> truncate(std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> tensors, torch::Scalar tol, torch::Scalar pow)
{	
	auto& [u,d,v] = tensors;
	return truncate(u,d,v,tol,pow);
}

std::tuple<torch::Tensor,torch::Tensor> truncate(torch::Tensor e,torch::Tensor u, torch::Scalar tol, torch::Scalar pow)
{
	using namespace torch::indexing;
	auto last_index = compute_last_index(e,tol,pow);
	return std::make_tuple(u.index({Ellipsis,Slice(None,last_index)}),e.index({Ellipsis,Slice(None,last_index)}));
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> svd(torch::Tensor A, size_t split)
{
	auto A_dim = A.sizes();
	auto left_dims = A_dim.slice(0,split);
	auto right_dims = A_dim.slice(split,A_dim.size()-split);
	auto rA = A.reshape({prod(left_dims),prod(right_dims)});
	auto [u,d,v] = rA.svd();
	auto bond_size =  d.sizes();
	u = u.reshape(concat(left_dims , bond_size ) );
	v = v.reshape(concat(right_dims, bond_size ));
	return std::make_tuple(u,d,v);
}


std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> svd(torch::Tensor A, size_t split,torch::Scalar tol, torch::Scalar pow)
{
	return truncate(quantt::svd(A,split), tol, pow);
}


std::tuple<torch::Tensor,torch::Tensor> symeig(torch::Tensor A, size_t split)
{
	auto A_dim = A.sizes();
	auto left_dims = A_dim.slice(0,split);
	auto right_dims = A_dim.slice(split,A_dim.size()-split);
	auto l = prod(left_dims);
	auto r = prod(right_dims);
	if (l != r) throw std::invalid_argument("The eigenvalue problem is undefined for rectangular matrices. Either you've input the wrong split, or you need svd");
	auto rA = A.reshape({l,r});
	auto [d,u] = rA.symeig(true);
	auto bond_size =  d.sizes();
	u = u.reshape(concat(left_dims , bond_size ) );
	return std::make_tuple(d,u);	
}

std::tuple<torch::Tensor,torch::Tensor> symeig(torch::Tensor A, size_t split,torch::Scalar tol, torch::Scalar pow)
{
	return truncate(quantt::symeig(A,split),tol,pow);
}

std::tuple<torch::Tensor,torch::Tensor> eig(torch::Tensor A, size_t split)
{
	throw std::logic_error("non-symetric eigen value not implemented: it will wait until a proper complex implementation is available in torch.");
}
std::tuple<torch::Tensor,torch::Tensor> eig(torch::Tensor A, size_t split,torch::Scalar tol, torch::Scalar pow)
{
	return truncate(quantt::eig(A,split),tol,pow);
}

}//namespace quantt