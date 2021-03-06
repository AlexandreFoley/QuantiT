/*
 * File: LinearAlgebra.cpp
 * Project: QuantiT
 * File Created: Wednesday, 5th August 2020 11:39:13 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Wednesday, 5th August 2020 3:34:30 pm
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * Licensed under GPL v3
 */

#include <exception>
#include <limits>
#include "LinearAlgebra.h"
#include "dimension_manip.h"
namespace quantit
{



int64_t compute_first_index_ascending(torch::Tensor d, torch::Scalar tol, torch::Scalar pow, size_t min_size,
                                    size_t max_size)
{

	// make no asumption regarding the batched nature of the tensors.
	// not trying to optimize based on underlying type of the tensors. might be necessary, but will require significant
	// efforts.
	using namespace torch::indexing;
	auto Last_dim_size = d.sizes()[d.sizes().size() - 1];
	// auto last_index = Last_dim_size-1;
	int64_t first_index = 0;
	auto trunc_val = d.index({Ellipsis, static_cast<int64_t>(first_index)}).abs().pow_(pow);
	auto min_index = Last_dim_size - min_size;
	auto max_index = Last_dim_size - max_size;
	while (first_index <= min_index)
	{
		if (  (trunc_val > tol).any().item().to<bool>() and first_index >= max_index)
			break;
		++first_index;
		trunc_val += d.index({Ellipsis, static_cast<int64_t>(first_index)}).abs().pow_(pow);
	}
	return first_index;
}
/**
 * @brief  compute the last index in a list of singular value such that the error on the trace of the singular value to the pow power smaller than tol.
 * So one can keep [0,last_index].
 * 
 * @param d 
 * @param tol 
 * @param pow 
 * @param min_size 
 * @param max_size 
 * @return int64_t 
 */
int64_t compute_last_index(torch::Tensor d,torch::Scalar tol,torch::Scalar pow,size_t min_size, size_t max_size)
{

	//make no asumption regarding the batched nature of the tensors.
	// not trying to optimize based on underlying type of the tensors. might be necessary, but will require significant efforts.
	using namespace torch::indexing;
	auto Last_dim_size = d.sizes()[d.sizes().size()-1];
	auto toln = torch::pow(torch::ones({})*tol,pow);
	int64_t last_index = Last_dim_size-1;
	auto trunc_val = d.index({Ellipsis,last_index}).abs().pow(pow);// will have to test to make sure we're not doing the abs and pow in place in d
	while (last_index >= min_size)
	{
		if (  (  trunc_val > toln).any().item().to<bool>() and last_index < max_size)
			break;
		--last_index;
		trunc_val += d.index({Ellipsis,last_index}).abs().pow(pow);//Again, abs and pow must not be in place...
	}
	return last_index;
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> truncate(torch::Tensor u,torch::Tensor d, torch::Tensor v, torch::Scalar tol,size_t min_size,size_t max_size, torch::Scalar pow)
{
	//make no asumption regarding the batched nature of the tensors.
	// not trying to optimize based on underlying type of the tensors. might be necessary, but will require significant efforts.
	using namespace torch::indexing;
	auto last_index = compute_last_index(d,tol,pow,min_size,max_size);
	return std::make_tuple(u.index({Ellipsis,Slice(None,last_index+1)}),d.index({Ellipsis,Slice(None,last_index+1) }),v.index({Ellipsis,Slice(None,last_index+1)} ));
}


std::tuple<torch::Tensor,torch::Tensor> truncate(std::tuple<torch::Tensor,torch::Tensor> tensors, torch::Scalar tol,size_t min_size, size_t max_size, torch::Scalar pow)
{	
	auto& [d,u] = tensors;
	return truncate(u,d,tol,min_size,max_size,pow);
}
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> truncate(std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> tensors, torch::Scalar tol,size_t min_size,size_t max_size, torch::Scalar pow)
{	
	auto& [u,d,v] = tensors;
	return truncate(u,d,v,tol,min_size,max_size,pow);
}

std::tuple<torch::Tensor,torch::Tensor> truncate(torch::Tensor e,torch::Tensor u, torch::Scalar tol,size_t min_size,size_t max_size, torch::Scalar pow)
{
	using namespace torch::indexing;
	auto last_index = compute_last_index(e,tol,pow,min_size,max_size);
	return std::make_tuple(u.index({Ellipsis,Slice(None,last_index+1)}),e.index({Ellipsis,Slice(None,last_index+1)}));
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> svd(torch::Tensor A, size_t split)
{
	//debug
	// if (A.isnan().any().item().to<bool>()) throw std::logic_error("nan found in input tensor");
	//debug
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
	size_t min_size = 1;
	size_t max_size = std::numeric_limits<size_t>::max();
	return svd(A,split,tol,min_size,max_size,pow);
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> svd(torch::Tensor A, size_t split,torch::Scalar tol,size_t min_size,size_t max_size, torch::Scalar pow)
{
	return truncate(quantit::svd(A,split), tol,min_size,max_size, pow);
}


std::tuple<torch::Tensor,torch::Tensor> eigh(torch::Tensor A, size_t split)
{
	auto A_dim = A.sizes();
	auto left_dims = A_dim.slice(0,split);
	auto right_dims = A_dim.slice(split,A_dim.size()-split);
	auto l = prod(left_dims);
	auto r = prod(right_dims);
	if (l != r) throw std::invalid_argument("The eigenvalue problem is undefined for rectangular matrices. Either you've input the wrong split, or you need svd");
	auto rA = A.reshape({l,r});
	auto [d,u] = torch::linalg::eigh(rA,"L");
	auto bond_size =  d.sizes();
	u = u.reshape(concat(left_dims , bond_size ) );
	return std::make_tuple(d,u);	
}

std::tuple<torch::Tensor,torch::Tensor> eigh(torch::Tensor A, size_t split,torch::Scalar tol, torch::Scalar pow)
{
	size_t min_size = 1;
	size_t max_size = std::numeric_limits<size_t>::max();
	return eigh(A,split,tol,min_size,max_size,pow);
}

std::tuple<torch::Tensor,torch::Tensor> eigh(torch::Tensor A, size_t split,torch::Scalar tol,size_t min_size,size_t max_size, torch::Scalar pow)
{
	return truncate(quantit::eigh(A,split),tol,min_size,max_size,pow);
}

std::tuple<torch::Tensor,torch::Tensor> eig(torch::Tensor A, size_t split)
{
	throw std::logic_error("non-symetric eigen value not implemented: it will wait until a proper complex implementation is available in torch.");
}
std::tuple<torch::Tensor,torch::Tensor> eig(torch::Tensor A, size_t split,torch::Scalar tol, torch::Scalar pow)
{
	size_t min_size = 1;
	size_t max_size = std::numeric_limits<size_t>::max();
	return eig(A,split,tol,min_size,max_size,pow);
}

std::tuple<torch::Tensor,torch::Tensor> eig(torch::Tensor A, size_t split,torch::Scalar tol,size_t min_size,size_t max_size, torch::Scalar pow)
{
	return truncate(quantit::eig(A,split),tol,min_size,max_size,pow);
}

}//namespace quantit