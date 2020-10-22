/*
 * File: LinearAlgebra.h
 * Project: quantt
 * File Created: Wednesday, 5th August 2020 11:38:57 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Wednesday, 5th August 2020 3:34:16 pm
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#ifndef C5116C03_2050_4F3F_8DCF_C1C103E0B22A
#define C5116C03_2050_4F3F_8DCF_C1C103E0B22A

#include <torch/torch.h>

#include <fmt/core.h>
#include <fmt/ostream.h>
#include "torch_formatter.h"

#include "doctest/doctest_proxy.h"

// //doctest always last. its' macro must work and conflict with pytorch's.
// #include "doctest_redef.h" // makes the redefinition appear without compiler warnings.
// // we don't use pytorch's macro so its fine to redefine them. 
// #include "doctest.h"

namespace quantt
{

//TODO: add possibility of pre-allocated output, and the possibility not to compute the eigen/singular vectors.

/**
 * truncate the input tensors according to the small values of d.
 * truncate all the values of d, d_i, such that \sum_i abs(d_i)^pow <= tol.
 * Assumes the entry of d are ordered in decreasing order.
 * The last dimension of u and v are the one that are truncated along with d.
 * Output is in the same order as input
 */
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> truncate(torch::Tensor u,torch::Tensor d, torch::Tensor v, torch::Scalar tol=0,size_t min_size=1,size_t max_size=-1, torch::Scalar pow=2);
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> truncate(std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> tensors, torch::Scalar tol=0,size_t min_size=1,size_t max_size=-1, torch::Scalar pow=2);

/**
 * truncate the input tensors according to the small values of e.
 * truncate all the values of e, e_i, such that \sum_i abs(e_i)^pow <= tol.
 * Assumes the entry of e are ordered in decreasing order.
 * The last dimension of u is the one that is truncated along with e.
 * Output is in the same order as input
 */
std::tuple<torch::Tensor,torch::Tensor> truncate(torch::Tensor u,torch::Tensor e, torch::Scalar tol=0,size_t min_size=1,size_t max_size=-1, torch::Scalar pow=1);
std::tuple<torch::Tensor,torch::Tensor> truncate(std::tuple<torch::Tensor,torch::Tensor> tensors, torch::Scalar tol=0,size_t min_size=1,size_t max_size=-1, torch::Scalar pow=1);

/**
 * perform the singular value decomposition of the rank N tensor A. 
 * The tensor A is implicitly reshape as a matrix: all the index [0,split] become the left index and ]split,N] is the right index.
 * This behavior is different from the batching done by pytorch's functions.
 * can additionnaly truncate the output tensors, if tol != 0. Consult truncate for mor details.
 * output U,D,V such that A = U*diag(D)*V^T
 */
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> svd(torch::Tensor A, size_t split,torch::Scalar tol,size_t min_size,size_t max_size, torch::Scalar pow = 2);
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> svd(torch::Tensor A, size_t split,torch::Scalar tol, torch::Scalar pow = 2);
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> svd(torch::Tensor A, size_t split);
inline std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> svd(torch::Tensor A, int split){return quantt::svd(A,size_t(split) );}

std::tuple<torch::Tensor,torch::Tensor> eig(torch::Tensor A, size_t split);
inline std::tuple<torch::Tensor,torch::Tensor> eig(torch::Tensor A, int split){return quantt::eig(A,size_t(split) );}
std::tuple<torch::Tensor,torch::Tensor> eig(torch::Tensor A, size_t split,torch::Scalar tol, torch::Scalar pow = 1);
std::tuple<torch::Tensor,torch::Tensor> eig(torch::Tensor A, size_t split,torch::Scalar tol,size_t min_size,size_t max_size, torch::Scalar pow = 1);

std::tuple<torch::Tensor,torch::Tensor> symeig(torch::Tensor A, size_t split);
inline std::tuple<torch::Tensor,torch::Tensor> symeig(torch::Tensor A, int split){return quantt::symeig(A,size_t(split) );}
std::tuple<torch::Tensor,torch::Tensor> symeig(torch::Tensor A, size_t split,torch::Scalar tol, torch::Scalar pow = 1);
std::tuple<torch::Tensor,torch::Tensor> symeig(torch::Tensor A, size_t split,torch::Scalar tol, size_t min_size,size_t max_size,torch::Scalar pow = 1);

qtt_TEST_CASE("Linear Algebra for Tensor network")
{
	torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kFloat64)); //otherwise the type promotion always goes to floats when promoting a tensor
	auto split = 2;
	auto tensor_shape = {10L,3L,10L,3L};
	auto matrix_shape = {30L,30L};
	auto u_shape = {10L,3L,30L};
	auto A = torch::rand(tensor_shape);
	auto ra = A.reshape(matrix_shape);

	auto [u_o,d_o,v_o] = ra.svd();
	auto [u,d,v] = svd(A,2);

	qtt_CHECK(u_o.sizes() == std::vector<int64_t>(matrix_shape));
	qtt_REQUIRE(u.sizes() == std::vector<int64_t>(u_shape));

	qtt_REQUIRE_NOTHROW(auto ru_o = u_o.reshape(u_shape));
	auto ru_o = u_o.reshape(u_shape);

	qtt_CHECK(torch::allclose(ru_o, u));

	auto [e_o,s_o] = ra.symeig(true); 
	auto rs_o = s_o.reshape(u_shape);
	auto [e,s] = symeig(A,2);

	qtt_REQUIRE(s.sizes() == std::vector<int64_t>(u_shape));
	// fmt::print("s {}\n",s);
	// fmt::print("s_o {}\n",rs_o);
	qtt_CHECK(torch::allclose(rs_o, s));
}

}//namespace quantt
#endif /* C5116C03_2050_4F3F_8DCF_C1C103E0B22A */
