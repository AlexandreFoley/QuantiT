/*
 * File: tensorgdot.h
 * Project: quantt
 * File Created: Tuesday, 10th November 2020 10:58:58 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Tuesday, 10th November 2020 10:58:58 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#ifndef BD5A9C21_B42C_4168_AD9F_A788AD4621C8
#define BD5A9C21_B42C_4168_AD9F_A788AD4621C8

#include "doctest/doctest_proxy.h"
#include <torch/torch.h>
namespace quantt
{
/**
 * @brief generalized tensordot, performs D_{ij...klm} = alpha*C_{ij...klm} + beta*\Sum_{...klm} A_{ij...klm}*B_{ij...klm}
 *        Like tensordot is the equivalent of the matrix mutiplcation for tensors, this routine is the equivalent of 
 *        the generalized matrix multiplication for tensor (torch::addmm).
 * @param add The input tensor C
 * @param mul1 input tensor A
 * @param mul2 input tensor B
 * @param dims1 list of dimensions of input1 to be summed
 * @param dims2 list of dimension of input2 to be summed
 * @param beta scalar coefficient to mulitply the result of the dot product with.
 * @param alpha scalar coefficient to scale the input tensor
 * @return torch::Tensor The output tensor D
 */
torch::Tensor tensorgdot(const torch::Tensor& add, const torch::Tensor& mul1, const torch::Tensor& mul2,
                         torch::IntArrayRef dims1, torch::IntArrayRef dims2,
                         torch::Scalar beta = 1, torch::Scalar alpha = 1);
/**
 * @brief generalized tensordot, performs C_{ij...klm} = alpha*C_{ij...klm} + beta*\Sum_{...klm} A_{ij...klm}*B_{ij...klm}
 *        Like tensordot is the equivalent of the matrix mutiplcation for tensors, this routine is the equivalent of 
 *        the generalized matrix multiplication for tensor (torch::addmm).
 * 
 * @param output the input tensor C, and the output tensor
 * @param mul1 input tensor A
 * @param mul2 input tensor B
 * @param dims1 list of dimensions of input1 to be summed
 * @param dims2 list of dimension of input2 to be summed
 * @param beta scalar coefficient to mulitply the result of the dot product with.
 * @param alpha scalar coefficient to scale the input tensor
 * @return torch::Tensor& refrence to the output tensor
 */
torch::Tensor& tensorgdot_(torch::Tensor& output, const torch::Tensor& mul1, const torch::Tensor& mul2,
                           torch::IntArrayRef dims1, torch::IntArrayRef dims2,
                           torch::Scalar beta = 1, torch::Scalar alpha = 1);

/**
 * @brief generalized tensordot, performs D_{ij...klm} = alpha*C_{ij...klm} + beta*\Sum_{...klm} A_{ij...klm}*B_{ij...klm}
 *        Like tensordot is the equivalent of the matrix mutiplcation for tensors, this routine is the equivalent of 
 *        the generalized matrix multiplication for tensor (torch::addmm).
 * @param output the output tensor D
 * @param add The input tensor C
 * @param mul1 input tensor A
 * @param mul2 input tensor B
 * @param dims1 list of dimensions of input1 to be summed
 * @param dims2 list of dimension of input2 to be summed
 * @param beta scalar coefficient to mulitply the result of the dot product with.
 * @param alpha scalar coefficient to scale the input tensor
 * @return torch::Tensor& Reference to the output tensor D
 */
torch::Tensor& tensorgdot_out(torch::Tensor& output, const torch::Tensor& add, const torch::Tensor& mul1, const torch::Tensor& mul2,
                              torch::IntArrayRef dims1, torch::IntArrayRef dims2,
                              torch::Scalar beta = 1, torch::Scalar alpha = 1);
/**
 * @brief generalized tensordot, performs C_{ij...klm} = alpha*C_{ij...klm} + beta*\Sum_{...klm} A_{ij...klm}*B_{ij...klm}
 *        Like tensordot is the equivalent of the matrix mutiplcation for tensors, this routine is the equivalent of 
 *        the generalized matrix multiplication for tensor (torch::addmm).
 * @param output output tensor C, result of the contraction is added to it.
 * @param input1 tensor A
 * @param input2 tensor B
 * @param dims the number of dimensions to contract
 * @param beta scalar coefficient for the addition
 * @param alpha scalar coefficient for the addition
 * @return torch::Tensor& reference to the output tensor.
 */
torch::Tensor& tensorgdot_(torch::Tensor& output, const torch::Tensor& input1, const torch::Tensor& input2,
                           int dims,
                           torch::Scalar beta = 1, torch::Scalar alpha = 1);
qtt_TEST_CASE("generalized tensor dot product")
{
	using namespace torch;
	Tensor A = torch::rand({5, 7, 6});
	Tensor B = torch::rand({5, 6, 7});
	Tensor out = torch::rand({5, 5});
	Tensor out_copy = clone(out);
	std::vector<int64_t> dims1 = {1, 2};
	std::vector<int64_t> dims2 = {2, 1};
	out += tensordot(A, B, dims1, dims2);
	qtt_SUBCASE("tensorgdot")
	{
		auto new_out = tensorgdot(out_copy, A, B, dims1, dims2);
		qtt_CHECK(allclose(new_out, out));
	}
	qtt_SUBCASE("tensorgdot")
	{
		tensorgdot_(out_copy, A, B, dims1, dims2);
		qtt_CHECK(allclose(out_copy, out));
	}
	qtt_SUBCASE("tensorgdot_out")
	{
		Tensor out_copy2 = clone(out_copy);
		tensorgdot_out(out_copy, out_copy2, A, B, dims1, dims2);
		qtt_CHECK(allclose(out_copy, out));
	}
	qtt_SUBCASE("tensorgdot_out v2")
	{
		//this usage is equivalent to a call to tensordot_
		tensorgdot_out(out_copy, out_copy, A, B, dims1, dims2);
		qtt_CHECK(allclose(out_copy, out));
	}
	qtt_SUBCASE("tensorgdot no permutation")
	{
		auto B2 = B.permute({2, 1, 0});
		tensorgdot_(out_copy, A, B2, 2);
		qtt_CHECK(allclose(out_copy, out));
		qtt_CHECK_THROWS(tensorgdot_(out_copy, A, B, 2));
	}
}

} // namespace quantt

#endif /* BD5A9C21_B42C_4168_AD9F_A788AD4621C8 */
