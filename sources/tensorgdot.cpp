/*
 * File: tensorgdot.cpp
 * Project: quantt
 * File Created: Tuesday, 10th November 2020 11:01:46 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Tuesday, 10th November 2020 11:01:46 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */
#include "tensorgdot.h"
#include <ATen/ATen.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
namespace quantt
{

/**
 * @brief compute the reshaping necessary to make this a matrix operation. Also does the broadcasting on the size 1 dimensions.
 * 
 * @param add 
 * @param t1 
 * @param t2 
 * @param input1 
 * @param input2 
 * @param dims1 
 * @param dims2 
 * @return std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, int64_t, int64_t, int64_t> 
 */
//          permutation1,        permutation2,         return shape,  left matrix size, common mat size, right mat size
std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, int64_t, int64_t, int64_t>
compute_shape(const torch::Tensor& add, torch::Tensor& t1, torch::Tensor& t2, const torch::Tensor& input1, const torch::Tensor& input2,
              torch::IntArrayRef dims1, torch::IntArrayRef dims2)
{
	using namespace torch;
	TORCH_CHECK(dims1.size() == dims2.size(), "both dimension lists should have same length");
	int64_t csize = 1; // total size of the contracted dimensions

	for (size_t i = 0; i < dims1.size(); ++i)
	{
		int s1 = input1.size(dims1[i]);
		int s2 = input2.size(dims2[i]);
		if (s2 == 1)
		{ // broadcasted dimensions can be summed right away
			t1 = t1.sum(dims1[i], true);
		}
		else if (s1 == 1)
		{
			t2 = t2.sum(dims2[i], true);
		}
		else
		{
			TORCH_CHECK(s1 == s2, fmt::format("contracted dimensions need to match, but first has size {} in dim {}  and second has size {} in dim {}", s1, dims1[i],
			                                  s2, dims2[i]));
			csize *= s1;
		}
	}

	auto cdims1 = at::dim_list_to_bitset(dims1, input1.dim());
	auto cdims2 = at::dim_list_to_bitset(dims2, input2.dim());
	std::vector<int64_t> p1, p2, rsizes; // p1, p2: input permutations, rsizes: sizes of the result
	p1.reserve(input1.dim());
	p2.reserve(input2.dim());
	rsizes.reserve(input1.dim() + input2.dim() - (int64_t)dims1.size());
	int64_t size1 = 1; // number of non-contracted elements in input1
	int64_t size2 = 1; // number of non-contracted elements in input2

	// fill the permutations and compute sizes
	for (int64_t i = 0; i < input1.dim(); i++)
	{
		if (!cdims1[i])
		{
			p1.emplace_back(i);
			size1 *= t1.size(i);
			rsizes.emplace_back(t1.size(i));
		}
	}
	for (size_t i = 0; i < dims1.size(); i++)
	{
		p1.emplace_back(dims1[i]);
	}
	for (size_t i = 0; i < dims2.size(); i++)
	{
		p2.emplace_back(dims2[i]);
	}
	for (int64_t i = 0; i < input2.dim(); i++)
	{
		if (!cdims2[i])
		{
			p2.emplace_back(i);
			size2 *= t2.size(i);
			rsizes.emplace_back(t2.size(i));
		}
	}
	TORCH_CHECK(rsizes == add.sizes(), "tensordot result incompatible with output tensor shape");
	return make_tuple(p1, p2, rsizes, size1, csize, size2);
}

torch::Tensor& tensorgdot_out(torch::Tensor& output, const torch::Tensor& add_tens, const torch::Tensor& input1, const torch::Tensor& input2,
                              torch::IntArrayRef dims1, torch::IntArrayRef dims2,
                              torch::Scalar beta, torch::Scalar alpha)
{
	using namespace torch;
	Tensor t1 = input1;
	Tensor t2 = input2;
	auto [p1, p2, rsizes, size1, csize, size2] = compute_shape(add_tens, t1, t2, input1, input2, dims1, dims2);
	// permut and reshape for matrix multiplication
	t1 = t1.permute(p1).reshape({size1, csize});
	t2 = t2.permute(p2).reshape({csize, size2});
	// multiply and reshape to target size
	auto rout = output.reshape({size1, size2});
	at::addmm_out(rout, add_tens.reshape({size1, size2}), t1, t2, beta, alpha);
	return output;
}

torch::Tensor tensorgdot(const torch::Tensor& output, const torch::Tensor& input1, const torch::Tensor& input2,
                         torch::IntArrayRef dims1, torch::IntArrayRef dims2,
                         torch::Scalar beta, torch::Scalar alpha)
{
	using namespace torch;
	TORCH_CHECK(dims1.size() == dims2.size(), "both dimension lists should have same length");
	Tensor t1 = input1;
	Tensor t2 = input2;
	auto [p1, p2, rsizes, size1, csize, size2] = compute_shape(output, t1, t2, input1, input2, dims1, dims2);
	// permut and reshape for matrix multiplication
	t1 = t1.permute(p1).reshape({size1, csize});
	t2 = t2.permute(p2).reshape({csize, size2});
	// multiply and reshape to target size
	auto rout = output.reshape({size1, size2});
	return at::addmm(rout, t1, t2, beta, alpha).reshape(rsizes);
}

torch::Tensor& tensorgdot_(torch::Tensor& output, const torch::Tensor& input1, const torch::Tensor& input2,
                           torch::IntArrayRef dims1, torch::IntArrayRef dims2,
                           torch::Scalar beta, torch::Scalar alpha)
{
	using namespace torch;
	Tensor t1 = input1;
	Tensor t2 = input2;
	auto [p1, p2, rsizes, size1, csize, size2] = compute_shape(output, t1, t2, input1, input2, dims1, dims2);
	// permut and reshape for matrix multiplication
	t1 = t1.permute(p1).reshape({size1, csize});
	t2 = t2.permute(p2).reshape({csize, size2});
	// multiply and reshape to target size
	auto rout = output.reshape({size1, size2});
	rout.addmm_(t1, t2, beta, alpha);
	return output;
}

torch::Tensor& tensorgdot_(torch::Tensor& output, const torch::Tensor& input1, const torch::Tensor& input2,
                           int dims,
                           torch::Scalar beta, torch::Scalar alpha)
{ //tensor dot without permutation. dims are counted from the last dim of the first tensor, and from the first dim of the second tensor
	using namespace torch;
	Tensor t1 = input1;
	Tensor t2 = input2;
	auto sizes1 = t1.sizes();
	auto sizes2 = t2.sizes();
	// auto [p1, p2, rsizes, size1, csize, size2] = compute_shape(output, t1, t2, input1, input2, dims1, dims2);
	std::vector<int64_t> rsizes(t1.dim() + t2.dim() - 2 * dims);
	auto rit = rsizes.begin();
	auto prod = [&rit](auto& a, auto& b) {
		*(rit++) = b; //assign b then increment iterator, this construct the output section size vector.
		return a * b;
	};
	int size1 = std::accumulate(sizes1.begin(), sizes1.end() - dims, 1, prod); //cannot use reduce, operation must be done in order
	int size2 = std::accumulate(sizes2.begin() + dims, sizes2.end(), 1, prod); //cannot use reduce, operation must be done in order
	constexpr auto output_tensor_shape = "output tensor shape {} does not match the result of the contraction {}."; 
	TORCH_CHECK(rsizes == output.sizes(), fmt::format(output_tensor_shape,fmt::join(output.sizes(),","),rsizes));
	int csize = 1;
	{
		auto it1 = sizes1.end() - dims;
		auto it2 = sizes2.begin();
		for (; it1 != sizes1.end(); ++it1, ++it2)
		{
			constexpr auto fmtstr = "contracted dimensions need to match, but first has "
			                        "size {} in dim {}  and second has size {} in dim {}";
			TORCH_CHECK(*it1 == *it2, fmt::format(fmtstr, *it1, it1 - sizes1.begin(),
			                                      *it2, it2 - sizes2.begin()));
			csize *= *it1;
		}
	}
	// reshape for matrix multiplication
	t1 = t1.reshape({size1, csize});
	t2 = t2.reshape({csize, size2});
	// multiply and reshape to target size
	auto rout = output.reshape({size1, size2});
	rout.addmm_(t1, t2, beta, alpha);
	return output;
}

} // namespace quantt