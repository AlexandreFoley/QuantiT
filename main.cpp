/*
 * File: main.cpp
 * Project: QuanTT
 * File Created: Thursday, 16th July 2020 1:47:39 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Thursday, 23rd July 2020 10:29:13 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest_proxy.h"
#include "tensorgdot.h"
#include "blockTensor/btensor.h"
#include "dmrg.h"
#include "fmt/ostream.h"
#include "include/torch_formatter.h"
#include "models.h"
#include <fmt/core.h>
#include <map>
#include <ostream>
#include <torch/torch.h>

void print_MPS_dims(const quantt::MPS &mps)
{
	fmt::print("MPS size: ");
	for (const auto &i : mps)
	{
		fmt::print("{},", i.sizes());
	}
	fmt::print("\n");
}

int main()
{
	doctest::Context doctest_context;
	doctest_context.addFilter("test-case-exclude", "**"); //don't run the tests. with this qtt_CHECKS, qtt_REQUIRES, etc. should work outside test context. not that i want to do that.

	using namespace quantt;
	fmt::print("C++ standard version in use {}\n", __cplusplus);
	torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kFloat64)); //otherwise the type promotion always goes to floats when promoting a tensor
	torch::Device cuda_device(torch::kCPU);									// default the cuda device to a cpu. small lie to keep the code working if there isn't one.
	if (torch::cuda::is_available())
	{
		fmt::print("CUDA is available!\n");
		cuda_device = torch::Device(torch::kCUDA); // set the cuda_device to the actual gpu if it would work
	}

	using cqt = conserved::C<5>;
	using index = btensor::index_list;
	any_quantity flux(cqt(0));

	btensor A({{{2, cqt(0)}, {3, cqt(1)}},
			   {{2, cqt(0)}, {3, cqt(1).inverse()}}},
			  flux);
	auto A00 = torch::rand({2, 2});
	auto A11 = torch::rand({3, 3});
	A.block({0, 0}) = A00;
	A.block({1, 1}) = A11;
	auto B00 = 2 * A00;
	auto B11 = 2 * A11;
	auto C00 = 3 * A00; //also A post add_
	auto C11 = 3 * A11; //also A post add_
	auto B = A.add(A);
	auto C = A.add(B);
	B.block_at({0, 0});
	C.add_(B);
	A.add_(std::move(B));

	return 0;
}