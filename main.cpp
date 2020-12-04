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


	using namespace torch;
	using cqt = conserved::C<5>;
	btensor B({{{3, cqt(4)}, {2, cqt(0)}}, {{2, cqt(0)}, {3, cqt(1)}}, {{1, cqt(1)}, {3, cqt(0)}}}, any_quantity(cqt(1)));


	return 0;
}