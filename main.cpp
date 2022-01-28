/*
 * File: main.cpp
 * Project: QuanTT
 * File Created: Thursday, 16th July 2020 1:47:39 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 *
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */
#define DOCTEST_CONFIG_IMPLEMENT
#include "blockTensor/btensor.h"
#include "dmrg.h"
#include "doctest/doctest_proxy.h"
#include "include/torch_formatter.h"
#include "models.h"
#include "tensorgdot.h"
#include <fmt/core.h>
#include <fmt/ostream.h>
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
	doctest_context.addFilter("test-case-exclude",
	                          "**"); // don't run the tests. with this qtt_CHECKS, qtt_REQUIRES, etc. should work
	                                 // outside test context. not that i want to do that.

	using namespace quantt;
	using namespace torch::indexing;
	at::init_num_threads();
	fmt::print("C++ standard version in use {}\n", __cplusplus);
	torch::set_default_dtype(torch::scalarTypeToTypeMeta(
	    torch::kFloat64)); // otherwise the type promotion always goes to floats when promoting a tensor
	torch::Device cuda_device(
	    torch::kCPU); // default the cuda device to a cpu. small lie to keep the code working if there isn't one.
	if (torch::cuda::is_available())
	{
		fmt::print("CUDA is available!\n");
		cuda_device = torch::Device(torch::kCUDA); // set the cuda_device to the actual gpu if it would work
	}

	auto X = torch::rand({5,10},torch::requires_grad());
	// auto Y = torch::rand({10,5}, torch::requires_grad());
	auto Y = X * 2;
	auto out = Y.mean();
	fmt::print("{}\n\n",out);
	out.backward();
	auto grad = X.grad();


	fmt::print("{}\n",grad); // all 50 elements should be 2/50 = 0.04

	using cval = quantt::quantity<quantt::conserved::Z,quantt::conserved::Z>;
	auto FermionShape = quantt::btensor({{{1,cval(0,0)},{1,cval(1,1)},{1,cval(1,-1)},{1,cval(2,0)}}},cval(0,0));

	fmt::print("{}\n\n",FermionShape);
	for(int i=0;i<4;++i)
	{
		for(auto& x:FermionShape.block_quantities({i}))
		{
			fmt::print("{} ",x);
		}
		for(auto& x:FermionShape.block_sizes({i}))
		{
			fmt::print("{} ",x);
		}
		fmt::print("\n\n");
	}
	
	return 0;
}