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

#include <torch/torch.h>
#include <fmt/core.h>
#include "include/torch_formatter.h"
#include "fmt/ostream.h"

#include "dmrg.h"
#include "models.h"

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
	using namespace quantt;
	fmt::print("C++ standard version in use {}\n", __cplusplus);
	torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kFloat64)); //otherwise the type promotion always goes to floats when promoting a tensor
	torch::Device cuda_device(torch::kCPU);									// default the cuda device to a cpu. small lie to keep the code working if there isn't one.
	if (torch::cuda::is_available())
	{
		fmt::print("CUDA is available!\n");
		cuda_device = torch::Device(torch::kCUDA); // set the cuda_device to the actual gpu if it would work
	}

	auto local_tens = 2 * torch::rand({4, 2, 4}) - 1;
	constexpr size_t size = 10;
	auto hamil = quantt::Heisenberg(torch::tensor(-1.), size);
	quantt::MPS state(size, local_tens);
	{
		using namespace torch::indexing;
		state[0] = state[0].index({Slice(0, 1), Ellipsis});
		state[size - 1] = state[size - 1].index({Ellipsis, Slice(3, 4)});
	}
	state.check_ranks();
	quantt::dmrg_options options;
	options.convergence_criterion = 1e-6;
	options.cutoff = options.def_cutoff * 1e-2;
	// auto two_sites_heis = quantt::details::compute_2sitesHamil(hamil);
	// fmt::print("hamil: \n{}\n",two_sites_heis[0].reshape({4,4}));
	// fmt::print("Length of the state: {}. norm of the state: \n{}\n", state.size(), contract(state, state));

	// auto ex1 = quantt::contract(state,state,hamil);

	auto start = std::chrono::steady_clock::now();
	auto E0 = quantt::dmrg(hamil, state, options);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	print_MPS_dims(state);
	fmt::print("{} sites AFM heisenberg Energy per sites {:.10}. obtained in {} seconds\n", size, E0.to<double>() / size, elapsed_seconds.count());
	fmt::print("norm of the state: \n{}\n", contract(state, state));
	fmt::print("total energy in the state: \n{}\n", contract(state, state, hamil));

	return 0;
}