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


int main()
{
	using namespace quantt;
	fmt::print("C++ standard version in use {}\n",__cplusplus);
	torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kFloat64)); //otherwise the type promotion always goes to floats when promoting a tensor
	torch::Device cuda_device(torch::kCPU); // default the cuda device to a cpu. small lie to keep the code working if there isn't one.
	if (torch::cuda::is_available())
	{
		fmt::print("CUDA is available!\n");
		cuda_device = torch::Device(torch::kCUDA); // set the cuda_device to the actual gpu if it would work
	}
	// auto hamil = quantt::Heisenberg(1,10);
	// quantt::dmrg_options options;
	// auto [E0,state] = quantt::dmrg(hamil,options);
	// fmt::print("10 sites AFM heisenberg Energy {}",E0.to<double>());
	
	MPO Hamil(5,torch::rand({2,5,2,5}));
	dmrg_options opt;
	opt.maximum_iterations = 10;
	{
		using namespace torch::indexing;
		Hamil[0] = Hamil[0].index({Slice(0,1),Ellipsis});
		Hamil[Hamil.size()-1] = Hamil[Hamil.size()-1].index({Ellipsis,Slice(0,1),Slice()});
	}
	torch::Scalar E;
	MPS state;
	// std::tie(E,state) = dmrg(Hamil,opt);
	(std::tie(E,state) = dmrg(Hamil,opt));

	return 0;
}