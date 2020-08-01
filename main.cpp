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
#include "include/tens_formatter.h"
#include "fmt/ostream.h"

using tens = torch::Tensor;
std::tuple<tens,tens,tens,tens> generate_tens()
{
	torch::Tensor c_up = torch::zeros({4,4},torch::kInt8);
	auto c_dn = torch::zeros({4,4},torch::kInt8);
	auto F = torch::zeros({4,4},torch::kInt8);
	auto id = torch::zeros({4,4},torch::kInt8);

	auto Acc_cup = c_up.accessor<int8_t,2>();
	Acc_cup[0][1] = 1;
	Acc_cup[2][3] = 1;
	auto Acc_cdn = c_dn.accessor<int8_t,2>();
	Acc_cdn[0][2] = 1;
	Acc_cdn[1][3] = -1;
	auto Acc_F = F.accessor<int8_t,2>();
	Acc_F[0][0] = Acc_F[3][3] = 1;
	Acc_F[1][1] = Acc_F[2][2] = -1;
	auto Acc_id = id.accessor<int8_t,2>();
	Acc_id[0][0] = Acc_id[1][1] = Acc_id[2][2] = Acc_id[3][3] = 1;
	return std::make_tuple(c_up,c_dn,F,id);
}

int main()
{
	fmt::print("C++ standard version in use {}",__cplusplus);
	torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kFloat64)); //otherwise the type promotion always goes to floats when promoting a tensor
	torch::Device cuda_device(torch::kCPU); // default the cuda device to a cpu. small lie to keep the code working if there isn't one.
	if (torch::cuda::is_available())
	{
		fmt::print("CUDA is available!\n");
		cuda_device = torch::Device(torch::kCUDA); // set the cuda_device to the actual gpu if it would work
	}
	const auto [c_up,c_dn,F,id4] = generate_tens();
	auto A = torch::rand({5,10},torch::kComplexDouble);
	auto B = torch::rand({5,10},torch::kComplexDouble);
	auto cc_up = A-B;
	auto [u,d,v] = cc_up.svd();

	

	auto reconstruct = torch::matmul(torch::matmul(u,torch::diag_embed(d.to(torch::kComplexDouble)) ), v.t() );

	using namespace torch::indexing;
	// auto inds = std::initializer_list<at::indexing::TensorIndex>({0,Slice(),0,Slice()});
	// MPO.index_put_(inds,5*c_up);
	fmt::print("real\n reconstructed:\n{}\noriginal:\n{}\n",torch::real(reconstruct),torch::real(cc_up));
	fmt::print("imag\n reconstructed:\n{}\noriginal:\n{}\n",torch::imag(reconstruct),torch::imag(cc_up));

	
	return 0;
}