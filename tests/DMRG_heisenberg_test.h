/*
 * File: DMRG_heisenberg_test.h
 * Project: quantt
 * File Created: Wednesday, 19th August 2020 11:39:47 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Wednesday, 19th August 2020 11:41:35 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#ifndef E0106B85_7787_42FD_9E7E_47803E425A61
#define E0106B85_7787_42FD_9E7E_47803E425A61

#include <chrono>
#include "dmrg.h"
#include "models.h"
#include "cond_doctest.h"
#include "torch_formatter.h"

TEST_CASE("solving the heisenberg model")
{
	torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kFloat64));
	auto local_tens = torch::zeros({4,2,4});
	{
		auto acc = local_tens.accessor<double,3>();
		acc[0][0][0] = 1.;
		acc[0][1][1] = 1;
		acc[0][0][2] = 1/std::sqrt(2);
		acc[0][1][2] = 1/std::sqrt(2);
		acc[0][0][3] = 1/std::sqrt(2);
		acc[0][1][3] = -1/std::sqrt(2);
		acc[1][0][3] = 1.;
		acc[1][1][0] = 1;
		acc[1][0][1] = 1/std::sqrt(2);
		acc[1][1][1] = 1/std::sqrt(2);
		acc[1][0][2] = 1/std::sqrt(2);
		acc[1][1][2] = -1/std::sqrt(2);
		acc[2][0][2] = 1.;
		acc[2][1][3] = 1;
		acc[2][0][0] = 1/std::sqrt(2);
		acc[2][1][0] = 1/std::sqrt(2);
		acc[2][0][1] = 1/std::sqrt(2);
		acc[2][1][1] = -1/std::sqrt(2);
		acc[3][0][1] = 1.;
		acc[3][1][2] = 1;
		acc[3][0][3] = 1/std::sqrt(2);
		acc[3][1][3] = 1/std::sqrt(2);
		acc[3][0][0] = 1/std::sqrt(2);
		acc[3][1][0] = -1/std::sqrt(2);
	}
	SUBCASE("10 sites AFM")
	{	
		auto hamil = quantt::Heisenberg(1,10);
		quantt::MPS state(10,local_tens);
		{
			using namespace torch::indexing;
			state[0] = state[0].index({Slice(0,1),Ellipsis});
			state[9] = state[9].index({Ellipsis,Slice(0,1)});
		}
		quantt::dmrg_options options;
		auto start = std::chrono::steady_clock::now();
		auto E0 = quantt::dmrg(hamil,state,options);
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		fmt::print("10 sites AFM heisenberg Energy {}. obtained in {}\n",E0.to<double>(),elapsed_seconds.count());
	}
	SUBCASE("20 sites AFM")
	{
		auto hamil = quantt::Heisenberg(1,20);
		quantt::MPS state(20,local_tens);
		{
			using namespace torch::indexing;
			state[0] = state[0].index({Slice(0,1),Ellipsis});
			state[19] = state[19].index({Ellipsis,Slice(0,1)});
		}
		quantt::dmrg_options options;
		auto start = std::chrono::steady_clock::now();
		auto E0 = quantt::dmrg(hamil,state,options);
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		fmt::print("20 sites AFM heisenberg Energy {}. obtained in {}\n",E0.to<double>(),elapsed_seconds.count());
	}
	SUBCASE("50 sites AFM")
	{
		auto hamil = quantt::Heisenberg(1,50);
		quantt::MPS state(50,local_tens);
		{
			using namespace torch::indexing;
			state[0] = state[0].index({Slice(0,1),Ellipsis});
			state[49] = state[49].index({Ellipsis,Slice(0,1)});
		}
		quantt::dmrg_options options;
		auto start = std::chrono::steady_clock::now();
		auto E0 = quantt::dmrg(hamil,state,options);
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		fmt::print("50 sites AFM heisenberg Energy {}. obtained in {}\n",E0.to<double>(),elapsed_seconds.count());
	}
	SUBCASE("100 sites AFM")
	{
		auto hamil = quantt::Heisenberg(1,100);
		quantt::MPS state(100,local_tens);
		{
			using namespace torch::indexing;
			state[0] = state[0].index({Slice(0,1),Ellipsis});
			state[99] = state[99].index({Ellipsis,Slice(0,1)});
		}
		quantt::dmrg_options options;
		auto start = std::chrono::steady_clock::now();
		auto E0 = quantt::dmrg(hamil,state,options);
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		fmt::print("100 sites AFM heisenberg Energy {}. obtained in {}\n",E0.to<double>(),elapsed_seconds.count());
	}
// 	SUBCASE("1000 sites AFM")
// 	{
// 		auto hamil = quantt::Heisenberg(1,1000);
// 		quantt::MPS state(1000,local_tens);
// 		{
// 			using namespace torch::indexing;
// 			state[0] = state[0].index({Slice(0,1),Ellipsis});
// 			state[999] = state[999].index({Ellipsis,Slice(0,1)});
// 		}
// 		quantt::dmrg_options options;
// 		auto start = std::chrono::steady_clock::now();
// 		auto E0 = quantt::dmrg(hamil,state,options);
// 		auto end = std::chrono::steady_clock::now();
// 		std::chrono::duration<double> elapsed_seconds = end - start;
// 		fmt::print("1000 sites AFM heisenberg Energy {}. obtained in {}\n",E0.to<double>(),elapsed_seconds.count());
// 	}
}


#endif /* E0106B85_7787_42FD_9E7E_47803E425A61 */
