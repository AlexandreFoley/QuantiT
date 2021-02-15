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

#include "dmrg.h"
#include "doctest/doctest_proxy.h"
#include "models.h"
#include "torch_formatter.h"
#include <chrono>

class dmrg_log_final final: public quantt::dmrg_logger
{
	public:
	size_t it_num;
	size_t middle_bond_dim;
	

	void log_step(size_t it) override {it_num = it;}
	void log_energy(torch::Tensor) override {}
	void log_bond_dims(const quantt::MPS& mps) override 
	{
		auto pos = mps.size()/2;
		middle_bond_dim = std::max(mps[pos].sizes()[0],mps[pos].sizes()[2]);
	}
	void it_log_all(size_t, torch::Tensor,const quantt::MPS&) override {}

};
class dmrg_log_sweeptime final: public quantt::dmrg_logger
{
	public:
	size_t it_num;
	size_t middle_bond_dim;
	std::chrono::steady_clock::time_point then;
	std::vector<double> time_list;
	std::vector<size_t> bond_list;

	void log_step(size_t it) override {it_num = it;}
	void log_energy(torch::Tensor) override {}

	void init(const quantt::dmrg_options& opt) override
	{
		then = std::chrono::steady_clock::now();
		time_list = std::vector<double>(opt.maximum_iterations);
		bond_list = std::vector<size_t>(opt.maximum_iterations);

	}

	void log_bond_dims(const quantt::MPS& mps) override 
	{
		auto pos = mps.size()/2;
		middle_bond_dim = std::max(mps[pos].sizes()[0],mps[pos].sizes()[2]);
	}
	void it_log_all(size_t it, torch::Tensor E0,const quantt::MPS& mps) override 
	{
		auto now = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed_seconds = now - then;
		then = now;
		log_bond_dims(mps);
		bond_list[it] = middle_bond_dim;
		time_list[it] = elapsed_seconds.count();
		log_step(it);
		log_energy(E0);
	}

};

qtt_TEST_CASE("solving the heisenberg model")
{
	torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kFloat64));
	auto local_tens = torch::rand({4, 2, 4});
	int J = -1.;
	std::string print_string = "{} sites AFM heisenberg Energy per sites {:.15}. obtained in {} seconds\n";
	dmrg_log_final logger;
	auto Heisen_afm_test = [&](size_t size) {
		auto hamil = quantt::Heisenberg(torch::tensor(J), size);
		quantt::MPS state(size, local_tens);
		{
			using namespace torch::indexing;
			state[0] = state[0].index({Slice(0, 1), Ellipsis});
			state[size - 1] = state[size - 1].index({Ellipsis, Slice(0, 1)});
		}
		quantt::dmrg_options options;
		auto start = std::chrono::steady_clock::now();
		auto E0 = quantt::dmrg(hamil, state, options,logger);
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		fmt::print(print_string, size, E0.to<double>() / size, elapsed_seconds.count());
		fmt::print("Obtained in {} iterations. Bond dimension at middle of MPS: {}.\n",logger.it_num,logger.middle_bond_dim);
	};
	qtt_SUBCASE("2 sites AFM")
	{
		constexpr size_t size = 2;
		Heisen_afm_test(size);
	}
	qtt_SUBCASE("10 sites AFM")
	{
		constexpr size_t size = 10;
		Heisen_afm_test(size);
	}
	qtt_SUBCASE("20 sites AFM")
	{
		Heisen_afm_test(20);
	}
	qtt_SUBCASE("50 sites AFM")
	{
		Heisen_afm_test(50);
	}
	qtt_SUBCASE("100 sites AFM")
	{
		Heisen_afm_test(100);
	}
	qtt_SUBCASE("ITensors julia comparison")
	{	
		auto init_num_threads = torch::get_num_threads();
		torch::set_num_threads(1);
		constexpr size_t size = 100;
		auto hamil = quantt::Heisenberg(torch::tensor(J), size);
		dmrg_log_sweeptime logger;
		quantt::MPS state(size);
		int p = 0;
		for (auto& site:state)
		{//antiferromagnetic slatter determinant.
			using namespace torch::indexing;
			site = torch::rand({8,2,8});
			site.index_put_({0,p,0},1);
			p = -p+1;
			site.index_put_({0,p,0},0);
		}
		{
			using namespace torch::indexing;
			state[0] = state[0].index({Slice(0, 1), Ellipsis});
			state[size - 1] = state[size - 1].index({Ellipsis, Slice(0, 1)});
		}
		quantt::dmrg_options options;
		options.convergence_criterion = 0;
		options.maximum_iterations=50;
		options.maximum_bond=10;
		options.cutoff=1e-11;
		auto E00 = quantt::dmrg(hamil, state, options);
		options.maximum_iterations=20;
		options.maximum_bond=130;
		options.cutoff=1e-11;
		auto start = std::chrono::steady_clock::now();
		auto E0 = quantt::dmrg(hamil, state, options,logger);
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		std::string print_string = "julia Itensor comparison: {} sites AFM heisenberg Energy per sites {:.15}. obtained in {} seconds\n";
		fmt::print(print_string, size, E0.to<double>() / size, elapsed_seconds.count());
		fmt::print("Obtained in {} iterations. Bond dimension at middle of MPS: {}.\n",logger.it_num,logger.middle_bond_dim);
		fmt::print("time in seconds for each sweeps: {}\n",logger.time_list);
		fmt::print("bond dimension after each sweeps: {}\n",logger.bond_list);
		torch::set_num_threads(init_num_threads);
	}
	qtt_SUBCASE("DMRjulia comparison")
	{	
		auto init_num_threads = torch::get_num_threads();
		torch::set_num_threads(1);
		constexpr size_t size = 100;
		auto hamil = quantt::Heisenberg(torch::tensor(J), size);
		dmrg_log_sweeptime logger;
		quantt::MPS state(size);
		int p = 0;
		for (auto& site:state)
		{//antiferromagnetic slatter determinant.
			using namespace torch::indexing;
			site = torch::zeros({1,2,1});
			site.index_put_({0,p,0},1);
			p = -p+1;
		}
		quantt::dmrg_options options;
		options.convergence_criterion = 0;
		options.maximum_iterations=20;
		options.maximum_bond=45;
		options.cutoff=1e-9;
		auto start = std::chrono::steady_clock::now();
		auto E0 = quantt::dmrg(hamil, state, options,logger);
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		std::string print_string = "DMRjulia comparison: {} sites AFM heisenberg Energy per sites {:.15}. obtained in {} seconds\n";
		fmt::print(print_string, size, E0.to<double>() / size, elapsed_seconds.count());
		fmt::print("Obtained in {} iterations. Bond dimension at middle of MPS: {}.\n",logger.it_num,logger.middle_bond_dim);
		fmt::print("time in seconds for each sweeps: {}\n",logger.time_list);
		fmt::print("bond dimension after each sweeps: {}\n",logger.bond_list);
		torch::set_num_threads(init_num_threads);
	}
	// 	qtt_SUBCASE("1000 sites AFM")
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
