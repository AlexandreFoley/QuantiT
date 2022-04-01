/*
 * File: DMRG_heisenberg_test.h
 * Project: QuantiT
 * File Created: Wednesday, 19th August 2020 11:39:47 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Wednesday, 19th August 2020 11:41:35 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * Licensed under GPL v3
 */

#ifndef E0106B85_7787_42FD_9E7E_47803E425A61
#define E0106B85_7787_42FD_9E7E_47803E425A61

#include "dmrg.h"
#include "doctest/doctest_proxy.h"
#include "models.h"
#include "torch_formatter.h"
#include <chrono>

auto Heisen_afm_test_bt(size_t size)
{
	using cval = quantit::quantity<quantit::conserved::Z>;
	quantit::btensor local_heisenberg_shape({{{1, cval(1)}, {1, cval(-1)}}},
	                                       cval(0));
	int J = -1.;
	fmt::print("{:=^80}\n", "Btensors");
	std::string print_string = "{} sites AFM heisenberg Energy per sites {:.15}. obtained in {} seconds\n";
	quantit::dmrg_log_simple logger;
	auto hamil = quantit::Heisenberg(torch::tensor(J), size, local_heisenberg_shape);
	hamil.coalesce();
	quantit::bMPS state = quantit::random_bMPS(4, hamil, cval(size % 2), {}, 0);
	state[0] /= sqrt(contract(state, state));
	state.move_oc(state.size() - 1);
	state.move_oc(0);
	quantit::dmrg_options options;
	auto start = std::chrono::steady_clock::now();
	auto E0 = quantit::dmrg(hamil, state, options, logger);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	fmt::print(print_string, size, E0.item().to<double>() / size, elapsed_seconds.count());
	// the reported energy per site is wrong. result with contract is ok. The bond dimension is on the high
	// side, convergence anomalously slow. (All those symptoms could be the same disease)
	fmt::print("Obtained in {} iterations. Bond dimension at middle of MPS: {}.\n", logger.it_num,
	           logger.middle_bond_dim);
	// if (size <= 2)
	// {
	// 	auto H2 = squeeze(tensordot(hamil[0], hamil[1], {2}, {0})).permute_({0, 2, 1, 3}).reshape({2});
	// 	auto S2 = squeeze(tensordot(state[0], state[1], {2}, {0})).reshape({});
	// 	fmt::print("Hamiltonian: {}\n\n state {}\n\n", H2, S2);

	// 	fmt::print("H0 {}\n\n,H1 {}\n\n",hamil[0],hamil[1]);
	// }
	// if (size >=4)	fmt::print("bulk hamiltonian tensor\n {}\n\n", hamil[3].to_dense().permute({0,2,1,3}));
	fmt::print("\nstate norm {}\n", contract(state, state).item().toDouble());
	fmt::print("E {}\n\n", contract(state, state, hamil).item().toDouble());
};
auto Heisen_afm_test_tt(size_t size)
{
	auto local_tens = torch::rand({4, 2, 4});
	local_tens /= sqrt(tensordot(local_tens,local_tens,{0,1,2},{0,1,2}));
	fmt::print("random state square norm: {}", tensordot(local_tens,local_tens,{0,1,2},{0,1,2}));
	int J = -1.;
	fmt::print("{:=^80}\n", "torch tensors");
	std::string print_string = "{} sites AFM heisenberg Energy per sites {:.15}. obtained in {} seconds\n";
	quantit::dmrg_log_simple logger;
	auto hamil = quantit::Heisenberg(torch::tensor(J), size);
	quantit::MPS state(size, local_tens);
	{
		using namespace torch::indexing;
		state[0] = state[0].index({Slice(0, 1), Ellipsis});
		state[size - 1] = state[size - 1].index({Ellipsis, Slice(0, 1)});
	}
	// fmt::print("MPS square norm {}\n", contract(state,state));
	state[0] /= sqrt(tensordot(state[0], state[0].conj(), {0, 1, 2}, {0, 1, 2}));
	state.move_oc(state.size() - 1);
	state.move_oc(0);
	state[0] /= sqrt(tensordot(state[0], state[0].conj(), {0, 1, 2}, {0, 1, 2}));
	// fmt::print("MPS square norm {}\n", contract(state,state));
	quantit::dmrg_options options;
	auto start = std::chrono::steady_clock::now();
	auto E0 = quantit::dmrg(hamil, state, options, logger);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	fmt::print(print_string, size, E0.item().to<double>() / size, elapsed_seconds.count());
	fmt::print("Obtained in {} iterations. Bond dimension at middle of MPS: {}.\n", logger.it_num,
	           logger.middle_bond_dim);
	// if (size <= 2)
	// {
	// 	auto H2 = squeeze(tensordot(hamil[0], hamil[1], {2}, {0})).permute({0, 2, 1, 3}).reshape({4, 4});
	// 	auto S2 = squeeze(tensordot(state[0], state[1], {2}, {0})).reshape({4});
	// 	fmt::print("Hamiltonian:\n {}\n\n state\n {}\n\n", H2, S2);
	// 	fmt::print("H0 {}\n\n,H1 {}\n\n",hamil[0],hamil[1]);
	// }
	// if (size >=4)	fmt::print("bulk hamiltonian tensor\n {}\n\n", hamil[3].permute({0,2,1,3}));
};
qtt_TEST_CASE("Solving the heisenberg model")
{
	torch::InferenceMode Inference_guard;
	torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kFloat64));
	qtt_SUBCASE("With Btensors")
	{
		qtt_SUBCASE("2 sites AFM")
		{
			fmt::print("2 sites\n");
			constexpr size_t size = 2;
			Heisen_afm_test_bt(size);
		}
		qtt_SUBCASE("3 sites AFM")
		{
			fmt::print("3 sites\n");
			constexpr size_t size = 3;
			Heisen_afm_test_bt(size);
		}
		qtt_SUBCASE("4 sites AFM")
		{
			fmt::print("4 sites\n");
			constexpr size_t size = 4;
			Heisen_afm_test_bt(size);
		}
		qtt_SUBCASE("10 sites AFM")
		{
			constexpr size_t size = 10;
			Heisen_afm_test_bt(size);
		}
		// qtt_SUBCASE("20 sites AFM")
		// {
		// 	Heisen_afm_test_bt(20);
		// }
		qtt_SUBCASE("50 sites AFM") { Heisen_afm_test_bt(50); }
	}
	qtt_SUBCASE("with torch tensors")
	{
		// qtt_SUBCASE("2 sites AFM")
		// {
		// 	constexpr size_t size = 2;
		// 	Heisen_afm_test_tt(size);
		// }
		// qtt_SUBCASE("3 sites AFM")
		// {
		// 	constexpr size_t size = 3;
		// 	Heisen_afm_test_tt(size);
		// }
		// qtt_SUBCASE("4 sites AFM")
		// {
		// 	constexpr size_t size = 4;
		// 	Heisen_afm_test_tt(size);
		// }
		// qtt_SUBCASE("10 sites AFM")
		// {
		// 	constexpr size_t size = 10;
		// 	Heisen_afm_test_tt(size);
		// }
		// qtt_SUBCASE("20 sites AFM")
		// {
		// 	Heisen_afm_test(20);
		// }
		qtt_SUBCASE("50 sites AFM") { Heisen_afm_test_tt(50); }
		// qtt_SUBCASE("100 sites AFM")
		// {
		// 	Heisen_afm_test_tt(100);
		// }
		// qtt_SUBCASE("ITensors julia comparison")
		// {
		// 	auto init_num_threads = torch::get_num_threads();
		// 	torch::set_num_threads(1);
		// 	constexpr size_t size = 100;
		// 	auto hamil = quantit::Heisenberg(torch::tensor(J), size);
		// 	dmrg_log_sweeptime logger;
		// 	quantit::MPS state(size);
		// 	int p = 0;
		// 	for (auto& site:state)
		// 	{//antiferromagnetic slatter determinant.
		// 		using namespace torch::indexing;
		// 		site = torch::rand({8,2,8});
		// 		site.index_put_({0,p,0},1);
		// 		p = -p+1;
		// 		site.index_put_({0,p,0},0);
		// 	}
		// 	{
		// 		using namespace torch::indexing;
		// 		state[0] = state[0].index({Slice(0, 1), Ellipsis});
		// 		state[size - 1] = state[size - 1].index({Ellipsis, Slice(0, 1)});
		// 	}
		// 	quantit::dmrg_options options;
		// 	options.convergence_criterion = 0;
		// 	options.maximum_iterations=50;
		// 	options.maximum_bond=10;
		// 	options.cutoff=1e-11;
		// 	auto E00 = quantit::dmrg(hamil, state, options);
		// 	options.maximum_iterations=20;
		// 	options.maximum_bond=130;
		// 	options.cutoff=1e-11;
		// 	auto start = std::chrono::steady_clock::now();
		// 	auto E0 = quantit::dmrg(hamil, state, options,logger);
		// 	auto end = std::chrono::steady_clock::now();
		// 	std::chrono::duration<double> elapsed_seconds = end - start;
		// 	std::string print_string = "julia Itensor comparison: {} sites AFM heisenberg Energy per sites {:.15}.
		// obtained in {} seconds\n"; 	fmt::print(print_string, size, E0.item().to<double>() / size,
		// elapsed_seconds.count()); 	fmt::print("Obtained in {} iterations. Bond dimension at middle of MPS:
		// {}.\n",logger.it_num,logger.middle_bond_dim); 	fmt::print("time in seconds for each sweeps:
		// {}\n",logger.time_list); 	fmt::print("bond dimension after each sweeps: {}\n",logger.bond_list);
		// 	torch::set_num_threads(init_num_threads);
		// }
		// qtt_SUBCASE("DMRjulia comparison")
		// {
		// 	auto init_num_threads = torch::get_num_threads();
		// 	torch::set_num_threads(1);
		// 	constexpr size_t size = 100;
		// 	auto hamil = quantit::Heisenberg(torch::tensor(J), size);
		// 	dmrg_log_sweeptime logger;
		// 	quantit::MPS state(size);
		// 	int p = 0;
		// 	for (auto& site:state)
		// 	{//antiferromagnetic slatter determinant.
		// 		using namespace torch::indexing;
		// 		site = torch::zeros({1,2,1});
		// 		site.index_put_({0,p,0},1);
		// 		p = -p+1;
		// 	}
		// 	quantit::dmrg_options options;
		// 	options.convergence_criterion = 0;
		// 	options.maximum_iterations=20;
		// 	options.maximum_bond=45;
		// 	options.cutoff=1e-9;
		// 	auto start = std::chrono::steady_clock::now();
		// 	auto E0 = quantit::dmrg(hamil, state, options,logger);
		// 	auto end = std::chrono::steady_clock::now();
		// 	std::chrono::duration<double> elapsed_seconds = end - start;
		// 	std::string print_string = "DMRjulia comparison: {} sites AFM heisenberg Energy per sites {:.15}. obtained
		// in {} seconds\n"; 	fmt::print(print_string, size, E0.item().to<double>() / size, elapsed_seconds.count());
		// 	fmt::print("Obtained in {} iterations. Bond dimension at middle of MPS:
		// {}.\n",logger.it_num,logger.middle_bond_dim); 	fmt::print("time in seconds for each sweeps:
		// {}\n",logger.time_list); 	fmt::print("bond dimension after each sweeps: {}\n",logger.bond_list);
		// 	torch::set_num_threads(init_num_threads);
		// }
		// 	qtt_SUBCASE("1000 sites AFM")
		// 	{
		// 		auto hamil = quantit::Heisenberg(1,1000);
		// 		quantit::MPS state(1000,local_tens);
		// 		{
		// 			using namespace torch::indexing;
		// 			state[0] = state[0].index({Slice(0,1),Ellipsis});
		// 			state[999] = state[999].index({Ellipsis,Slice(0,1)});
		// 		}
		// 		quantit::dmrg_options options;
		// 		auto start = std::chrono::steady_clock::now();
		// 		auto E0 = quantit::dmrg(hamil,state,options);
		// 		auto end = std::chrono::steady_clock::now();
		// 		std::chrono::duration<double> elapsed_seconds = end - start;
		// 		fmt::print("1000 sites AFM heisenberg Energy {}. obtained in
		// {}\n",E0.to<double>(),elapsed_seconds.count());
		// 	}
	}
}

#endif /* E0106B85_7787_42FD_9E7E_47803E425A61 */
