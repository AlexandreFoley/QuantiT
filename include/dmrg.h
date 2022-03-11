/*
 * File: dmrg.h
 * Project: QuantiT
 * File Created: Tuesday, 11th August 2020 9:46:51 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Tuesday, 11th August 2020 9:46:52 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * Licensed under GPL v3
 */

#ifndef E8650E72_8C05_4D74_98C7_61F4FD428B39
#define E8650E72_8C05_4D74_98C7_61F4FD428B39

#include "MPT.h"
#include "dmrg_logger.h"
#include "dmrg_options.h"
#include <cmath>
#include <limits>
#include <torch/torch.h>
#include "doctest/doctest_proxy.h"

namespace quantit
{

namespace
{
dmrg_default_logger dummy_logger;
}
/**
 * Apply the DMRG algorithm to solve the ground state of the input hamiltonian given as a MPO.
 * Uses the supplied MPS in_out_state as a starting point, and store the optimized MPS there.
 * The associated energy is the return value.
 */
torch::Tensor dmrg( MPO &hamiltonian, MPS &in_out_state, const dmrg_options &options,
                   dmrg_logger &logger = dummy_logger);
btensor dmrg( bMPO &hamiltonian, bMPS &in_out_state, const dmrg_options &options,
                   dmrg_logger &logger = dummy_logger);

/**
 * Apply the DMRG algorithm to solve the ground state of the input hamiltonian given as a MPO.
 * uses a random starting MPS with minimum_bond bond dimension.
 * return the ground state energy and optimized MPS.
 */
std::tuple<torch::Tensor, MPS> dmrg( MPO &hamiltonian, const dmrg_options &options,
                                    dmrg_logger &logger = dummy_logger);
std::tuple<btensor, bMPS> dmrg( bMPO &hamiltonian, any_quantity_cref state_constraint , const dmrg_options &options,
                                    dmrg_logger &logger = dummy_logger);

namespace details
{

btensor dmrg_impl(const bMPO &hamiltonian, const bMPT &two_sites_hamil, bMPS &in_out_state, const dmrg_options &options,
                  benv_holder &Env, dmrg_logger &logger);
torch::Tensor dmrg_impl(const MPO &hamiltonian, const MPT &twosites_hamil, MPS &in_out_state,
                        const dmrg_options &options, env_holder &Env, dmrg_logger &logger);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> eig2x2Mat(const torch::Tensor &a0, const torch::Tensor &a1,
                                                                  const torch::Tensor &b);
std::tuple<btensor, btensor, btensor> eig2x2Mat(const btensor &a0, const btensor &a1, const btensor &b);
torch::Tensor hamil2site_times_state(const torch::Tensor &state, const torch::Tensor &hamil, const torch::Tensor &Lenv,
                                     const torch::Tensor &Renv);
btensor hamil2site_times_state(const btensor &state, const btensor &hamil, const btensor &Lenv, const btensor &Renv);
MPT compute_2sitesHamil(const MPO &hamil);
bMPT compute_2sitesHamil(const bMPO &hamil);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> one_step_lanczos(const torch::Tensor &state,
                                                                                        const torch::Tensor &hamil,
                                                                                        const torch::Tensor &Lenv,
                                                                                        const torch::Tensor &Renv);
std::tuple<btensor, btensor, btensor, btensor> one_step_lanczos(const btensor &state, const btensor &hamil,
                                                                const btensor &Lenv, const btensor &Renv);
btensor edge_shape_prep(const btensor &tens,int64_t dim);
torch_shape edge_shape_prep(const torch_shape &tens, int64_t dim);
torch::Tensor trivial_edge(const torch::Tensor &lower_state, const torch::Tensor &Hamil, const torch::Tensor &upper_state,
                              int64_t index_low, int64_t index_op, int64_t index_up);
btensor trivial_edge(const btensor &lower_state, const btensor &Hamil, const btensor &upper_state, int64_t index_low,
                     int64_t index_op, int64_t index_up);
} // namespace details
qtt_TEST_CASE("btensor dmrg run test")
{	
	//TODO: make it so dmrg run with silly target cval (like 0 particles) behave in a more graceful (e.g. finnish quickly) manner than currently(crash)
	using cval = quantity<conserved::Z>;
	auto T = quantit::rand({{{1,cval(1)},{1,cval(-1)}}, {{3,cval(-1)},{2,cval(1)}}, {{1,cval(-1)},{1,cval(1)}}, {{3,cval(1)},{2,cval(-1)}}},cval(0));
	bMPO Hamil(5, T);
	dmrg_options opt;
	opt.maximum_iterations = 10;
	{
		Hamil[0] = Hamil[0].basic_create_view({0,-1,-1,-1},true);
		Hamil[Hamil.size() - 1] = Hamil[Hamil.size() - 1].basic_create_view({-1,-1, 0,-1},true);
	}
	qtt_REQUIRE(Hamil.check_ranks());
	btensor E;
	bMPS state;
	// std::tie(E,state) = dmrg(Hamil,opt);
	qtt_CHECK_NOTHROW(std::tie(E, state) = dmrg(Hamil, cval(1), opt));
	// fmt::print("E {}\n\n",E);
}
qtt_TEST_CASE("dmrg run test")
{
	auto T = torch::rand({2, 5, 2, 5});
	MPO Hamil(5, T);
	dmrg_options opt;
	opt.maximum_iterations = 10;
	{
		using namespace torch::indexing;
		Hamil[0] = Hamil[0].index({Slice(0, 1), Ellipsis});
		Hamil[Hamil.size() - 1] = Hamil[Hamil.size() - 1].index({Ellipsis, Slice(0, 1), Slice()});
	}
	torch::Tensor E;
	MPS state;
	// std::tie(E,state) = dmrg(Hamil,opt);
	qtt_CHECK_NOTHROW(std::tie(E, state) = dmrg(Hamil, opt));
}
qtt_TEST_CASE("2x2 eigen value problem")
{
	// setup: a random answer from which we construct a matrix
	auto angle = torch::rand({}, torch::kFloat64) * 2 * M_PI;
	auto E0 = torch::rand({}, torch::kFloat64) * -1;
	auto E1 = torch::rand({}, torch::kFloat64);
	auto psi00 = torch::zeros({}, torch::kFloat64);
	auto psi01 = torch::zeros({}, torch::kFloat64);
	psi00 = cos(angle);
	psi01 = sin(angle);
	auto psi10 = torch::zeros({}, torch::kFloat64);
	auto psi11 = torch::zeros({}, torch::kFloat64);
	psi10 = sin(angle);
	psi11 = -cos(angle);
	auto a0 = psi00.pow(2) * E0 + psi10.pow(2) * E1;  // matrix element
	auto a1 = psi01.pow(2) * E0 + psi11.pow(2) * E1;  // matrix element
	auto b = psi01 * psi00 * E0 + psi11 * psi10 * E1; // matrix element

	auto [T_E0, T_psi00, T_psi01] = details::eig2x2Mat(a0, a1, b);
	// for the phase gauge freedom.
	bool state_check = (torch::allclose(psi00, T_psi00) and torch::allclose(psi01, T_psi01)) or
	                   (torch::allclose(-1 * psi00, T_psi00) and torch::allclose(-1 * psi01, T_psi01));
	qtt_CHECK(state_check);
	qtt_CHECK(torch::allclose(T_E0, E0));
}

qtt_TEST_CASE("Btensor two sites MPO")
{
	torch::set_default_dtype(torch::scalarTypeToTypeMeta(
	    torch::kFloat64)); // otherwise the type promotion always goes to floats when promoting a tensor
	using namespace torch::indexing;
	using namespace details;
	auto l_ising = torch::zeros({3, 2, 3, 2});
	{
		//only the Sz spin model can be written in MPO form with conservation laws.
		auto acc = l_ising.accessor<double, 4>();
		acc[1][0][0][0] = acc[2][0][1][0] = 1;
		acc[1][1][0][1] = acc[2][1][1][1] = -1;
		//Identities
		acc[0][0][0][0] = acc[0][1][0][1] = 1;
		acc[2][0][2][0] = acc[2][1][2][1] = 1;
	}
	using cval = quantity<conserved::Z>;
	auto bl_ising = btensor({{{3,cval(0)}},{{1,cval(1)},{1,cval(-1)}},{{3,cval(0)}},{{1,cval(-1)},{1,cval(1)}}},cval(0));
	bl_ising = from_basic_tensor_like(bl_ising,l_ising);
	bMPO ising(2, bl_ising);
	ising[0] = ising[0].basic_create_view({2, -1,-1,-1},true);//preserve the rank when creating the view
	ising[1] = ising[1].basic_create_view({-1, -1, 0,-1},true);//preserve the rank when create the view.

	auto two_s_ising = details::compute_2sitesHamil(ising);

	bMPS rstate = random_bMPS(4,ising,cval(0));//edges have bond dimension 1. if you want a uniform bond dimenison, ask for a MPS longer than you need, and chop off the edges.

	// fmt::print("Norm2 \n{}\n",contract(rstate,rstate));
	// fmt::print("state\n{}\n{}\n",rstate[0].reshape({4}),rstate[1].reshape({4}));
	rstate[0] = rstate[0] / sqrt(contract(rstate, rstate)); // normalize
	// fmt::print("normed state\n{}\n{}\n",rstate[0].reshape({4}),rstate[1].reshape({4}));
	auto norm = contract(rstate,rstate);
	qtt_CHECK(allclose(norm, quantit::ones({},cval(0))));
	// fmt::print("Norm2 \n{}\n",contract(rstate,rstate));
	// fmt::print("MPS \n{}\n\n",squeeze(rstate[0])); 
	// fmt::print("{}\n\n",squeeze(rstate[1]));
	auto two_s_state = tensordot(rstate[0], rstate[1], {2}, {0});
	// fmt::print("{}\n\n",two_s_state);
	auto avg = tensordot(two_s_state, two_s_state.conj(), {0, 1, 2, 3}, {0, 1, 2, 3});
	// fmt::print("avg {}\n\n",avg);
	qtt_CHECK(allclose(avg,ones_like(avg) ));

	auto H_av = contract(rstate, rstate, ising);
	auto H_av_2s = squeeze(tensordot(two_s_state.conj(), tensordot(two_s_ising[0], two_s_state, {4, 5}, {1, 2}), {1, 2}, {1, 2}));
	auto Lenv = trivial_edge(two_s_state,two_s_ising[0],two_s_state.inverse_cvals(),0,0,0);
	auto Renv = trivial_edge(two_s_state,two_s_ising[0],two_s_state.inverse_cvals(),3,3,3);
	auto H_av_upd = tensordot(
	    two_s_state.conj(), quantit::details::hamil2site_times_state(two_s_state, two_s_ising[0], Lenv, Renv),
	    {0, 1, 2, 3}, {0, 1, 2, 3});
	// fmt::print("H_av {}\n\nH_av_2s {}\n\n H_av_upd {}\n\n",H_av,H_av_2s,H_av_upd);
	qtt_CHECK(allclose(H_av, H_av_2s));
	qtt_CHECK(allclose(H_av, H_av_upd));
	// fmt::print("Correct value: \n{}\n",H_av);
	// fmt::print("from two sites: \n{}\n",H_av_2s);
	// fmt::print("from two sites update: \n{}\n",H_av_upd);
	auto [Hpsi, a0, a1, b] = details::one_step_lanczos(two_s_state, two_s_ising[0], Lenv, Renv);
	// fmt::print("Ising Hamil: {} \n\n", squeeze(two_s_ising[0]).reshape({2}) );
	// fmt::print("state: {} \n\n", squeeze(two_s_state).reshape({}) );
	// fmt::print("one step lanczos: \n{}\n",a0);
	auto [E, o_coeff, n_coeff] = details::eig2x2Mat(a0, a1, b);
	qtt_CHECK(allclose(o_coeff.pow(2) + n_coeff.pow(2), ones_like(o_coeff)));
	auto psi_update = o_coeff * two_s_state + n_coeff * Hpsi;
	auto test = tensordot(psi_update.conj(), psi_update, {0, 1, 2, 3}, {0, 1, 2, 3});
	qtt_CHECK(allclose(test, ones_like(test)));
	test = tensordot(two_s_state.conj(), two_s_state, {0, 1, 2, 3}, {0, 1, 2, 3});
	qtt_CHECK(allclose(test, ones_like(test)));
	test =tensordot(Hpsi, two_s_state.conj(), {0, 1, 2, 3}, {0, 1, 2, 3});
	qtt_CHECK(allclose(test, zeros_like(test)));
	test = tensordot(Hpsi.conj(), Hpsi, {0, 1, 2, 3}, {0, 1, 2, 3});
	qtt_CHECK(allclose(test, zeros_like(test)));// the input random state is always an eigenstate (in a degenerate manifold), the model is that simple.
	auto a1_test =
	    tensordot(Hpsi.conj(), quantit::details::hamil2site_times_state(Hpsi, two_s_ising[0], Lenv, Renv),
	                     {0, 1, 2, 3}, {0, 1, 2, 3});
	qtt_CHECK(allclose(a1, a1_test));
	// fmt::print("a1 \n{}\na1_test\n{}\n",a1,a1_test);
	auto H_av_Pupd = tensordot(
	    psi_update.conj(), quantit::details::hamil2site_times_state(psi_update, two_s_ising[0], Lenv, Renv),
	    {0, 1, 2, 3}, {0, 1, 2, 3});
	// fmt::print("actual update energie: \n{}\n", H_av_Pupd);
	// fmt::print("predicted update energie: \n{}\n", E);
}
qtt_TEST_CASE("Two sites MPO")
{
	torch::set_default_dtype(torch::scalarTypeToTypeMeta(
	    torch::kFloat64)); // otherwise the type promotion always goes to floats when promoting a tensor
	using namespace torch::indexing;
	//Sx *Sx version
	auto l_ising = torch::zeros({3, 2, 3, 2});
	{
		auto acc = l_ising.accessor<double, 4>();
		//Sx
		acc[1][0][0][1] = acc[1][1][0][0] = 1;
		acc[2][0][1][1] = acc[2][1][1][0] = 1;

		//corner identities
		acc[0][0][0][0] = acc[0][1][0][1] = 1;
		acc[2][0][2][0] = acc[2][1][2][1] = 1;
	}
	MPO ising(2, l_ising);
	ising[0] = ising[0].index({Slice(2, 3), Ellipsis});
	ising[1] = ising[1].index({Slice(), Slice(), Slice(0, 1), Slice()});

	auto two_s_ising = details::compute_2sitesHamil(ising);

	MPS rstate(2, torch::rand({2, 2, 2}));
	rstate[0] = rstate[0].index({Slice(0, 1), Ellipsis});
	rstate[1] = rstate[1].index({Ellipsis, Slice(1, 2)});

	// fmt::print("Norm2 \n{}\n",contract(rstate,rstate));
	// fmt::print("state\n{}\n{}\n",rstate[0].reshape({4}),rstate[1].reshape({4}));
	rstate[0] = rstate[0] / torch::sqrt(contract(rstate, rstate)); // normalize
	// fmt::print("normed state\n{}\n{}\n",rstate[0].reshape({4}),rstate[1].reshape({4}));
	qtt_CHECK(torch::allclose(contract(rstate, rstate), torch::ones({})));
	// fmt::print("Norm2 \n{}\n",contract(rstate,rstate));
	auto two_s_state = torch::tensordot(rstate[0], rstate[1], {2}, {0});
	qtt_CHECK(torch::allclose(tensordot(two_s_state, two_s_state, {0, 1, 2, 3}, {0, 1, 2, 3}), torch::ones({})));

	auto H_av = contract(rstate, rstate, ising);
	auto H_av_2s = torch::squeeze(
	    torch::tensordot(two_s_state, torch::tensordot(two_s_ising[0], two_s_state, {4, 5}, {1, 2}), {1, 2}, {1, 2}));
	auto triv_env = torch::ones({1, 1, 1});
	auto H_av_upd = torch::tensordot(
	    two_s_state, quantit::details::hamil2site_times_state(two_s_state, two_s_ising[0], triv_env, triv_env),
	    {0, 1, 2, 3}, {0, 1, 2, 3});

	qtt_CHECK(torch::allclose(H_av, H_av_2s));
	qtt_CHECK(torch::allclose(H_av, H_av_upd));
	// fmt::print("Correct value: \n{}\n",H_av);
	// fmt::print("from two sites: \n{}\n",H_av_2s);
	// fmt::print("from two sites update: \n{}\n",H_av_upd);
	auto [Hpsi, a0, a1, b] = details::one_step_lanczos(two_s_state, two_s_ising[0], triv_env, triv_env);
	// fmt::print("one step lanczos: \n{}\n",a0);
	auto [E, o_coeff, n_coeff] = details::eig2x2Mat(a0, a1, b);
	qtt_CHECK(torch::allclose(o_coeff.pow(2) + n_coeff.pow(2), torch::ones({})));
	auto psi_update = o_coeff * two_s_state + n_coeff * Hpsi;
	qtt_CHECK(torch::allclose(tensordot(psi_update, psi_update, {0, 1, 2, 3}, {0, 1, 2, 3}), torch::ones({})));
	qtt_CHECK(torch::allclose(tensordot(two_s_state, two_s_state, {0, 1, 2, 3}, {0, 1, 2, 3}), torch::ones({})));
	qtt_CHECK(torch::allclose(tensordot(Hpsi, two_s_state, {0, 1, 2, 3}, {0, 1, 2, 3}), torch::zeros({})));
	qtt_CHECK(torch::allclose(tensordot(Hpsi, Hpsi, {0, 1, 2, 3}, {0, 1, 2, 3}), torch::ones({})));
	auto a1_test =
	    torch::tensordot(Hpsi, quantit::details::hamil2site_times_state(Hpsi, two_s_ising[0], triv_env, triv_env),
	                     {0, 1, 2, 3}, {0, 1, 2, 3});
	qtt_CHECK(torch::allclose(a1, a1_test));
	// fmt::print("a1 \n{}\na1_test\n{}\n",a1,a1_test);
	auto H_av_Pupd = torch::tensordot(
	    psi_update, quantit::details::hamil2site_times_state(psi_update, two_s_ising[0], triv_env, triv_env),
	    {0, 1, 2, 3}, {0, 1, 2, 3});
	// fmt::print("actual update energie: \n{}\n", H_av_Pupd);
	// fmt::print("predicted update energie: \n{}\n", E);
}
} // namespace quantit

#endif /* E8650E72_8C05_4D74_98C7_61F4FD428B39 */
