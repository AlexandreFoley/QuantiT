/*
 * File: dmrg.h
 * Project: quantt
 * File Created: Tuesday, 11th August 2020 9:46:51 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Tuesday, 11th August 2020 9:46:52 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#ifndef E8650E72_8C05_4D74_98C7_61F4FD428B39
#define E8650E72_8C05_4D74_98C7_61F4FD428B39

#include <torch/torch.h>
#include "MPT.h"
#include <limits>
#include <cmath>

#include "cond_doctest.h"

namespace quantt{

struct dmrg_options
{
	double cutoff;
	double convergence_criterion;
	size_t maximum_bond;
	size_t minimum_bond;
	size_t maximum_iterations;
	bool pytorch_gradient; //will default to off! I can't think of a situation where we might want to compute a gradient through DMRG, but who knows.

	//default values for constructors.
	// if a constructor doesn't require user input for some member, it use the values found in the following definition.
	constexpr static double def_cutoff = 1e-6;
	constexpr static double def_conv_crit = 1e-5;
	constexpr static size_t def_max_bond = std::numeric_limits<size_t>::max(); // a rather large number.
	constexpr static size_t def_min_bond = 4; //I have found that dmrg behave better if we prevent bond dimension from going to low.
	constexpr static size_t def_max_it = 1000;
	constexpr static bool def_pytorch_gradient = false;

	dmrg_options(double _cutoff,double _convergence_criterion): cutoff(_cutoff),convergence_criterion(_convergence_criterion),maximum_bond(def_max_bond),
	                                                            minimum_bond(def_min_bond),maximum_iterations(def_max_it),pytorch_gradient(def_pytorch_gradient) {}
	dmrg_options(size_t _max_bond,size_t _min_bond,size_t _max_iterations)
	: cutoff(def_cutoff), convergence_criterion(def_conv_crit), maximum_bond(_max_bond), minimum_bond(_min_bond),
	  maximum_iterations(_max_iterations), pytorch_gradient(def_pytorch_gradient) {}
	dmrg_options(double _cutoff,double _convergence_criterion,size_t _max_bond,size_t _min_bond,size_t _max_iterations,bool _pytorch_gradient = def_pytorch_gradient)
	: cutoff(_cutoff), convergence_criterion(_convergence_criterion), maximum_bond(_max_bond), minimum_bond(_min_bond),
	  maximum_iterations(_max_iterations), pytorch_gradient(_pytorch_gradient) {}
	dmrg_options():dmrg_options(def_cutoff,def_conv_crit) {}
	dmrg_options(const dmrg_options&) = default;
	dmrg_options(dmrg_options&&) = default;
	
	dmrg_options& operator=(const dmrg_options&) = default;
	dmrg_options& operator=(dmrg_options&&) = default;
};

/**
 * Apply the DMRG algorithm to solve the ground state of the input hamiltonian given as a MPO.
 * Uses the supplied MPS in_out_state as a starting point, and store the optimized MPS there.
 * The associated energy is the return value. This might change in the future to a status bundle.
 */
torch::Scalar dmrg(const MPO& hamiltonian, MPS& in_out_state,const dmrg_options& options);

/**
 * Apply the DMRG algorithm to solve the ground state of the input hamiltonian given as a MPO.
 * uses a random starting MPS with minimum_bond bond dimension.
 * return the ground state energy and optimized MPS.
 */
std::tuple<torch::Scalar,MPS> dmrg(const MPO& hamiltonian,const dmrg_options& options);

namespace details{
	torch::Scalar dmrg_impl(const MPO& hamiltonian,const MPT& twosites_hamil, MPS& in_out_state,const dmrg_options& options, env_holder& Env);
	std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> eig2x2Mat(torch::Tensor a0,torch::Tensor a1, torch::Tensor b);
}

TEST_CASE("dmrg run test")
{//only test that dmrg runs and finish as expected.

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
	CHECK_NOTHROW(std::tie(E,state) = dmrg(Hamil,opt));

}
TEST_CASE("2x2 eigen value problem")
{
	// setup: a random answer from which we construct a matrix
	auto angle = torch::rand({},torch::kFloat64)*2*M_PI;
	auto E0 = torch::rand({},torch::kFloat64)*-1;
	auto E1 = torch::rand({},torch::kFloat64);
	auto psi00 = torch::zeros({},torch::kFloat64);
	auto psi01 = torch::zeros({},torch::kFloat64);
	psi00=cos(angle); psi01=sin(angle);
	auto psi10 = torch::zeros({},torch::kFloat64);
	auto psi11 = torch::zeros({},torch::kFloat64);
	psi10= sin(angle); psi11 = -cos(angle);
	auto a0 = psi00.pow(2)*E0 + psi10.pow(2)*E1;//matrix element
	auto a1 = psi01.pow(2)*E0 + psi11.pow(2)*E1;//matrix element
	auto b = psi01*psi00*E0 + psi11*psi10*E1;//matrix element

	auto[T_E0,T_psi00,T_psi01] = details::eig2x2Mat(a0,a1,b);
	                                                                                     //for the phase gauge freedom.
	bool state_check = (torch::allclose(psi00,T_psi00) and torch::allclose(psi01,T_psi01)) or (torch::allclose(-1*psi00,T_psi00) and torch::allclose(-1*psi01,T_psi01));
	CHECK(state_check);
	CHECK(torch::allclose(T_E0,E0));
}

}//quantt

#endif /* E8650E72_8C05_4D74_98C7_61F4FD428B39 */
