/*
 * File: dmrg_options.h
 * Project: quantt
 * File Created: Friday, 29th October 2021 4:17:42 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * Copyright (c) 2021 Alexandre Foley
 * All rights reserved
 */

#ifndef INCLUDE_DMRG_OPTIONS_H
#define INCLUDE_DMRG_OPTIONS_H
namespace quantt
{

struct dmrg_options
{
	double cutoff;
	double convergence_criterion;
	size_t maximum_bond;
	size_t minimum_bond;
	size_t maximum_iterations;
	bool state_gradient; // will default to off! I can't think of a situation where we might want to compute a
	bool hamil_gradient; // will default to off! I can't think of a situation where we might want to compute a
	                       // gradient through DMRG, but who knows.

	// default values for constructors.
	// if a constructor doesn't require user input for some member, it use the values found in the following definition.
	constexpr static double def_cutoff = 1e-6;
	constexpr static double def_conv_crit = 1e-5;
	constexpr static size_t def_max_bond = std::numeric_limits<size_t>::max(); // a rather large number.
	constexpr static size_t def_min_bond =
	    4; // I have found that dmrg behave better if we prevent bond dimension from going too low.
	constexpr static size_t def_max_it = 1000;
	constexpr static bool def_pytorch_gradient = false;

	dmrg_options(double _cutoff, double _convergence_criterion)
	    : cutoff(_cutoff), convergence_criterion(_convergence_criterion), maximum_bond(def_max_bond),
	      minimum_bond(def_min_bond), maximum_iterations(def_max_it), state_gradient(def_pytorch_gradient), hamil_gradient(def_pytorch_gradient)
	{
	}
	dmrg_options(size_t _max_bond, size_t _min_bond, size_t _max_iterations)
	    : cutoff(def_cutoff), convergence_criterion(def_conv_crit), maximum_bond(_max_bond), minimum_bond(_min_bond),
	      maximum_iterations(_max_iterations), state_gradient(def_pytorch_gradient), hamil_gradient(def_pytorch_gradient)
	{
	}
	dmrg_options(double _cutoff, double _convergence_criterion, size_t _max_bond, size_t _min_bond,
	             size_t _max_iterations, bool _state_gradient = def_pytorch_gradient,bool _hamil_gradient = def_pytorch_gradient)
	    : cutoff(_cutoff), convergence_criterion(_convergence_criterion), maximum_bond(_max_bond),
	      minimum_bond(_min_bond), maximum_iterations(_max_iterations), state_gradient(_state_gradient), hamil_gradient(_hamil_gradient)
	{
	}
	dmrg_options() : dmrg_options(def_cutoff, def_conv_crit) {}
	dmrg_options(const dmrg_options &) = default;
	dmrg_options(dmrg_options &&) = default;

	dmrg_options &operator=(const dmrg_options &) = default;
	dmrg_options &operator=(dmrg_options &&) = default;
};
}


#endif // INCLUDE_DMRG_OPTIONS_H
