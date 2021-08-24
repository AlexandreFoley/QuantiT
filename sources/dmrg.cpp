/*
 * File: dmrg.cpp
 * Project: quantt
 * File Created: Tuesday, 11th August 2020 9:48:36 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Tuesday, 11th August 2020 9:48:36 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#include "dmrg.h"
#include "LinearAlgebra.h"
#include "numeric.h"
#include "torch_formatter.h"
#include <fmt/core.h>
#include <random>
namespace quantt
{

using namespace details;

template <class X>
struct env_holder_impl
{
	static_assert(std::is_same_v<X, MPT> or std::is_same_v<X, bMPT>,
	              "must be either a MPT of basic tensor or block tensors");
	using Tens = typename X::Tens;
	X env;
	Tens &operator[](int64_t i) { return env[i + 1]; }
	const Tens &operator[](int64_t i) const { return env[i + 1]; }
};
class env_holder : public env_holder_impl<MPT>
{
}; // you might think a "using" statement would be cleaner, but
   // "using" statement don't mix well with forward declaration, and since this class is so simple this is ok.
class benv_holder : public env_holder_impl<bMPT>
{
};

torch::Tensor dmrg_impl(const MPO &hamiltonian, const MPT &two_site_hamil, MPS &in_out_state,
                        const dmrg_options &options, env_holder &Env);
btensor dmrg_impl(const bMPO &hamiltonian, const bMPT &two_site_hamil, bMPS &in_out_state, const dmrg_options &options,
                  benv_holder &Env);
env_holder generate_env(const MPO &hamiltonian, const MPS &in_out_state);
benv_holder generate_env(const bMPO &hamiltonian, const bMPS &in_out_state);
torch::Tensor compute_left_env(const torch::Tensor &Hamil, const torch::Tensor &MPS, const torch::Tensor &left_env);
btensor compute_left_env(const btensor &Hamil, const btensor &MPS, const btensor &left_env);
torch::Tensor compute_right_env(const torch::Tensor &Hamil, const torch::Tensor &MPS, const torch::Tensor &left_env);
btensor compute_right_env(const btensor &Hamil, const btensor &MPS, const btensor &left_env);
std::tuple<btensor, btensor> two_sites_update(const btensor &state, const btensor &hamil,
                                              const btensor &Left_environment, const btensor &Right_environment);
std::tuple<torch::Tensor, torch::Tensor> two_sites_update(const torch::Tensor &state, const torch::Tensor &hamil,
                                                          const torch::Tensor &Left_environment,
                                                          const torch::Tensor &Right_environment);

torch::Tensor dmrg(const MPO &hamiltonian, MPS &in_out_state, const dmrg_options &options, dmrg_logger &logger)
{
	if (!options.pytorch_gradient)
	{
		torch::NoGradGuard Gradientdisabled; // Globally (but Thread local (that's a problem)) disable gradient
		                                     // computation while this object exists.
		auto Env = generate_env(hamiltonian, in_out_state);
		auto TwositesH = compute_2sitesHamil(hamiltonian);
		return details::dmrg_impl(hamiltonian, TwositesH, in_out_state, options, Env, logger);
	} // Gradientdisabled gets destroyed here, gradient computation status is restore to what it was before.
	else
	{
		auto Env = generate_env(hamiltonian, in_out_state);
		auto TwositesH = compute_2sitesHamil(hamiltonian);
		return details::dmrg_impl(hamiltonian, TwositesH, in_out_state, options, Env, logger);
	}
}

std::tuple<torch::Tensor, MPS> dmrg(const MPO &hamiltonian, const dmrg_options &options, dmrg_logger &logger)
{
	using namespace torch::indexing;
	auto length = hamiltonian.size();
	auto out_mps = random_MPS(options.minimum_bond, hamiltonian);
	auto E0 = dmrg(hamiltonian, out_mps, options);
	return std::make_tuple(E0, out_mps);
}

template <class T>
auto sweep(MPS &state, T update, int step, size_t Nstep, size_t right_edge, size_t left_edge = 0)
{
	torch::Tensor E0;
	// fmt::print("sweep!\n");
	for (size_t i = 0; i < Nstep; ++i)
	{
		E0 = update(state, step);
		auto &oc = state.orthogonality_center;
		// fmt::print("i {} oc {} step {} || E \n {}\n ",i,oc,step,E0);
		step *= 1 - 2 * ((oc == left_edge) or (oc == right_edge)); // reverse the step direction if we're on the edge.
	}
	// fmt::print("\n");
	return std::make_tuple(E0, step);
}

struct dmrg_2sites_update
{
	const MPO &hamil;
	const MPT &twosite_hamil;
	size_t &oc;
	env_holder &Env;
	const dmrg_options &options;

	dmrg_2sites_update(const MPO &_hamil, const MPT &_twosites_hamil, size_t &_oc, env_holder &_Env,
	                   const dmrg_options &_options)
	    : hamil(_hamil), twosite_hamil(_twosites_hamil), oc(_oc), Env(_Env), options(_options)
	{
	}

	torch::Tensor operator()(MPS &state, int step)
	{
		bool forward = step == 1;
		torch::Tensor E0;
		auto local_state = torch::tensordot(state[oc], state[oc + 1], {2}, {0});
		std::tie(E0, local_state) = two_sites_update(local_state, twosite_hamil[oc], Env[oc - 1], Env[oc + 2]);
		auto [u, d, v] = quantt::svd(local_state, 2, options.cutoff, options.minimum_bond, options.maximum_bond);
		d /= sqrt(sum(d.pow(2)));
		if (forward)
		{
			// the orthogonality center was at oc
			state[oc] = u;
			state[oc + 1] = torch::tensordot(torch::diag(d), v, {1}, {2});
			Env[oc] = compute_left_env(hamil[oc], state[oc], Env[oc - 1]);
		}
		else
		{
			// the orthognality center was at oc+1
			state[oc] = torch::tensordot(u, torch::diag(d), {2}, {0});
			state[oc + 1] = v.permute({2, 0, 1});
			Env[oc + 1] = compute_right_env(hamil[oc + 1], state[oc + 1], Env[oc + 2]);
		}
		// fmt::print("full norm: \n{}\n",contract(sta6te,state));
		// fmt::print("full E: \n{}\n",contract(state,state,hamil));
		oc += step;
		return E0;
	}
};

btensor details::dmrg_impl(const bMPO &hamiltonian, const bMPT &two_sites_hamil, bMPS &in_out_state,
                           const dmrg_options &options, benv_holder &Env, dmrg_logger &logger)
{
	btensor E0 = quantt::full({}, hamiltonian[0].selection_rule->neutral(), 100000.0, hamiltonian[0].options());
	return btensor();
}
/**
 * The actual implementation.
 */
torch::Tensor details::dmrg_impl(const MPO &hamiltonian, const MPT &twosites_hamil, MPS &in_out_state,
                                 const dmrg_options &options, env_holder &Env, dmrg_logger &logger)
{
	torch::Tensor E0 = torch::full({}, 100000.0, in_out_state[0].options().merge_in(torch::kDouble));
	auto sweep_dir = 1;
	size_t init_pos = in_out_state.orthogonality_center;
	auto N_step = twosites_hamil.size() - 1 + (twosites_hamil.size() == 1);
	torch::Tensor E0_update;
	int step = (in_out_state.orthogonality_center == 0) ? 1 : -1;
	if (twosites_hamil.size() == 1)
		step = 0;
	// fmt::print("step {}\n",step);
	auto &oc = in_out_state.oc;
	if (oc == in_out_state.size() - 1)
		--oc;
	dmrg_2sites_update update(hamiltonian, twosites_hamil, oc, Env, options);
	auto iteration = 0u;
	logger.init(options);
	for (iteration = 0u; iteration < options.maximum_iterations; ++iteration)
	{
		torch::Tensor E0_tens;
		// fmt::print("\nSweep\n\n");
		std::tie(E0_tens, step) =
		    sweep(in_out_state, update, step, 2 * N_step, in_out_state.size() - 2); // sweep from the oc and back to it.
		logger.it_log_all(iteration, E0_tens, in_out_state);
		E0_update = E0_tens;
		std::swap(E0, E0_update);
		if (!((abs(E0_update - E0) > options.convergence_criterion))
		         .item()
		         .to<bool>()) // looks weird? it's so it stop on nan (nan
		                      // compare false with everything).
		{
			break;
		}
	}
	if (oc != init_pos)
	{
		if (oc != init_pos - 1 and init_pos != in_out_state.size() - 1)
			throw std::runtime_error(fmt::format(
			    "the orthogonality center finished somewhere surprising! final oc: {}. original oc: {}", oc, init_pos));
	}
	if (oc != init_pos)
		in_out_state.move_oc(init_pos);

	logger.end_log_all(iteration, E0, in_out_state);

	return E0;
}

template <class MPO_T, class MPS_T, class env_hold_T>
void generate_env_impl(const MPO_T &hamiltonian, const MPS_T &state, env_hold_T &Env)
{
	using MPT_t = typename dependant_tensor_network<MPS_T>::MPT_type;
	using TEST_MPT_B = typename dependant_tensor_network<MPO_T>::MPT_type;
	static_assert(std::is_same_v<MPT_t, TEST_MPT_B>, "MPS incompatible with MPO");
	static_assert(std::is_same_v<typename env_hold_T::Tens, typename MPS_T::Tens>,
	              "Environment holder incompatible with MPS and MPO");

	Env.env = MPT_t(hamiltonian.size() + 2);

	auto LS = shape_from(state.front(), {-1, 0, 0});
	auto trivial_Ledge =
	    ones_like(shape_from(LS, shape_from(hamiltonian.front(), {-1, 0, 0, 0}), LS.inverse_cvals()).neutral_shape());
	auto RS = shape_from(state.back(), {0, 0, -1});
	auto trivial_Redge =
	    ones_like(shape_from(RS, shape_from(hamiltonian.back(), {0, 0, -1, 0}), RS.inverse_cvals()).neutral_shape());

	Env[-1] = trivial_Ledge;
	Env[hamiltonian.size()] = trivial_Redge;
	size_t i = 0;
	while (i < state.orthogonality_center)
	{
		// generate left environment
		Env[i] = compute_left_env(hamiltonian[i], state[i], Env[i - 1]);
		++i;
	}
	i = hamiltonian.size() - 1;
	while (i > state.orthogonality_center)
	{
		// generate right environement
		Env[i] = compute_right_env(hamiltonian[i], state[i], Env[i + 1]);
		--i;
	}
}

env_holder generate_env(const MPO &hamiltonian, const MPS &state)
{
	env_holder Env;
	generate_env_impl(hamiltonian, state, Env);
	return Env;
}
benv_holder generate_env(const bMPO &hamiltonian, const bMPS &state)
{
	benv_holder Env;
	generate_env_impl(hamiltonian, state, Env);
	return Env;
}

template <class Tensor>
Tensor compute_left_env_impl(const Tensor &Hamil, const Tensor &MPS, const Tensor &left_env)
{
	/**
	       ┌─┐ ┌─┐
	       │ ├─┤Y├ 2
	       │ │ └┬┘
	       │ │ ┌┴┐
	out =  │L├─┤H├ 1
	       │ │ └┬┘
	       │ │ ┌┴┐
	       │ ├─┤Y├ 0
	       └─┘ └─┘

	tensor index ordering:

	  1
	 ┌┴┐
	0┤H├2
	 └┬┘
	  3

	  1
	 ┌┴┐
	0┤Y├2
	 └─┘

	left_env has same ordering as out.
	H = Hamil
	Y = MPS
 */
	auto out = tensordot(left_env, MPS, {0}, {0});
	out = tensordot(out, Hamil, {0, 2}, {0, 3});
	out = tensordot(out, MPS.conj(), {0, 2}, {0, 1});
	return out;
}
torch::Tensor compute_left_env(const torch::Tensor &Hamil, const torch::Tensor &MPS, const torch::Tensor &left_env)
{
	return compute_left_env_impl(Hamil, MPS, left_env);
}
btensor compute_left_env(const btensor &Hamil, const btensor &MPS, const btensor &left_env)
{
	return compute_left_env_impl(Hamil, MPS, left_env);
}
template <class Tensor>
Tensor compute_right_env_impl(const Tensor &Hamil, const Tensor &MPS, const Tensor &right_env)
{
	/**
	 * Left-right mirror to compute_left_env, with same index ordering (no mirroring) for Y and H.
	 */
	auto out = tensordot(right_env, MPS, {0}, {2});
	out = tensordot(out, Hamil, {0, 3}, {2, 3});
	out = tensordot(out, MPS.conj(), {3, 0}, {1, 2});
	return out;
}
torch::Tensor compute_right_env(const torch::Tensor &Hamil, const torch::Tensor &MPS, const torch::Tensor &right_env)
{
	return compute_right_env_impl(Hamil, MPS, right_env);
}
btensor compute_right_env(const btensor &Hamil, const btensor &MPS, const btensor &right_env)
{
	return compute_right_env_impl(Hamil, MPS, right_env);
}

template <class MPO_type>
auto compute_2sitesHamil_impl(const MPO_type &hamil)
{
	auto l = hamil.size();
	typename dependant_tensor_network<MPO_type>::MPT_type out(l - 1);
	for (size_t i = 0; i < l - 1; ++i)
	{
		out[i] = tensordot(hamil[i], hamil[i + 1], {2}, {0}).permute({0, 1, 3, 4, 2, 5});
		// out[i].contiguous(); //not sure it's a good idea, possibly tensordot does some reordoring on its input, in
		// which case it might not be worth it to do it here. Mesure!
	}
	return out;
}

MPT details::compute_2sitesHamil(const MPO &hamil) { return compute_2sitesHamil_impl(hamil); }
bMPT details::compute_2sitesHamil(const bMPO &hamil) { return compute_2sitesHamil_impl(hamil); }

template <class Tensor>
Tensor hamil2site_times_state_impl(const Tensor &state, const Tensor &hamil, const Tensor &Lenv, const Tensor &Renv)
{
	auto out = tensordot(Lenv, state, {0}, {0});
	out = tensordot(out, hamil, {0, 2, 3}, {0, 4, 5});
	out = tensordot(out, Renv, {1, 4}, {0, 1});
	return out;
}
torch::Tensor details::hamil2site_times_state(const torch::Tensor &state, const torch::Tensor &hamil,
                                              const torch::Tensor &Lenv, const torch::Tensor &Renv)
{
	return hamil2site_times_state_impl(state, hamil, Lenv, Renv);
}
btensor details::hamil2site_times_state(const btensor &state, const btensor &hamil, const btensor &Lenv,
                                        const btensor &Renv)
{
	return hamil2site_times_state_impl(state, hamil, Lenv, Renv);
}

template <class Tensor>
std::tuple<Tensor, Tensor, Tensor> eig2x2Mat_impl(const Tensor &a0, const Tensor &a1, const Tensor &b)
{
	auto crit = sqrt(pow(a0 - a1, 2) + 4 * (b.pow(2)));
	auto E0 = (a0 + a1 - crit) / 2; // smallest eigenvalue. largest is (a0+a1 +crit)/2

	auto o_coeff = sqrt((E0 - a1) / (-crit)); // from arxiv.org/pdf/1908.03795.pdf
	auto n_coeff = -b * o_coeff / (a1 - E0);  // can't use o^2+n^2 = 1: loose important phase information that way.
	return std::make_tuple(E0, o_coeff, n_coeff);
}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> details::eig2x2Mat(const torch::Tensor &a0,
                                                                           const torch::Tensor &a1,
                                                                           const torch::Tensor &b)
{
	return eig2x2Mat_impl(a0, a1, b);
}
std::tuple<btensor, btensor, btensor> details::eig2x2Mat(const btensor &a0, const btensor &a1, const btensor &b)
{
	return eig2x2Mat_impl(a0, a1, b);
}

template <class Tensor>
std::tuple<Tensor, Tensor, Tensor, Tensor> one_step_lanczos_impl(const Tensor &state, const Tensor &hamil,
                                                                 const Tensor &Lenv, const Tensor &Renv)
{
	auto psi_ip = hamil2site_times_state(state, hamil, Lenv, Renv);
	// auto a0 = torch::real(torch::tensordot(psi_ip, state.conj(), {0, 1, 2, 3}, {0, 1, 2, 3}));//real doesn't work if
	// the dtype isn't complex... hopefully will be solved on pytorch's end once the complex support is completed
	auto a0 = (tensordot(psi_ip, state.conj(), {0, 1, 2, 3}, {0, 1, 2, 3}));
	psi_ip -= state * a0;
	auto b = sqrt((tensordot(psi_ip, psi_ip.conj(), {0, 1, 2, 3}, {0, 1, 2, 3})));
	const bool non_singular = [=]()
	{
		btensor::Scalar X = ge(b.abs(), 1e-15).item();
		return X.to<bool>();
	}();
	if (non_singular)
		psi_ip /= b;
	auto a1 = (tensordot(psi_ip.conj(), quantt::details::hamil2site_times_state(psi_ip, hamil, Lenv, Renv),
	                     {0, 1, 2, 3}, {0, 1, 2, 3}));
	return std::make_tuple(psi_ip, a0, a1, b);
}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> details::one_step_lanczos(
    const torch::Tensor &state, const torch::Tensor &hamil, const torch::Tensor &Lenv, const torch::Tensor &Renv)
{
	return one_step_lanczos_impl(state, hamil, Lenv, Renv);
}
std::tuple<btensor, btensor, btensor, btensor> details::one_step_lanczos(const btensor &state, const btensor &hamil,
                                                                         const btensor &Lenv, const btensor &Renv)
{
	return one_step_lanczos_impl(state, hamil, Lenv, Renv);
}

/**
 * return the energy, and the state. In that order
 */
template <class Tensor>
std::tuple<Tensor, Tensor> two_sites_update_impl(const Tensor &state, const Tensor &hamil, const Tensor &Lenv,
                                                 const Tensor &Renv)
{
	auto [psi_ip, a0, a1, b] = one_step_lanczos(state, hamil, Lenv, Renv);
	auto [E, o_coeff, n_coeff] = eig2x2Mat(a0, a1, b);
	// fmt::print("ZIM\n E: {}\n O: {}\n N: {}\n",E,o_coeff,n_coeff);
	// fmt::print("a0: {}\na1: {}\nb: {}\nZOOM\n",a0,a1,b);
	auto psi_update = o_coeff * state + n_coeff * psi_ip;
	return std::make_tuple(E, psi_update);
}
/**
 * return the energy, and the state. In that order
 */
std::tuple<torch::Tensor, torch::Tensor> two_sites_update(const torch::Tensor &state, const torch::Tensor &hamil,
                                                          const torch::Tensor &Lenv, const torch::Tensor &Renv)
{
	return two_sites_update_impl(state, hamil, Lenv, Renv);
}
std::tuple<btensor, btensor> two_sites_update(const btensor &state, const btensor &hamil, const btensor &Lenv,
                                              const btensor &Renv)
{
	return two_sites_update_impl(state, hamil, Lenv, Renv);
}

} // namespace quantt
