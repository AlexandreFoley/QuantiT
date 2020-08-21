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
#include <fmt/core.h>
namespace quantt{

using namespace details;

struct env_holder
{
	MPT env;
	torch::Tensor& operator[](int64_t i){return env[i+1];}
	const torch::Tensor& operator[](int64_t i) const {return env[i+1];}
};

//forward declaration... yet to be implemented.
MPS random_MPS(size_t length,size_t bond_dim);
MPS random_MPS(size_t length,int64_t bond_dim,MPO hamil);
MPS random_MPS(size_t length,int64_t bond_dim,const std::vector<int64_t>& phys_dims); //should be done as with a range of sort for the phys_dim, when it becomes standard
torch::Scalar dmrg_impl(const MPO& hamiltonian,const MPT& two_site_hamil, MPS& in_out_state,const dmrg_options& options,env_holder& Env);
env_holder generate_env(const MPO& hamiltonian,const MPS& in_out_state);
torch::Tensor compute_left_env(torch::Tensor Hamil,torch::Tensor MPS,torch::Tensor left_env);
torch::Tensor compute_right_env(torch::Tensor Hamil,torch::Tensor MPS,torch::Tensor left_env);
MPT compute_2sitesHamil(const MPO& hamiltonian);
std::tuple<torch::Tensor,torch::Tensor> two_sites_update(torch::Tensor state,torch::Tensor hamil,torch::Tensor Left_environment,torch::Tensor Right_environment);


torch::Scalar dmrg(const MPO& hamiltonian, MPS& in_out_state,const dmrg_options& options)
{
	if (!options.pytorch_gradient)
	{
		torch::NoGradGuard Gradientdisabled;//Globally (but Thread local) disable gradient computation while this object exists.
		auto Env = generate_env(hamiltonian,in_out_state);
		auto TwositesH = compute_2sitesHamil(hamiltonian);
		return details::dmrg_impl(hamiltonian,TwositesH,in_out_state,options,Env);
	}//Gradientdisabled gets destroyed here, gradient computation status is restore to what it was before.
	else
	{
		auto Env = generate_env(hamiltonian,in_out_state);
		auto TwositesH = compute_2sitesHamil(hamiltonian);
		return details::dmrg_impl(hamiltonian,TwositesH,in_out_state,options,Env);
	}
	
}


std::tuple<torch::Scalar,MPS> dmrg(const MPO& hamiltonian,const dmrg_options& options)
{
	using namespace torch::indexing;
	auto length = hamiltonian.size();
	auto out_mps = random_MPS(length,options.minimum_bond,hamiltonian); 
	out_mps[0] = out_mps[0].index({Slice(0,1),Ellipsis});// chop off the extra bond on the edges of the MPS
	out_mps[length-1] = out_mps[length-1].index({Ellipsis,Slice(0,1)}); // the other end.
	auto E0 = dmrg(hamiltonian,out_mps,options);
	return std::make_tuple(E0,out_mps);
}

template<class T> auto sweep(MPS& state,T update,int step,size_t Nstep, size_t right_edge,size_t left_edge=0)
{
	torch::Tensor E0;
	// fmt::print("sweep!\n");
	for (size_t i = 0; i<Nstep;++i)
	{
		E0 = update(state,step);
		auto& oc = state.orthogonality_center;
		step += 2*(oc == left_edge ) - 2*(oc == right_edge);
		// fmt::print("i {} oc {} step {} || ",i,oc,step);
	}
	// fmt::print("\n");
	return std::make_tuple(E0,step);
}

struct dmrg_2sites_update
{
	const MPO& hamil;
	const MPT& twosite_hamil;
	size_t& oc;
	env_holder& Env;
	const dmrg_options& options;

	dmrg_2sites_update(const MPO& _hamil,const MPT& _twosites_hamil,size_t& _oc,env_holder& _Env,const dmrg_options& _options)
	:hamil(_hamil),twosite_hamil(_twosites_hamil),oc(_oc),Env(_Env),options(_options) {}
	
	torch::Tensor operator()(MPS& state,int step)
	{
		bool forward = step == 1;
		torch::Tensor E0;
		auto local_state = torch::tensordot(state[oc],state[oc+1],{2},{0});
		std::tie(E0,local_state) = two_sites_update(local_state,twosite_hamil[oc],Env[oc-1],Env[oc+2]);
		auto [u,d,v] = quantt::svd(local_state,2,options.cutoff,options.minimum_bond,options.maximum_bond);
		if (forward)
		{
			state[oc] = u;
			state[oc+1] = torch::tensordot(torch::diag(d),v,{1},{2});
			Env[oc] = compute_left_env(hamil[oc],state[oc],Env[oc-1]);
		}
		else
		{
			state[oc] = torch::tensordot(u,torch::diag(d),{2},{0});
			state[oc+1] = v.permute({2,0,1});
			Env[oc+1] = compute_right_env(hamil[oc],state[oc+1],Env[oc+2]);
		}
		oc += step;
		return E0;
	}
};

/**
 * The actual implementation.
 */
torch::Scalar details::dmrg_impl(const MPO& hamiltonian,const MPT& twosites_hamil, MPS& in_out_state,const dmrg_options& options, env_holder& Env)
{
	double E0 = 100000.0;
	auto sweep_dir = 1;
	size_t init_pos = in_out_state.orthogonality_center;
	auto N_step = twosites_hamil.size()-1;
	double E0_update;
	int step = (in_out_state.orthogonality_center==0)? 1 : -1;
	// fmt::print("step {}\n",step);
	auto& oc = in_out_state.oc;
	if (oc == in_out_state.size()-1) --oc;
	dmrg_2sites_update update(hamiltonian,twosites_hamil,oc,Env,options);
	for (auto iteration=0u;iteration<options.maximum_iterations;++iteration)
	{	
		torch::Tensor E0_tens;
		std::tie(E0_tens,step) = sweep(in_out_state,update,step,2*N_step,in_out_state.size()-2); //sweep from the oc and back to it.
		E0_update = E0_tens.item().to<double>();
		std::swap(E0,E0_update);
		if (!(std::abs(E0_update-E0) > options.convergence_criterion)) // looks weird? it's so it stop on nan (nan compare false with everything).
		{
			break;
		}
	}
	if (oc != init_pos) 
	{
		if (oc != init_pos-1 and init_pos != in_out_state.size()-1 ) throw std::runtime_error(fmt::format("the orthogonality center finished somewhere surprising! final oc: {}. original oc: {}" ,oc,init_pos) );
	}
	if (oc != init_pos) in_out_state.move_oc(init_pos);
		
	
	return E0;
}
template<class T> MPS random_MPS_impl(size_t length,int64_t bond_dim,T phys_dim)
{
	MPS out(length);
	for (auto i=0u; i<length;++i)
	{
		out[i] = torch::rand({bond_dim,phys_dim(i),bond_dim});
	}
	return out;
}

MPS random_MPS(size_t length,int64_t bond_dim,MPO hamil)
{
	return random_MPS_impl(length,bond_dim,[&hamil](size_t i){
		return hamil[i].sizes()[3];
	});
}
MPS random_MPS(size_t length,int64_t bond_dim,int64_t phys_dim)
{
	return random_MPS_impl(length,bond_dim,[phys_dim](size_t i){
		return phys_dim;
	});
}
MPS random_MPS(size_t length,int64_t bond_dim,std::vector<int64_t> phys_dims)
{
	return random_MPS_impl(length,bond_dim,[&phys_dims](size_t i)
	{
		return phys_dims[i];
	});
}

env_holder generate_env(const MPO& hamiltonian,const MPS& state)
{
	env_holder Env;
	Env.env = MPT(hamiltonian.size()+2);
	torch::Tensor trivial_edge = torch::ones({1,1,1});
	Env[-1] = trivial_edge;
	Env[hamiltonian.size()] = trivial_edge;
	size_t i = 0;
	while (i<state.orthogonality_center)
	{
		//generate left environment
		Env[i] = compute_left_env(hamiltonian[i],state[i],Env[i-1]);
		++i;
	}
	i = hamiltonian.size()-1;
	while (i>state.orthogonality_center)
	{
		//generate right environement
		Env[i] = compute_left_env(hamiltonian[i],state[i],Env[i+1]);
		--i;
	}
	return Env;
}

torch::Tensor compute_left_env(torch::Tensor Hamil,torch::Tensor MPS,torch::Tensor left_env)
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
	auto out = torch::tensordot(left_env,MPS,{0},{0});
	out = torch::tensordot(out,Hamil,{0,2},{0,3});
	out = torch::tensordot(out,MPS.conj(),{0,2},{0,1});
	return out;
}

torch::Tensor compute_right_env(torch::Tensor Hamil,torch::Tensor MPS,torch::Tensor right_env)
{
	/**
	 * Left-right mirror to compute_left_env, with same index ordering (no mirroring) for Y and H.
	 */
	auto out = torch::tensordot(right_env,MPS,{0},{2});
	out = torch::tensordot(out,Hamil,{0,3},{2,3});
	out = torch::tensordot(out,MPS.conj(),{3,0},{1,2});
	return out;
}

MPT compute_2sitesHamil(const MPO & hamil)
{
	auto l = hamil.size(); 
	MPT out(l-1);
	for (size_t i = 0; i<l-1;++i)
	{
		out[i] = torch::tensordot(hamil[i],hamil[i + 1],{2},{0});
		out[i] = out[i].permute({0,1,3,4,5,2});
		// out[i].contiguous(); //not sure it's a good idea, possibly tensordot does some reordoring on its input, in which case it might not be worth it to do it here. Mesure!
	}
	return out;
}

torch::Tensor hamil2site_times_state(torch::Tensor state,torch::Tensor hamil,torch::Tensor Lenv,torch::Tensor Renv)
{
	auto out = torch::tensordot(Lenv,state,{0},{0});
	out = torch::tensordot(out,hamil,{0,2,3},{0,5,4});
	out = torch::tensordot(out,Renv,{1,4},{0,1});
	return out;
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> details::eig2x2Mat(torch::Tensor a0,torch::Tensor a1, torch::Tensor b)
{
	auto E = (a0+a1 - torch::sqrt(torch::pow(a0-a1,2)+4*b.pow(2)))/2;
	auto o_coeff = b/torch::sqrt(torch::pow(a0-E,2)+b.pow(2));
	auto n_coeff = -(a0-E)*o_coeff/b;//can't use o^2+n^2 = 1: loose important phase information that way.
	return std::make_tuple(E,o_coeff,n_coeff);
}

/**
 * return the energy, and the state. In that order
 */
std::tuple<torch::Tensor,torch::Tensor> two_sites_update(torch::Tensor state,torch::Tensor hamil,torch::Tensor Lenv,torch::Tensor Renv)
{
	// one step of lanczos
	auto n = torch::sqrt(torch::tensordot(state,state,{0,1,2,3},{0,1,2,3}));
	state /= n;
	auto psi_ip = hamil2site_times_state(state,hamil,Lenv,Renv);
	auto b = torch::sqrt(torch::tensordot(psi_ip,psi_ip.conj(),{0,1,2,3},{0,1,2,3}));
	psi_ip /= b;
	auto a0 = torch::tensordot(psi_ip,state.conj(),{0,1,2,3},{0,1,2,3});
	auto a1 = torch::tensordot(hamil2site_times_state(psi_ip,hamil,Lenv,Renv),state.conj(),{0,1,2,3},{0,1,2,3});
	auto [E,o_coeff,n_coeff] = eig2x2Mat(a0,a1,b);
	auto psi_update = o_coeff*state + n_coeff*psi_ip;
	return std::make_tuple(E, psi_update);
	
}




}//quantt
