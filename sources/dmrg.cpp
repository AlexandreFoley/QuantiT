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

namespace quantt{

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
torch::Tensor two_sites_update(torch::Tensor state,torch::Tensor hamil,torch::Tensor Left_environment,torch::Tensor Right_environment);


torch::Scalar dmrg(const MPO& hamiltonian, MPS& in_out_state,const dmrg_options& options)
{
	if (options.pytorch_gradient)
	{
		torch::NoGradGuard Gradientdisabled;//Globally (but Thread local) disable gradient computation while this object exists.
		auto Env = generate_env(hamiltonian,in_out_state);
		auto TwositesH = compute_2sitesHamil(hamiltonian);
		return dmrg_impl(hamiltonian,TwositesH,in_out_state,options,Env);
	}//Gradientdisabled gets destroyed here, gradient computation status is restore to what it was before.
	else
	{
		auto Env = generate_env(hamiltonian,in_out_state);
		auto TwositesH = compute_2sitesHamil(hamiltonian);
		return dmrg_impl(hamiltonian,TwositesH,in_out_state,options,Env);
	}
	
}


std::tuple<torch::Scalar,MPS> dmrg(const MPO& hamiltonian,const dmrg_options& options)
{
	using namespace torch::indexing;
	auto length = hamiltonian.size();
	auto out_mps = random_MPS(length,options.minimum_bond,); 
	out_mps[0] = out_mps[0].index({Slice(0,1),Ellipsis});// chop off the extra bond on the edges of the MPS
	out_mps[length-1] = out_mps[length-1].index({Ellipsis,Slice(0,1)}); // the other end.
	auto E0 = dmrg(hamiltonian,out_mps,options);
	return std::make_tuple(E0,out_mps);
}

/**
 * The actual implementation.
 * 
 */
torch::Scalar dmrg_impl(const MPO& hamiltonian,const MPT& twosites_hamil, MPS& in_out_state,const dmrg_options& options, env_holder& Env)
{
	double E0 = 100000.0;
	auto sweep_dir = 1;
	size_t init_pos = in_out_state.orthogonality_center;
	auto N_step = twosites_hamil.size();
	double E0_update;
	if ((in_out_state.orthogonality_center == 0 or in_out_state.orthogonality_center == in_out_state.size()) and options.maximum_iterations == 1)
	{
		//halfsweep for parallel implementation.
		auto step = in_out_state.orthogonality_center == 0 ? 1 : -1;
		auto Nstep = hamiltonian.size();
		for (auto i= 0; i<Nstep; i+=step)
		{
			auto cur_pos = in_out_state.orthogonality_center;
			torch::Tensor local_state = torch::tensordot(in_out_state[cur_pos],in_out_state[cur_pos+1],{3},{1});
			local_state = two_sites_update(local_state,twosites_hamil[cur_pos],Env[cur_pos-1],Env[cur_pos+2]);
			//DOIT
		} 
	}
	else
	{
		N_step *= 2;
		for (auto iteration=0u;iteration<options.maximum_iterations;++iteration)
		{	
			
			std::swap(E0,E0_update);
			if (std::abs(E0_update-E0) < options.convergence_criterion)
			{
				break;
			}
		}
		
	}
	
	return E0;
}

MPS random_MPS(size_t length,int64_t bond_dim,MPO hamil)
{
	MPS out(length);
	for (auto i=0u; i<length;++i)
	{
		out[i] = torch::rand({bond_dim,hamil[i].sizes()[3],bond_dim}); 
	}
	return out;
}
MPS random_MPS(size_t length,int64_t bond_dim,int64_t phys_dim)
{
	MPS out(length);
	for (auto i=0u; i<length;++i)
	{
		out[i] = torch::rand({bond_dim,phys_dim,bond_dim}); 
	}
	return out;
}
MPS random_MPS(size_t length,int64_t bond_dim,std::vector<int64_t> phys_dims)
{
	MPS out(length);
	for (auto i=0u; i<length;++i)
	{
		out[i] = torch::rand({bond_dim,phys_dims[i],bond_dim}); 
	}
	return out;
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

torch::Tensor compute_right_env(torch::Tensor Hamil,torch::Tensor MPS,torch::Tensor left_env)
{
	/**
	 * Left-right mirror to compute_left_env, with same index ordering (no mirroring) for Y and H.
	 */
	auto out = torch::tensordot(left_env,MPS,{0},{2});
	out = torch::tensordot(out,Hamil,{0,3},{2,3});
	out = torch::tensordot(out,MPS.conj(),{0,3},{1,2});
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
	auto out = torch::tensordot(Renv,state,{0},{0});
	out = torch::tensordot(out,hamil,{0,2,3},{0,5,4});
	out = torch::tensordot(out,Renv,{4,1},{0,1});
	return out;
}
std::tuple<double,torch::Tensor> two_sites_update(torch::Tensor state,torch::Tensor hamil,torch::Tensor Lenv,torch::Tensor Renv)
{
	// one step of lanczos
	auto psi_ip = hamil2site_times_state(state,hamil,Lenv,Renv);
	auto b = torch::sqrt(torch::tensordot(psi_ip,psi_ip.conj(),{0,1,2,3},{0,1,2,3}));
	psi_ip /= b;
	auto a0 = torch::tensordot(psi_ip,state.conj(),{0,1,2,3},{0,1,2,3});
	auto a1 = torch::tensordot(hamil2site_times_state(psi_ip,hamil,Lenv,Renv),state.conj(),{0,1,2,3},{0,1,2,3});
	auto dtd = torch::get_default_dtype();
	auto E = (a0+a1 - torch::sqrt(torch::pow(a0-a1,2)+4*b.pow(2)))/2;
	auto o_coeff = b/torch::sqrt(torch::pow(a-E,2)+b.pow(2));
	auto n_coeff = torch::sqrt(1-o_coeff.pow(2));
	auto psi_update = o_coeff*state + n_coeff*psi_ip;
	return std::make_tuple(E.item().to<double>(), psi_update);
	
}




}//quantt
