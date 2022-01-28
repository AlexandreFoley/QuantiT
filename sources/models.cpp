/*
 * File: models.cpp
 * Project: quantt
 * File Created: Monday, 17th August 2020 10:31:08 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Monday, 17th August 2020 10:31:08 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#include "models.h"
#include "operators.h"
#include "MPT.h"
#include "blockTensor/btensor.h"
#include <exception>
#include <fmt/core.h>
#include <torch/torch.h>
namespace quantt
{

/**
 * @brief build the MPO for a heisenberg model with coupling J
 * Each tensor are defined so:
 * 
 * 1    0     0     0     0
 * lo   0     0     0     0
 * hi   0     0     0     0
 * S_z  0     0     0     0   
 * 0    2J*hi 2J*lo J*S_z 1
 * 
 * @param J 
 * @param lenght 
 * @return MPO 
 */
MPO details::Heisenberg_impl(torch::Tensor J, size_t lenght)
{
	constexpr char errorstr[] = "{0} must be a rank 0 tensor (a scalar), the supplied {0} is rank {1}";
	if (J.sizes().size() > 0)
		throw std::invalid_argument(fmt::format(errorstr, "J", J.sizes().size()));
	auto type = torch::get_default_dtype();
	if (!J.is_floating_point())
		type = torch::scalarTypeToTypeMeta(torch::kInt8);
	auto local_tens = torch::zeros({5, 2, 5, 2}, type);
	auto [sx, isy, sz, lo, id] = pauli();
	auto hi = lo.t();
	using namespace torch::indexing;
	local_tens.index_put_({0, Slice(), 0, Slice()}, id);
	local_tens.index_put_({1, Slice(), 0, Slice()}, lo);
	local_tens.index_put_({2, Slice(), 0, Slice()}, hi);
	local_tens.index_put_({3, Slice(), 0, Slice()}, sz);
	local_tens.index_put_({4, Slice(), 0, Slice()}, 0);
	local_tens.index_put_({4, Slice(), 1, Slice()}, 2*J * hi);
	local_tens.index_put_({4, Slice(), 2, Slice()}, 2*J * lo);
	local_tens.index_put_({4, Slice(), 3, Slice()}, J * sz);
	local_tens.index_put_({4, Slice(), 4, Slice()}, id);
	local_tens.contiguous();
	MPO out(lenght, local_tens);
	out[0] = out[0].index({Slice(4, 5), Slice(), Slice(), Slice()});
	out[lenght - 1] = out[lenght - 1].index({Slice(), Slice(), Slice(0, 1), Slice()});
	return out;
}
MPO Heisenberg(torch::Tensor J, size_t lenght)
{
	return details::Heisenberg_impl(-J / 4.0, lenght);
}

bMPO to_bMPO(MPO&& tmp_hamil, btensor&& local_shape)
{
	auto length = tmp_hamil.size();
	bMPO out(length);
	out.front() = from_basic_tensor_like(local_shape.basic_create_view({local_shape.sizes()[0]-1,-1,-1,-1},true),tmp_hamil.front(),1e-4);
	constexpr auto message = "the local shape is incompatible with the MPO at site {}.";
	if (!torch::allclose(out.front().to_dense(),tmp_hamil.front())) 
	{
		fmt::print("out: {}\n\n", out.front());
		throw std::invalid_argument(fmt::format(message,0));
	}
	size_t i =1;
	for(; i<length-1;++i)
	{
		out[i] = from_basic_tensor_like(local_shape,tmp_hamil[i],1e-4);
		if (!torch::allclose(out[i].to_dense(),tmp_hamil[i]))throw std::invalid_argument(fmt::format(message,i));
	}
	out.back() = from_basic_tensor_like(local_shape.basic_create_view({-1,-1,0,-1},true),tmp_hamil.back(),1e-4);
	if (!torch::allclose(out.back().to_dense(),tmp_hamil.back())) throw std::invalid_argument(fmt::format(message,i));
	return out;
}

bMPO Heisenberg(torch::Tensor J, size_t lenght,const btensor& phys_shape)
{
	auto p = [&phys_shape](size_t i)-> any_quantity_cref{return phys_shape.element_conserved_qtt(0,i);};
	auto neutral = p(0).neutral();
	auto left_bond = btensor({{{1,neutral},{1,p(0)},{1,p(1)},{1,neutral},{1,neutral}}},p(0),phys_shape.options());
	auto local_shape = shape_from(left_bond,phys_shape,left_bond.conj(),phys_shape.conj());
	MPO tmp_hamil = Heisenberg(J,lenght);
	return to_bMPO(std::move(tmp_hamil),std::move(local_shape)).coalesce();
}

/**
 * @brief 
 * 
 * The local tensor are defined by the following matrix of operators:
 * 1           0         0       0       0
 * c_up        0         0       0       0
 * c_dn        0         0       0       0
 * c_up^dg     0         0       0       0
 * c_dn^dg     0         0       0       0
 * U-mu*N  c_up^dg*F c_dn^dg*F F*c_up F*c_dn 1
 * 
 * @param U e-e interaction
 * @param mu chemical potential
 * @param lenght number of sites in the chain
 * @return MPO Hamiltonian operator
 */
MPO Hubbard(torch::Tensor U, torch::Tensor mu, size_t lenght)
{
	constexpr char errorstr[] = "{0} must be a rank 0 tensor (a scalar), the supplied {0} is rank {1}";
	if (U.sizes().size() > 0)
		throw std::invalid_argument(fmt::format(errorstr, "U", U.sizes().size()));
	if (mu.sizes().size() > 0)
		throw std::invalid_argument(fmt::format(errorstr, "mu", mu.sizes().size()));
	auto local_tens = torch::zeros({6, 4, 6, 4});
	auto [c_up, c_dn, F, id] = fermions();
	auto n_up = torch::matmul(c_up.conj().t(), c_up);
	auto n_dn = torch::matmul(c_dn.conj().t(), c_dn);
	auto Local = U * torch::matmul(n_up, n_dn) - mu * (n_up + n_dn);
	using namespace torch::indexing;
	local_tens.index_put_({0, Slice(), 0, Slice()}, id);
	local_tens.index_put_({1, Slice(), 0, Slice()}, c_up);
	local_tens.index_put_({2, Slice(), 0, Slice()}, c_dn);
	local_tens.index_put_({3, Slice(), 0, Slice()}, c_up.conj().t());
	local_tens.index_put_({4, Slice(), 0, Slice()}, c_dn.conj().t());
	local_tens.index_put_({5, Slice(), 0, Slice()}, Local);
	local_tens.index_put_({5, Slice(), 1, Slice()}, torch::matmul(c_up.conj().t(), F));
	local_tens.index_put_({5, Slice(), 2, Slice()}, torch::matmul(c_dn.conj().t(), F));
	local_tens.index_put_({5, Slice(), 3, Slice()}, torch::matmul(F, c_up));
	local_tens.index_put_({5, Slice(), 4, Slice()}, torch::matmul(F, c_dn));
	local_tens.index_put_({5, Slice(), 5, Slice()}, id);
	local_tens.contiguous();
	MPO out(lenght, local_tens);
	out[0] = out[0].index({Slice(5, 6), Ellipsis});
	out[lenght - 1] = out[lenght - 1].index({Ellipsis, Slice(0, 1), Slice()});
	return out;
}
/**
 * @brief create the MPO representation for the 1D Hubbard hamiltonian with the specified conservation laws.
 * 
 * The local tensor are defined by the following matrix of operators:
 * 1           0         0       0       0    0
 * c_up        0         0       0       0    0
 * c_dn        0         0       0       0    0
 * c_up^dg     0         0       0       0    0
 * c_dn^dg     0         0       0       0    0
 * Û-{mu}^\^  c_up^dg*F c_dn^dg*F F*c_up F*c_dn  1
 * 
 * @param U e-e interaction
 * @param mu chemical potential
 * @param lenght number of sites in the chain
 * @return MPO Hamiltonian operator
 */
bMPO Hubbard(torch::Tensor U, torch::Tensor mu, size_t lenght,const btensor& Phys_shape)
{
	// element: 0 empty, 1 single down electron, 2 single up electron,3 one up and one down
	auto p = [&Phys_shape](size_t i)->any_quantity{return Phys_shape.element_conserved_qtt(0,i);};
	auto left_bond = btensor({{{1,p(0)},{1,p(1)},{1,p(2)},{1,p(1).inverse()},{1,p(2).inverse()},{1,p(0)}}},p(0),Phys_shape.options());
	auto local_shape = shape_from(left_bond,Phys_shape,left_bond.conj(),Phys_shape.conj());
	MPO tmp_hamil = Hubbard(U,mu,lenght);
	return to_bMPO(std::move(tmp_hamil),std::move(local_shape)).coalesce();
}

} // namespace quantt