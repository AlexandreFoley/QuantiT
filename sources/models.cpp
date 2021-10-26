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
#include <exception>
#include <fmt/core.h>
#include <torch/torch.h>
namespace quantt
{
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
bMPO Heisenberg(torch::Tensor J, size_t lenght,btensor local_shape)
{
	MPO tmp_hamil = Heisenberg(J,lenght);
	bMPO out(lenght);
	out.front() = from_basic_tensor_like(local_shape.basic_create_view({4,-1,-1,-1},true),tmp_hamil.front(),1e-4);
	constexpr auto message = "the local shape is incompatible with the MPO at site {}.";
	if (!torch::allclose(out.front().to_dense(),tmp_hamil.front())) throw std::invalid_argument(fmt::format(message,0));
	size_t i =1;
	for(; i<lenght-1;++i)
	{
		out[i] = from_basic_tensor_like(local_shape,tmp_hamil[i],1e-4);
		if (!torch::allclose(out[i].to_dense(),tmp_hamil[i]))throw std::invalid_argument(fmt::format(message,i));
	}
	out.back() = from_basic_tensor_like(local_shape.basic_create_view({-1,-1,0,-1},true),tmp_hamil.back(),1e-4);
	if (!torch::allclose(out.back().to_dense(),tmp_hamil.back())) throw std::invalid_argument(fmt::format(message,i));
	return out;
}

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
	auto Local = U * torch::matmul(n_up, n_dn) + mu * (n_up + n_dn);
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
	out[0] = out[0].index({Slice(4, 5), Ellipsis});
	out[lenght - 1] = out[lenght - 1].index({Ellipsis, Slice(0, 1), Slice()});
	return out;
}

} // namespace quantt