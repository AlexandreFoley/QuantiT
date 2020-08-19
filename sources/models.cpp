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


#include <torch/torch.h>
#include "models.h"
#include "operators.h"

namespace quantt
{
	MPO details::Heisenberg_impl(torch::Scalar J,size_t lenght)
	{
		auto type = torch::get_default_dtype();
		if (J.isIntegral(true)) type = torch::scalarTypeToTypeMeta(torch::kInt8);
		auto local_tens = torch::zeros({5,2,5,2},type);
		auto [sx,isy,sz,lo,id] = pauli();
		using namespace torch::indexing;
		local_tens.index_put_({0,Slice(),0,Slice()},id);
		local_tens.index_put_({1,Slice(),0,Slice()},sx);
		local_tens.index_put_({2,Slice(),0,Slice()},-isy);
		local_tens.index_put_({3,Slice(),0,Slice()},sz);
		local_tens.index_put_({4,Slice(),0,Slice()},0);
		local_tens.index_put_({4,Slice(),1,Slice()},J*sx);
		local_tens.index_put_({4,Slice(),2,Slice()},J*isy);
		local_tens.index_put_({4,Slice(),3,Slice()},J*sz);
		local_tens.index_put_({4,Slice(),4,Slice()},id);
		local_tens.contiguous();
		MPO out(lenght,local_tens);
		out[0] = out[0].index({Slice(4,5),Ellipsis});
		out[lenght-1] = out[lenght-1].index({Ellipsis,Slice(0,1),Slice()});
		return out;
	}
	MPO Heisenberg(torch::Scalar J, size_t lenght)
	{	
		return details::Heisenberg_impl(-J/2.0,lenght);
	}

	MPO Hubbard(torch::Scalar U,torch::Scalar mu,size_t lenght)
	{
		auto local_tens = torch::zeros({6,4,6,4});
		auto [c_up,c_dn,F,id] = fermions();
		auto n_up = torch::matmul(c_up.conj().t(),c_up);
		auto n_dn = torch::matmul(c_dn.conj().t(),c_dn);
		auto Local = U*torch::matmul(n_up,n_dn) + mu*(n_up+n_dn);
		using namespace torch::indexing;
		local_tens.index_put_({0,Slice(),0,Slice()},id);
		local_tens.index_put_({1,Slice(),0,Slice()},c_up);
		local_tens.index_put_({2,Slice(),0,Slice()},c_dn);
		local_tens.index_put_({3,Slice(),0,Slice()},c_up.conj().t());
		local_tens.index_put_({4,Slice(),0,Slice()},c_dn.conj().t());
		local_tens.index_put_({5,Slice(),0,Slice()},Local);
		local_tens.index_put_({5,Slice(),1,Slice()},torch::matmul(c_up.conj().t(),F));
		local_tens.index_put_({5,Slice(),2,Slice()},torch::matmul(c_dn.conj().t(),F));
		local_tens.index_put_({5,Slice(),3,Slice()},torch::matmul(F,c_up));
		local_tens.index_put_({5,Slice(),4,Slice()},torch::matmul(F,c_dn));
		local_tens.index_put_({5,Slice(),5,Slice()},id);
		local_tens.contiguous();
		MPO out(lenght,local_tens);
		out[0] = out[0].index({Slice(4,5),Ellipsis});
		out[lenght-1] = out[lenght-1].index({Ellipsis,Slice(0,1),Slice()});
		return out;
	} 

}//quantt