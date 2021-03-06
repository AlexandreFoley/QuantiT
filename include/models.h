/*
 * File: models.h
 * Project: QuantiT
 * File Created: Monday, 17th August 2020 10:31:00 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Monday, 17th August 2020 10:31:00 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * Licensed under GPL v3
 */

#ifndef FC769B14_0341_4EB5_9CD2_E7CA2865AD73
#define FC769B14_0341_4EB5_9CD2_E7CA2865AD73

#include <torch/torch.h>
#include "MPT.h"

#include "doctest/doctest_proxy.h"

namespace quantit
{
	/**
	 * Generate the hamiltonian for the Heisenberg model.
	 */
	MPO Heisenberg(torch::Tensor J,size_t lenght);
	bMPO Heisenberg(torch::Tensor J, size_t lenght,const btensor& local_shape);
	inline MPO Heisenberg(torch::Scalar J, size_t lenght) { return Heisenberg(torch::full({},J),lenght); }
	inline bMPO Heisenberg(torch::Scalar J, size_t lenght,const btensor local_shape) { return Heisenberg(torch::full({},J),lenght,local_shape); }
	namespace details
	{
		MPO Heisenberg_impl(torch::Tensor J,size_t lenght);
	}

	/**
	 * Generate the Hamiltonian for the first neighbor 1D Hubbard model. The enrgy scale is defined by the first neighbor hopping t=1.
	 */
	MPO Hubbard(torch::Tensor U,torch::Tensor mu,size_t lenght); 
	bMPO Hubbard(torch::Tensor U,torch::Tensor mu,size_t lenght,const btensor& local_shape); 
	inline bMPO Hubbard(torch::Scalar U, torch::Scalar mu, size_t lenght,const btensor& local_shape){return Hubbard(torch::full({},U),torch::full({},mu),lenght,local_shape);}
	inline MPO Hubbard(torch::Scalar U, torch::Scalar mu, size_t lenght){return Hubbard(torch::full({},U),torch::full({},mu),lenght);}

qtt_TEST_CASE("Heisenberg")
{
	int J(1);
	// qtt_REQUIRE(J.isIntegral(false) );
	auto Heis = details::Heisenberg_impl(torch::tensor(J),3);
	qtt_CHECK(Heis[0].sizes().size() == 4);
	auto size_left_edge = std::vector<int64_t>{1l,2l,5l,2l};
	qtt_CHECK(Heis[0].sizes().vec() == size_left_edge);
	qtt_CHECK(Heis[1].sizes().size() == 4);
	auto size_middle = std::vector<int64_t>{5l,2l,5l,2l};
	qtt_CHECK(Heis[1].sizes().vec() == size_middle);
	qtt_CHECK(Heis[2].sizes().size() == 4);
	auto size_right_edge = std::vector<int64_t>{5l,2l,1l,2l}; 
	qtt_CHECK(Heis[2].sizes().vec() == size_right_edge);

	qtt_REQUIRE(!Heis[0].is_floating_point());


	auto test_Heis =  torch::zeros({8,8},torch::kInt8);
	{
		auto acc = test_Heis.accessor<int8_t,2>();
		acc[0][0] = acc[2][1] = acc[1][2] = 2;
		acc[4][2] = acc[2][4] = acc[5][3] = acc[3][5] = 2;
		acc[7][7] = acc[6][5] = acc[5][6] = 2;
		acc[2][2] = acc[5][5] = -2;
	}

	auto cont_Heis = torch::tensordot(Heis[0],Heis[1],{2},{0}).permute({0,1,3,4,2,5}).reshape({1,4,5,4});
	cont_Heis = torch::tensordot(cont_Heis,Heis[2],{2},{0}).permute({0,1,3,4,2,5}).reshape({8,8});
	qtt_CHECK(torch::equal(cont_Heis,test_Heis));
}


}//QuantiT


#endif /* FC769B14_0341_4EB5_9CD2_E7CA2865AD73 */
