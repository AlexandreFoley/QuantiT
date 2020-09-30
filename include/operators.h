/*
 * File: operators.h
 * Project: quantt
 * File Created: Monday, 17th August 2020 9:16:18 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Monday, 17th August 2020 9:16:18 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#ifndef A4334AE3_0ED7_40A8_999C_F388D4675687
#define A4334AE3_0ED7_40A8_999C_F388D4675687

#include <torch/torch.h>

#include "doctest/cond_doctest.h"

#include <fmt/core.h>
namespace quantt{


namespace {
using tens = torch::Tensor;
}

/**
 * generate the set of operator for spins 1/2 fermions.
 * return them in this order: c_up,c_dn,F,id where c_up is the up spin fermion annihilation operator,
 * c_dn is the down spin fermion annihilation operator, F is the fermion phase operator (necessary for non local anti-commutations)
 * and id is the identity matrix. The creation operator are obtained by taking the hermitian conjugate of the annihilation operators.
 * All those operators are 4x4 matrices.
 */
std::tuple<tens,tens,tens,tens> fermions();

/**
 * Generate the set of Pauli matrices, proportionnal to spin 1/2 spin operator.
 * return them in this order: Sx,iSy,Sz,lo,id where Sx is the x Pauli matrix, 
 * iSy is the y Pauli matrix multiplied with the imaginary unity, Sz is the z Pauli matrix.
 * lo is the lowering operator: it reduce the spin by 1. if the spin is already -1/2 it destroy the state.
 * the raising operator is lo's hermitian conjugate.
 * id is the identity matrix.
 * All those operators are 2x2 matrices.
 * iSy is given instead of Sy to avoid unnecarily introducing complex number. Time reversible model can be written in term of iSy without any complex numbers.
 */
std::tuple<tens,tens,tens,tens,tens> pauli();



TEST_CASE("half spin fermions")
{
	auto [c_up,c_dn,F,id] = fermions();

	auto cd_up = c_up.conj().t();
	auto cd_dn = c_dn.conj().t();
	//The specific content of those matrices isn't all that important, so long as they have the following properties
	//anticommutation
	CHECK(torch::equal(torch::matmul(cd_up,c_up),id - torch::matmul(c_up,cd_up)));
	CHECK(torch::equal(torch::matmul(cd_dn,c_dn),id - torch::matmul(c_dn,cd_dn)));
	
	CHECK(torch::equal(torch::matmul(cd_up,c_dn),-1*torch::matmul(c_dn,cd_up)));
	CHECK(torch::equal(torch::matmul(cd_dn,c_up),-1*torch::matmul(c_up,cd_dn)));

	CHECK(torch::equal(torch::matmul(c_up,c_dn),-1*torch::matmul(c_dn,c_up)));
	CHECK(torch::equal(torch::matmul(c_up,c_up),-1*torch::matmul(c_up,c_up)));
	CHECK(torch::equal(torch::matmul(c_dn,c_dn),-1*torch::matmul(c_dn,c_dn)));
	
	CHECK(torch::equal(torch::matmul(c_up,F),-1*torch::matmul(F,c_up)));
	CHECK(torch::equal(torch::matmul(c_dn,F),-1*torch::matmul(F,c_dn)));
	//F^2 = 1
	CHECK(torch::equal(torch::matmul(F,F),id));
	//identity
	CHECK(torch::equal(torch::matmul(id,id),id));
	CHECK(torch::equal(torch::matmul(id,c_up),c_up));
	CHECK(torch::equal(torch::matmul(id,F),F));
	CHECK(torch::equal(torch::matmul(id,c_dn),c_dn));
}
TEST_CASE("Pauli matrices")
{
	auto [sx,isy,sz,lo,id] = pauli();

	CHECK(torch::equal(torch::matmul(lo,lo),torch::zeros({2,2},torch::kInt8)));
	CHECK(torch::equal(torch::matmul(sx,sx),id));
	CHECK(torch::equal(torch::matmul(isy,isy),-1*id));
	CHECK(torch::equal(torch::matmul(sz,sz),id));
	CHECK(torch::equal(torch::matmul(sx,torch::matmul(isy,sz)),-1*id));
	// fmt::print("size of torch tensor {}\n",sizeof(sx));
}

}//quantt

#endif /* A4334AE3_0ED7_40A8_999C_F388D4675687 */
