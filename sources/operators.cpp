/*
 * File: operators.cpp
 * Project: QuantiT
 * File Created: Monday, 17th August 2020 9:41:54 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Monday, 17th August 2020 9:41:54 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * Licensed under GPL v3
 */

#include "operators.h"



namespace quantit
{

std::tuple<tens,tens,tens,tens> fermions()
{
	torch::Tensor c_up = torch::zeros({4,4},torch::kInt8);
	auto c_dn = torch::zeros({4,4},torch::kInt8);
	auto F = torch::zeros({4,4},torch::kInt8);
	auto id = torch::zeros({4,4},torch::kInt8);

{	auto Acc_cup = c_up.accessor<int8_t,2>();
	Acc_cup[0][1] = 1;
	Acc_cup[2][3] = 1;}
{	auto Acc_cdn = c_dn.accessor<int8_t,2>();
	Acc_cdn[0][2] = 1;
	Acc_cdn[1][3] = -1;}
{	auto Acc_F = F.accessor<int8_t,2>();
	Acc_F[0][0] = Acc_F[3][3] = 1;
	Acc_F[1][1] = Acc_F[2][2] = -1;}
{	auto Acc_id = id.accessor<int8_t,2>();
	Acc_id[0][0] = Acc_id[1][1] = Acc_id[2][2] = Acc_id[3][3] = 1;}
	return std::make_tuple(c_up,c_dn,F,id);
}

std::tuple<btensor,btensor,btensor,btensor> fermions(const btensor& shape) {
	return for_each(fermions(),[&shape](const torch::Tensor& tens) -> btensor
	{
		btensor local_shape = shape;
		return from_basic_tensor_like(local_shape.set_selection_rule_(find_selection_rule(tens,shape)),tens);
	});
}

std::tuple<btensor,btensor,btensor> pauli(const btensor& shape) {
	auto [Sx,iSy, Sz,lo,id] = pauli();
	return for_each(std::make_tuple(Sz,lo,id),[shape](const torch::Tensor& tens) -> btensor
	{
		btensor local_shape = shape;
		return from_basic_tensor_like(local_shape.set_selection_rule_(find_selection_rule(tens,shape)),tens);
	});
}

std::tuple<tens,tens,tens,tens,tens> pauli()
{
	torch::Tensor Sx = torch::zeros({2,2},torch::kInt8);
	auto iSy = torch::zeros({2,2},torch::kInt8);
	auto Sz = torch::zeros({2,2},torch::kInt8);
	auto lo = torch::zeros({2,2},torch::kInt8);
	auto id = torch::zeros({2,2},torch::kInt8);

	{
		auto Acc = Sx.accessor<int8_t,2>();
		Acc[0][1] = Acc[1][0] = 1;
	}
	{
		auto Acc = iSy.accessor<int8_t,2>();
		Acc[0][1] = -(Acc[1][0] = -1);
	}
	{
		auto Acc = Sz.accessor<int8_t,2>();
		Acc[0][0] = -(Acc[1][1] = -1);
	}
	{
		auto Acc = id.accessor<int8_t,2>();
		Acc[0][0] = (Acc[1][1] = 1);
	}
	{
		auto Acc = lo.accessor<int8_t,2>();
		Acc[0][1] = 1;
	}
	return std::make_tuple(Sx,iSy,Sz,lo,id);
}

}//QuantiT