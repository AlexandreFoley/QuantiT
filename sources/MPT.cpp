/*
 * File: MPT.cpp
 * Project: QuanTT
 * File Created: Thursday, 23rd July 2020 10:32:10 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Thursday, 23rd July 2020 10:46:12 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#include "MPT.h"
#include <exception>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <random>
#include "blockTensor/btensor.h"
#include "blockTensor/LinearAlgebra.h"
#include "LinearAlgebra.h"
#include "dmrg.h"
// TODO: remove all explicit torch:: can ADL be my friend here?
namespace quantt
{

void MPS::move_oc(int i)
{
	if (not(i >= 0 and i < size()))
		throw std::invalid_argument(" Proposed orthogonality center falls outside the MPS");
	auto dims = (*this)[orthogonality_center].sizes();
	auto prod = [&dims](size_t start, size_t end)
	{
		std::remove_const_t<std::remove_reference_t<decltype(dims[0])>> out = 1;
		for (auto i = start; i < end; ++i)
		{
			out *= dims[i];
		}
		return out;
	};

	while (i < orthogonality_center)
	{
		// move right
		auto &curr_oc = (*this)[orthogonality_center];
		auto &next_oc = (*this)[orthogonality_center - 1];
		dims = curr_oc.sizes();

		// TODO: rewrite this to use quantt's SVD implementation. takes care of the reshaping
		// auto reshaped = curr_oc.reshape({dims[0], prod(1, dims.size())});
		auto [u, d, v] = quantt::svd(curr_oc,1);
		curr_oc = v.permute({2,0,1}).conj(); // needs testing. svd documentation makes no mention of complex numbers case.
		auto ud = u.mul(d); 
		next_oc = torch::tensordot(next_oc, ud, {2}, {0});
		--oc; 
	}

	while (i > orthogonality_center)
	{
		// move left
		auto &curr_oc = (*this)[orthogonality_center];
		auto &next_oc = (*this)[orthogonality_center + 1];
		dims = curr_oc.sizes();
		// TODO: use quantt's SVD implementation. takes care of the reshaping
		// auto reshaped = curr_oc.reshape({prod(0, dims.size() - 1), dims[dims.size() - 1]});
		auto [u, d, v] = quantt::svd(curr_oc,2);
		curr_oc = u;

		auto dv = v.mul(d).t().conj(); 
		next_oc = torch::tensordot(dv, next_oc, {1}, {0});

		++oc;
	}
	// otherwise we're already there, do nothing.
}
void bMPS::move_oc(int i)
{
	if (not(i >= 0 and i < size()))
		throw std::invalid_argument(" Proposed orthogonality center falls outside the MPS");

	while (i < orthogonality_center)
	{
		// move right
		auto &curr_oc = (*this)[orthogonality_center];
		auto &next_oc = (*this)[orthogonality_center - 1];

		auto [u, d, v] = quantt::svd(curr_oc,1);
		curr_oc = v.conj().permute({2,0,1}); // needs testing. svd documentation makes no mention of complex numbers case.

		// testing shows that v is only transposed in the complex number case as well.
		auto ud = u.mul(d);
		next_oc = tensordot(next_oc, ud, {2}, {0});
		--oc;
	}

	while (i > orthogonality_center)
	{
		// move left
		auto &curr_oc = (*this)[orthogonality_center];
		auto &next_oc = (*this)[orthogonality_center + 1];
		// TODO: use quantt's SVD implementation. takes care of the reshaping
		auto [u, d, v] = svd(curr_oc,2);
		curr_oc = u;

		auto dv = v.mul(d).conj(); 
		next_oc = tensordot(dv, next_oc, {0}, {0});

		++oc;
	}
	// otherwise we're already there, do nothing.
}

bool MPS::check_one(const Tens &tens)
{
	// check correctness on fill candidate
	auto sizes = tens.sizes();
	return ((sizes.size() == 3 and sizes[0] == sizes[2]));
}

bool MPS::check_ranks() const
{
	auto prev_bond = operator[](0).sizes()[0];
	bool all_rank_3 = std::all_of(begin(), end(),
	                              [&prev_bond](const Tens &el)
	                              {
		                              using std::swap;
		                              auto bond = el.sizes()[2];
		                              swap(bond, prev_bond);
		                              return el.sizes().size() == 3 and bond == el.sizes()[0];
	                              });
	return all_rank_3;
}

bool MPO::check_one(const Tens &tens)
{
	// check correctness on fill candidate
	auto sizes = tens.sizes();
	// if (!( sizes.size() == 4 and sizes[0] == sizes[2] )) throw std::invalid_argument("The input tensor must have rank
	// 4 and equal bond dimensions (dims 0 and 2).");
	return (sizes.size() == 4 and sizes[0] == sizes[2]);
}

bool bMPO::check_one(const Tens &tens) { return tens.dim() == 4 and Tens::check_product_compat(tens, tens, {0}, {2}); }

bool MPO::check_ranks() const
{
	auto prev_bond = operator[](0).sizes()[0];
	bool all_rank_4 = std::all_of(begin(), end(),
	                              [&prev_bond](const Tens &el)
	                              {
		                              using std::swap;
		                              auto bond = el.sizes()[2];
		                              swap(bond, prev_bond);
		                              return el.sizes().size() == 4 and bond == el.sizes()[0];
	                              });
	// assert(all_rank_4);// a MPS must have only rank 3 tensors
	return all_rank_4;
}
bool bMPO::check_ranks() const
{
	auto it = begin();
	auto next = begin()+1;
	bool all_rank_4 = it->dim() == 4;
	while (all_rank_4 and it != end() and next != end())
	{
		all_rank_4 = next->dim() == 4 and btensor::check_product_compat(*it, *next, {2}, {0});
		++it;
		++next;
	}
	return all_rank_4;
}
bool bMPS::check_ranks() const
{
	auto next = begin()+1;
	auto it = begin();
	bool all_rank_3 = it->dim() == 3;
	while (all_rank_3 and it != end() and next != end())
	{
		all_rank_3 = next->dim() == 3 and btensor::check_product_compat<false>(*it, *next, {2}, {0});
		++it;
		++next;
	}
	return all_rank_3;
}

btensor contract(const bMPS &a, const bMPS &b, const bMPO &obs, btensor left_edge, const btensor &right_edge)
{
	assert(a.size() == b.size());
	for (size_t i = 0; i < a.size(); ++i)
	{
		left_edge = tensordot(left_edge, a[i], {0}, {0});
		left_edge = tensordot(left_edge, obs[i], {0, 2}, {0, 3});
		left_edge = tensordot(left_edge, b[i].conj(), {0, 2}, {0, 1});
	}
	return tensordot(left_edge, right_edge, {0, 1, 2}, {0, 1, 2});
}
btensor contract(const bMPS &a, const bMPS &b, const bMPO &obs)
{
	// todo:: adapt to work with Btensors.
	// need a btensor implementation of ones. must be a one_like thing.
	auto left_edge = ones_like(shape_from(details::edge_shape_prep(a.front(),0),details::edge_shape_prep(obs.front(),0),details::edge_shape_prep(b.front().inverse_cvals(),0)));
	auto right_edge = ones_like(shape_from(details::edge_shape_prep(a.back(),2),details::edge_shape_prep(obs.back(),2),details::edge_shape_prep(b.back().inverse_cvals(),2)));
	return contract(a, b, obs, std::move(left_edge), right_edge);
}
torch::Tensor contract(const MPS &a, const MPS &b, const MPO &obs, torch::Tensor left_edge,
                       const torch::Tensor &right_edge)
{
	assert(a.size() == b.size());
	for (size_t i = 0; i < a.size(); ++i)
	{
		left_edge = tensordot(left_edge, a[i], {0}, {0});
		left_edge = tensordot(left_edge, obs[i], {0, 2}, {0, 3});
		left_edge = tensordot(left_edge, b[i].conj(), {0, 2}, {0, 1});
	}
	return tensordot(left_edge, right_edge, {0, 1, 2}, {0, 1, 2});
}

torch::Tensor contract(const MPS &a, const MPS &b, const MPO &obs)
{
	auto left_edge = ones_like(
	    shape_from(shape_from(a[0], {-1, 0, 0}), shape_from(obs[0], {-1, 0, 0, 0}), shape_from(b[0], {-1, 0, 0}))
	        .neutral_shape_());
	auto right_edge = ones_like(shape_from(shape_from(a.back(), {0, 0, -1}), shape_from(obs.back(), {0, 0, -1, 0}),
	                                       shape_from(b.back(), {0, 0, -1}))
	                                .neutral_shape_());
	return contract(a, b, obs, std::move(left_edge), right_edge);
}

torch::Tensor contract(const MPS &a, const MPS &b, torch::Tensor left_edge, const torch::Tensor &right_edge)
{
	assert(a.size() == b.size());
	for (size_t i = 0; i < a.size(); ++i)
	{
		left_edge = torch::tensordot(left_edge, a[i], {0}, {0});
		left_edge = torch::tensordot(left_edge, inverse_cvals(b[i].conj()), {0, 1}, {0, 1});
	}
	return torch::tensordot(left_edge, right_edge, {0, 1}, {0, 1});
}
torch::Tensor contract(const MPS &a, const MPS &b)
{
	auto E = ones_like(
	    shape_from(shape_from(a[0], {-1, 0, 0}), shape_from(inverse_cvals(b[0]), {-1, 0, 0})).neutral_shape_());
	auto right_edge = ones_like(
	    shape_from(shape_from(a.back(), {0, 0, -1}), shape_from(inverse_cvals(b.back()), {0, 0, -1})).neutral_shape_());
	return contract(a, b, E, right_edge);
}

btensor contract(const bMPS &a, const bMPS &b, btensor left_edge, const btensor &right_edge)
{
	assert(a.size() == b.size());
	for (size_t i = 0; i < a.size(); ++i)
	{
		left_edge = tensordot(left_edge, a[i], {0}, {0});
		left_edge = tensordot(left_edge, (b[i].conj()), {0, 1}, {0, 1});
	}
	return tensordot(left_edge, right_edge, {0, 1}, {0, 1});
}
btensor contract(const bMPS &a, const bMPS &b)
{
	auto left_edge = ones_like(shape_from(details::edge_shape_prep(a.front(),0),details::edge_shape_prep(b.front().inverse_cvals(),0)));
	auto right_edge = ones_like(shape_from(details::edge_shape_prep(a.back(),2),details::edge_shape_prep(b.back().inverse_cvals(),2)));
	return contract(a, b, left_edge, right_edge);
}

/**
 * @brief generate a string of index O for phys_dim such that Sum_i { phys_dim[i][O[i]] } == constraint.
 *
 * By default, the algorithm attempt to generate a string satisfying the constraint in a single pass. However a greater
 * number of pass can be required when phys_dim doesn't return an identical result for every i. I believe 1 pass for
 * every unique physical dimension is enough.
 *
 * Throws if the constraint cannot be satisfied in the number of pass specified, or if the constraint cannot be
 * satisfied at all.
 *
 * @param L length of the MPS
 * @param phys_dim function like object that return the list of conserved value for each of the free index of the random
 * MPS we want to create.
 * @param constraint the constraint to satisfy.
 * @param random_gen random number generator to create a uniform distribution.
 * @param N_pass must be greater than 0, default to 1
 */
template <class T, class R>
void generate_random_string(std::vector<size_t>::iterator out, size_t L, T &&phys_dim, any_quantity_cref constraint,
                            R &&random_gen, size_t N_pass = 1)
{
	if (N_pass < 1)
		throw std::invalid_argument("N_pass must be greater than 1");
	// randomize the input a bit.
	std::uniform_int_distribution<> distrib(0, 1);
	auto neutral = constraint.neutral();
	using param_type = std::uniform_int_distribution<>::param_type;
	any_quantity Sum(neutral);
	for (size_t i = 0; i < L; ++i)
	{ // generate a random starting point. does not satisfy the constraint.
		int x = phys_dim(i).size();
		const auto &a = phys_dim(i);

		distrib.param(param_type(0, x - 1));
		auto r = distrib(random_gen);
		any_quantity apply;
		out[i] = r;
		any_quantity_cref aa = a[r];
		Sum *= aa;
	}
	// fmt::print("random initial set of cvals: {}\n Resulting qnum {}\n", fmt::join(out, out + L, ","), Sum);
	int64_t curr_dist = distance2(Sum, constraint);
	for (size_t n = 0; n < N_pass; ++n)
	{
		for (size_t i = 0; i < L; ++i)
		{ // make adjustement to bring it closer to compliance
			// should be compliant by the time we've done a single pass if all the phys_ind are identical.
			// fmt::print("sum {}\n constraint {}\n distance {} \n", Sum, constraint, curr_dist);
			if (curr_dist == 0)
				return;
			int x = phys_dim(i).size();
			size_t new_o = out[i];
			Sum *= phys_dim(i)[out[i]].inverse();
			// fmt::print("site {} , partial sum {}\n", i , Sum);
			for (int j = 0; j < x; ++j)
			{
				auto curr_sum = Sum * phys_dim(i)[j];
				auto new_dist = distance2(curr_sum, constraint);
				// fmt::print("\tcandidate {}: new Sum {}, new distance {}, candidate value {}\n",j,curr_sum,new_dist, phys_dim(i)[j]);
				if (new_dist < curr_dist)
				// select the candidate in the list that reduce the distance to the target most
				{
					new_o = j;
					curr_dist = new_dist;
				}
			}
			// fmt::print("picked {} {}\n",new_o, phys_dim(i)[new_o]);
			out[i] = new_o;
			Sum *= phys_dim(i)[out[i]];
		}
		if (curr_dist == 0)
			return;
	}
	throw std::invalid_argument("The physical dimensions cannot satisfy the sum rule for the MPS with this number of "
	                            "pass, verify that the constraint is possible or increase the number of pass");
	// error we failed!
}
template <class T>
MPS random_MPS_impl(size_t length, int64_t bond_dim, T phys_dim, torch::TensorOptions opt)
{
	MPS out(length);
	for (auto i = 0u; i < length; ++i)
	{
		out[i] = torch::rand({bond_dim, phys_dim(i), bond_dim}, opt);
	}
	using namespace torch::indexing;
	out[0] = out[0].index({Slice(0, 1), Ellipsis});                   // chop off the extra bond on the edges of the MPS
	out[length - 1] = out[length - 1].index({Ellipsis, Slice(0, 1)}); // the other end.
	return out;
}

template <class T>
btensor make_right_side(const std::vector<size_t> &ind, any_quantity_vector &accum, any_quantity_cref sel_rule,
                        T &&phys_dim, size_t i, size_t bond_dim, size_t length)
{
	for (size_t j = 0; j < bond_dim; ++j)
	{
		accum[j] += phys_dim(i)[ind[j * length + i]].inverse();
	}
	any_quantity_vector L(accum); // a copy because we cannot sort accum. the precise ordering is a critical component
	                              // of an acceptable chains.
	std::sort(L.begin(), L.end());
	auto N_sector = 1;
	auto stop_at = L.end() - 1;
	for (auto it = L.begin(); it != stop_at; ++it)
	{ // sorting make this loop simple
		N_sector += (*it) != (*(it + 1));
	}
	any_quantity_vector cvals(N_sector, phys_dim(i)[0]);
	btensor::index_list section_by_dim(N_sector);
	size_t sect_size = 0;
	size_t sect_index = 0;

	for (auto it = L.begin(); it != L.end(); ++it, ++i)
	{
		++sect_size;
		if ((it + 1) == L.end() or (*it) != (*(it + 1)))
		{
			cvals[sect_index] = *it;
			section_by_dim[sect_index] = sect_size;
			sect_size = 0;
			++sect_index;
		}
	}
	return btensor(btensor::index_list({N_sector}), std::move(cvals), std::move(section_by_dim), sel_rule);
}

template <class T>
bMPS random_bMPS_impl(size_t length, int64_t bond_dim, T &&phys_dim, any_quantity_cref constraint, size_t string_N_pass,
                      torch::TensorOptions opt = {})
{
	std::mt19937 gen((std::random_device())()); // Standard mersenne_twister_engine seeded with rd()
	std::vector<size_t> phys_inds(bond_dim * length);
	static_assert(std::is_same_v<decltype(phys_dim(0)), btensor>,
	              "the phys_dim function like object is invalid, it must return a rank-1 btensor for all index");
	auto it = phys_inds.begin();
	auto phys_ind_cvals = [&phys_dim](size_t i) { return phys_dim(i).get_cvals(); };
	for (size_t i = 0; i < bond_dim; ++i)
	{
		generate_random_string(
		    it, length, phys_ind_cvals, constraint, gen,
		    string_N_pass); // cannot be parallelized without creating a generator (gen) for each parallel instance.
		it += length;
	}
	any_quantity sel_rule = constraint.neutral();
	bMPS out(length);
	btensor left_side({{{1, sel_rule}}}, sel_rule);
	btensor right_side;
	any_quantity_vector accumulate_in_out(bond_dim, constraint);
	for (size_t i = 0; i < length; ++i)
	{
		any_quantity_cref local_sel_rule = i == 0 ? any_quantity_cref(constraint) : any_quantity_cref(sel_rule);
		right_side = make_right_side(phys_inds, accumulate_in_out, local_sel_rule, phys_ind_cvals, i, bond_dim, length);
		out[i] = rand_like(shape_from(left_side, phys_dim(i), right_side), opt);
		if (i == 0)
			right_side.set_selection_rule_(sel_rule);
		right_side.inverse_cvals_();
		swap(right_side, left_side);
	}
	out.back() =  out.back().basic_create_view({-1,-1,0},true);
	#ifndef NDEBUG
	if (! out.check_ranks()) throw std::runtime_error("random MPS generator failed.");
	#endif
	return out;
}

MPS random_MPS(size_t bond_dim, const MPO &hamil, torch::TensorOptions opt)
{
	return random_MPS_impl(
	    hamil.size(), bond_dim, [&hamil](size_t i) { return hamil[i].sizes()[3]; }, opt);
}
MPS random_MPS(size_t length, int64_t bond_dim, int64_t phys_dim, torch::TensorOptions opt = {})
{
	return random_MPS_impl(
	    length, bond_dim, [phys_dim](size_t i) { return phys_dim; }, opt);
}
MPS random_MPS(size_t bond_dim, std::vector<int64_t> phys_dims, torch::TensorOptions opt = {})
{
	return random_MPS_impl(
	    phys_dims.size(), bond_dim, [&phys_dims](size_t i) { return phys_dims[i]; }, opt);
}
bMPS random_bMPS(size_t bond_dim, const bMPO &hamil, any_quantity_cref quantum_number, torch::TensorOptions opt)
{
	auto S = hamil.size();
	std::vector<btensor> x = [&hamil, S, &quantum_number]()
	{
		std::vector<btensor> out(S);
		auto neutral = quantum_number.neutral();
		for (size_t i = 0; i < S; ++i)
		{
			out[i] = shape_from(hamil[i].shape_from({0, 0, 0, -1})).set_selection_rule_(neutral).inverse_cvals();
		}
		return out;
	}();
	return random_bMPS(bond_dim, x, quantum_number, opt);
}
bMPS random_bMPS(size_t length, size_t bond_dim, const btensor &phys_dim_spec, any_quantity_cref q_num,
                 torch::TensorOptions opt)
{
	return random_bMPS_impl(
	    length, bond_dim, [&phys_dim_spec](size_t i) { return phys_dim_spec; }, q_num, 1, opt);
}
bMPS random_bMPS(size_t bond_dim, const std::vector<btensor> &phys_dim_spec, any_quantity_cref q_num,
                 torch::TensorOptions opt)
{
	auto S = phys_dim_spec.size();
	auto phys_dim = [&phys_dim_spec, S](size_t i) { return phys_dim_spec[i]; };
	size_t N_pass = [S, &phys_dim_spec]()
	{
		std::vector<std::tuple<any_quantity_vector::const_iterator, any_quantity_vector::const_iterator>> X(
		    phys_dim_spec.size());
		auto it_X = X.begin();
		for (auto it_pds = phys_dim_spec.begin(); it_pds != phys_dim_spec.end(); ++it_pds, ++it_X)
		{
			*it_X = it_pds->section_conserved_qtt_range(0);
		}
		std::sort(
		    X.begin(), X.end(),
		    [](auto &X, auto &Y)
		    { return std::lexicographical_compare(std::get<0>(X), std::get<1>(X), std::get<0>(Y), std::get<1>(Y)); });
		auto last = X.cend() - 1;
		size_t count = 1;
		if (X.cbegin() == X.cend())
			throw std::invalid_argument("The supplied MPO is empty!");
		for (auto it = X.cbegin(); it != last; ++it)
		{
			auto range_A = *it;
			auto range_B = *(it + 1);
			count +=
			    !std::equal(std::get<0>(range_A), std::get<1>(range_A), std::get<0>(range_B), std::get<1>(range_B));
		}
		return count;
	}();
	return random_bMPS_impl(S, bond_dim, phys_dim, q_num, N_pass, opt);
}

} // namespace quantt