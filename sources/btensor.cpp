/*
 * File: btensor.cpp
 * Project: quantt
 * File Created: Monday, 12th October 2020 12:20:33 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Monday, 12th October 2020 12:20:34 pm
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#include "blockTensor/btensor.h"
#include "tensorgdot.h"
#include <ATen/ATen.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <algorithm>
#include <exception>
#include <torch/torch.h>
namespace quantt
{
void throw_on_bad_arg_blocks(size_t index, size_t block, size_t rank, size_t section_size)
{
	if (index >= rank)
		throw std::invalid_argument(fmt::format("given index {} is too large for rank {}.", index, rank));
	if (block >= section_size)
		std::invalid_argument(fmt::format("there are only {} blocks along the dimension {}. block requested {}", section_size, index, block));
}

size_t btensor::section_size(size_t index, size_t block) const
{
#ifndef NDEBUG
	throw_on_bad_arg_blocks(index, block, rank, sections_sizes[index]);
	qtt_REQUIRE(index < sections_by_dim.size());
#endif
	auto ori = std::accumulate(sections_by_dim.begin(), sections_by_dim.begin() + index, 0);
	return sections_sizes[ori + block];
}

any_quantity_cref btensor::section_conserved_qtt(size_t index, size_t block) const
{
#ifndef NDEBUG
	throw_on_bad_arg_blocks(index, block, rank, sections_sizes[index]);
#endif
	auto ori = std::accumulate(sections_by_dim.begin(), sections_by_dim.begin() + index, 0);
	return c_vals[ori + block];
}

std::tuple<size_t, any_quantity_cref> btensor::section_size_cqtt(size_t index, size_t block) const
{
#ifndef NDEBUG
	throw_on_bad_arg_blocks(index, block, rank, sections_sizes[index]);
#endif
	auto ori = std::accumulate(sections_by_dim.begin(), sections_by_dim.begin() + index, 0);
	return std::make_tuple(sections_sizes[ori + block], c_vals[ori + block]);
}
std::tuple<btensor::index_list::const_iterator, btensor::index_list::const_iterator> btensor::section_sizes(size_t dim) const
{
	auto ori = std::accumulate(sections_by_dim.begin(), sections_by_dim.begin() + dim, 0);
	return std::make_tuple(sections_sizes.begin() + ori, sections_sizes.begin() + ori + sections_by_dim[dim]);
}
std::tuple<any_quantity_vector::const_iterator, any_quantity_vector::const_iterator> btensor::section_cqtts(size_t dim) const
{
	auto ori = std::accumulate(sections_by_dim.begin(), sections_by_dim.begin() + dim, 0);
	return std::make_tuple(c_vals.begin() + ori, c_vals.begin() + ori + sections_by_dim[dim]);
}
std::tuple<btensor::index_list::const_iterator, btensor::index_list::const_iterator, any_quantity_vector::const_iterator, any_quantity_vector::const_iterator>
btensor::section_sizes_cqtts(size_t dim) const
{
	auto ori = std::accumulate(sections_by_dim.begin(), sections_by_dim.begin() + dim, 0);
	return std::make_tuple(sections_sizes.begin() + ori, sections_sizes.begin() + ori + sections_by_dim[dim], c_vals.begin() + ori, c_vals.begin() + ori + sections_by_dim[dim]);
}

void btensor::block_increment(btensor::index_list& block_index) const
{ //function to increment a block index
	TORCH_CHECK(block_index.size() == rank, fmt::format("block index invalid for this tensor: it has rank {} instead of expected {}", block_index.size(), rank));
	bool cond_add = true;
	for (size_t i = 0; i < rank; ++i) //reverse the loop to have right-major incrementation.
	{
		bool cond_reset = block_index[i] < (sections_by_dim[i] - 1) or !cond_add;
		block_index[i] = (cond_reset) * (block_index[i] + 1 * cond_add);
		cond_add &= !cond_reset;
	}
};
size_t btensor::btensor_compute_max_size(const btensor& btens, size_t max)
{
	size_t block_num = 0;
	btensor::index_list block_index(btens.rank, 0);
	//update max such that we can't go over the number of block when there are no selection rule.
	max = std::min(max, std::accumulate(btens.sections_by_dim.begin(), btens.sections_by_dim.end(), 1ul, [](auto&& a, auto&& b) { return a * b; })); //total number of blocks zero or not.
	for (size_t i = 0; i < max; ++i)
	{
		any_quantity qt = btens.selection_rule->neutral();
		auto qts = btens.block_quantities(block_index);
		for (const auto& q : qts)
		{
			qt += q;
		}
		block_num += (qt == btens.selection_rule); //add 1 if the selection rule is satisfied
		btens.block_increment(block_index);        //index to next block.
	}
	return block_num;
}

size_t tensor_list_size_guess(const btensor::init_list_t& list,
                              any_quantity_cref sel_rul, size_t rank, const btensor::index_list& sections_by_dims)
{
	constexpr size_t max_guess = 50;
	size_t guess = 0;
	// auto rank = list.size(); //computed the rank. will get it from the btensor instead. implies reordering of members.
	// btensor::index_list sections_by_dims(rank);
	// for (size_t i = 0; i < rank; ++i) //computed the number of section along each dims, will get it from the btensor instead. implies reordering of members.
	// {
	// 	sections_by_dims[i] = list.begin()[i].size();
	// }
	size_t block_num = std::accumulate(sections_by_dims.begin(), sections_by_dims.end(), 1, [](auto&& a, auto&& b) { return a * b; }); //total number of blocks zero or not.
	btensor::index_list block_index(rank, 0);
	auto increment = [&sections_by_dims, &rank](btensor::index_list& block_index) { //function to increment a block index, might be useful enough to break it out.
		                                                                            //in fact it increment any tensor index. might not be related to the memory layout.
		                                                                            //left-major incrementation. (i.e. column-major for matrices)
		bool cond_add = true;
		for (size_t i = 0; i < rank; ++i) //reverse the loop to have right-major incrementation.
		{
			bool cond_reset = block_index[i] < (sections_by_dims[i] - 1) or !cond_add;
			block_index[i] = (cond_reset) * (block_index[i] + 1 * cond_add);
			cond_add &= !cond_reset;
		}
	};
	for (size_t i = 0; i < block_num; ++i)
	{
		any_quantity qt = sel_rul.neutral();
		for (size_t j = 0; j < rank; ++j)
		{
			qt += std::get<1>(list.begin()[j].begin()[block_index[j]]);
		}
		guess += qt == sel_rul;
		if (guess == max_guess)
			break;
		increment(block_index);
	}
	return guess;
}
btensor::index_list block_shapes_from_struct_list(const btensor::init_list_t& list, size_t rank)
{
	btensor::index_list sections_by_dims(rank);
	for (size_t i = 0; i < rank; ++i) //computed the number of section along each dims, will get it from the btensor instead. implies reordering of members.
	{
		sections_by_dims[i] = list.begin()[i].size();
	}
	return sections_by_dims;
}
btensor::index_list block_sizes_from_struct_list(const btensor::init_list_t& list, btensor::index_list sections_by_dim)
{
	btensor::index_list section_sizes(std::accumulate(sections_by_dim.begin(), sections_by_dim.end(), 0));
	size_t i = 0;
	for (const auto& pair_list : list)
	{
		for (const auto& pair : pair_list)
		{
			section_sizes[i] = std::get<0>(pair);
			++i;
		}
	}
	return section_sizes;
}
any_quantity_vector c_vals_from_struct_list(const btensor::init_list_t& list, size_t size, any_quantity_cref sel_rul)
{
	any_quantity_vector c_vals(size, sel_rul.neutral());
	size_t i = 0;
	for (const auto& pair_list : list)
	{
		for (const auto& pair : pair_list)
		{
			any_quantity x = std::get<1>(pair);
			c_vals[i] = std::get<1>(pair);
			++i;
		}
	}
	return c_vals;
}

btensor::btensor(size_t _rank, block_list_t _blocks, index_list _sections_by_dims, index_list _block_sizes,
                 any_quantity_vector _c_vals, any_quantity _sel_rule)
    : selection_rule(std::move(_sel_rule)), rank(_rank), blocks(std::move(_blocks)), sections_by_dim(std::move(_sections_by_dims)), sections_sizes(std::move(_block_sizes)),
      c_vals(std::move(_c_vals))
{
	std::string check_result = check_tensor(*this);
	if (check_result.size())
		throw std::invalid_argument(fmt::format("Invalid argument to construct a block tensor: \n{}", check_result));
}
btensor::btensor(btensor::init_list_t dir_block_size_cqtt,
                 any_quantity_cref selection_rule)
    : selection_rule(std::move(selection_rule)), rank(dir_block_size_cqtt.size()),
      sections_by_dim(block_shapes_from_struct_list(dir_block_size_cqtt, rank)),
      sections_sizes(block_sizes_from_struct_list(dir_block_size_cqtt, sections_by_dim)),
      blocks(tensor_list_size_guess(dir_block_size_cqtt, selection_rule, rank, sections_by_dim)),
      c_vals(c_vals_from_struct_list(dir_block_size_cqtt, sections_sizes.size(), selection_rule))
{
#ifndef NDEBUG
	std::string check_result = check_tensor(*this);
	if (check_result.size())
		throw std::invalid_argument(fmt::format("Invalid argument to construct a block tensor: {}", check_result));
#endif
}
btensor::btensor(btensor::init_list_t dir_block_size_cqtt,
                 any_quantity_cref selection_rule, size_t num_blocks)
    : selection_rule(std::move(selection_rule)), rank(dir_block_size_cqtt.size()),
      sections_by_dim(block_shapes_from_struct_list(dir_block_size_cqtt, rank)),
      sections_sizes(block_sizes_from_struct_list(dir_block_size_cqtt, sections_by_dim)), blocks(num_blocks),
      c_vals(c_vals_from_struct_list(dir_block_size_cqtt, sections_sizes.size(), selection_rule))
{
	std::string check_result = check_tensor(*this);
	if (check_result.size())
		throw std::invalid_argument(fmt::format("Invalid argument to construct a block tensor: {}", check_result));
}
bool btensor::block_conservation_rule_test(const index_list& block_index) const
{
	any_quantity out = selection_rule.value.neutral();
	auto bl_qt = block_quantities(block_index);
	for (const auto& qt : bl_qt)
	{
		out += qt;
	}
	return out == selection_rule;
}
torch::Tensor& btensor::block_at(const index_list& block_index)
{
	return blocks.at(block_index);
}

/**
 * @brief This function always return something (exception such as out-of-memory are possible). It can return an 
 * uninitialized tensor, if the block index is too large in any direction or is not allowed by the selection rule.
 * If you do an in-place operation on the returned tensor and its' block_index is allowed, the change will be performed
 * within the block tensor.
 * 
 * 
 * @param block_index the block you wish to access
 * @return torch::Tensor the block you asked for.
 */
torch::Tensor& btensor::block(const index_list& block_index) //create the block if it is allowed, otherwise, return a freestanding sparrse tensor of zeros.
{
	if (!block_conservation_rule_test(block_index))
	{
		throw std::invalid_argument(fmt::format("block index {} not allowed by selection rule", block_index));
	}
	return blocks[block_index];
}

void btensor::throw_bad_tensor(const btensor& T)
{
	auto test_string = check_tensor(T);
	if (test_string != "")
		throw std::domain_error(test_string.c_str());
}
std::string btensor::check_tensor(const btensor& T)
{
	//things to check:
	//coherent redondent information:
	//    - the non-zero block have same sizes as stored in section_sizes
	//    - all the same rank, same as the number of block indexes.
	//    -
	//all non-zero block satisfy the conservation rule.
	std::string M = "";
	if (T.rank != T.sections_by_dim.size())
		M += fmt::format("rank ({}) incoherent with with internal sections_by_dim (size {})\n", T.rank, T.sections_by_dim.size());
	auto total_sections = std::accumulate(T.sections_by_dim.begin(), T.sections_by_dim.end(), 0);
	if (total_sections != T.sections_sizes.size())
		M += fmt::format("number of section accross all dimension ({}) incoherent with number of specified section sizes ({})\n", total_sections, T.sections_sizes.size());

	for (const auto& a : T.blocks)
	{
		auto& ind = std::get<0>(a);
		if (ind.size() != T.rank)
			M += fmt::format("block index {} invalid: number of index differ from rank", std::get<0>(a));
		any_quantity sel_test = T.selection_rule.value.neutral();
		{
			std::string cq = "";
			for (auto i = 0U; i < ind.size(); ++i)
			{
				if (!(ind[i] < T.sections_by_dim[i]))
					M += fmt::format("block index {} {}th element is greater than the number of section along that dimension ({})\n", ind, i, T.sections_by_dim[i]);
				auto qt = T.section_conserved_qtt(i, ind[i]);
				sel_test += qt;
				cq += fmt::format("index {}: ", i) +
				      fmt::format("{}\n", qt);
			}
			if (sel_test != T.selection_rule)
			{
				M += fmt::format("block with index {} incompatible with selection rule {}.\n conserved quantities of the block: \n {}", ind, T.selection_rule.value, cq);
			}
		} //destroy cq
		auto sizes = std::get<1>(a).sizes();
		if (sizes.size() != T.rank)
			M += fmt::format("block with index {} has rank ({}) incompatible with the btensor ({})\n", ind, sizes.size(), T.rank);
		else
		{
			std::string sub = "";
			for (auto i = 0U; i < T.rank; ++i)
			{
				if (T.section_size(i, ind[i]) != sizes[i])
					sub += fmt::format("\t- {}th dimension size incompatible: btensor has {} and block {}\n", i, T.section_size(i, ind[i]), sizes[i]);
			}
			if (sub != "")
			{
				M += fmt::format("for block index {}:\n{}", ind, sub);
			}
		}
	}
	return M;
}

btensor::const_block_qtt_view btensor::block_quantities(index_list block_index) const
{
	return const_block_qtt_view(c_vals.cbegin(), c_vals.cend(), sections_by_dim, std::move(block_index));
}
// btensor::block_qtt_view btensor::block_quantities(index_list block_index)
// {
// 	return block_qtt_view(c_vals.begin(), c_vals.end(), sections_by_dim, std::move(block_index));
// }
btensor::const_block_size_view btensor::block_sizes(index_list block_index) const
{
	auto a = sections_sizes.begin();
	return const_block_size_view(sections_sizes.begin(), sections_sizes.end(), sections_by_dim, std::move(block_index));
}

/**
 * @brief preparatory work and error checking for the contraction of two btensors.
 * 
 * @param input1 
 * @param input2 
 * @param dims1 
 * @param dims2 
 * @return std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>> 
 */
std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>
compute_tdot_shape(const btensor& input1, const btensor& input2,
                   torch::IntArrayRef dims1, torch::IntArrayRef dims2)
{
	TORCH_CHECK(dims1.size() == dims2.size(), "both dimension lists should have the same length.")
	TORCH_CHECK(input1.selection_rule->get().same_type(input2.selection_rule->get()), "the two tensors have different type of conserved quantities");

	for (size_t i = 0; i < dims1.size(); ++i)
	{
		int s1 = input1.section_number(dims1[i]);
		int s2 = input2.section_number(dims2[i]);
		constexpr auto mismatch = "contracted dimensions need to match, but first "
		                          "has {} sections along dim {}  and second has {} sections along dim {}";
		TORCH_CHECK(s1 == s2, fmt::format(mismatch, s1, dims1[i], s2, dims2[i]));
		//no broadcast dimension like torch::tensordot. i can't think of a way for it to make sense
		// with the quantum number.
	}
	auto cdims1 = at::dim_list_to_bitset(dims1, input1.dim());
	auto cdims2 = at::dim_list_to_bitset(dims2, input2.dim());
	std::vector<int64_t> p1, p2, out_section_by_dim; // p1, p2: input permutations, out_section_by_dim: sizes of the result
	p1.reserve(input1.dim());
	p2.reserve(input2.dim());
	out_section_by_dim.reserve(input1.dim() + input2.dim() - static_cast<int64_t>(dims1.size()));

	// fill the permutations and compute sizes
	for (int64_t i = 0; i < input1.dim(); i++)
	{
		if (!cdims1[i])
		{
			p1.emplace_back(i);
			out_section_by_dim.emplace_back(input1.section_number(i));
		}
	}
	for (size_t i = 0; i < dims1.size(); i++)
	{
		p1.emplace_back(dims1[i]);
	}
	for (size_t i = 0; i < dims2.size(); i++)
	{
		p2.emplace_back(dims2[i]);
	}
	for (int64_t i = 0; i < input2.dim(); i++)
	{
		if (!cdims2[i])
		{
			p2.emplace_back(i);
			out_section_by_dim.emplace_back(input2.section_number(i));
		}
	}
	return make_tuple(p1, p2, out_section_by_dim);
}
void btensor::swap(btensor& other)
{
	using std::swap;
	swap(rank, other.rank);
	swap(c_vals, other.c_vals);
	blocks.swap(other.blocks);
	swap(selection_rule.value, other.selection_rule.value);
	swap(sections_by_dim, other.sections_by_dim);
	swap(sections_sizes, other.sections_sizes);
}
btensor btensor::permute(torch::IntArrayRef permutation) const
{
	block_list_t::content_t out_block_list; //unordered.
	out_block_list.reserve(blocks.size());
	index_list out_section_by_dim;
	//permute the tensors and their position in the block matrix
	for (auto& block : blocks)
	{
		auto permute_index = [rank = this->rank](auto permutation, auto& index) {
			btensor::index_list out(rank);
			for (int i = 0; i < rank; ++i)
			{
				out[i] = index[permutation[i]];
			}
			return out;
		};
		out_block_list.emplace_back(permute_index(permutation, std::get<0>(block)), std::get<1>(block).permute(permutation));
	}
	any_quantity_vector out_c_vals = c_vals.permute(permutation.begin(), permutation.end(), sections_by_dim);
	index_list out_section_sizes(sections_sizes.size());
	{
		size_t p = 0;
		for (auto perm : permutation) //to properly break this up would while reusing the s value would require a coroutine.
		{
			auto rep = sections_by_dim[perm];
			auto s = std::accumulate(sections_by_dim.begin(), sections_by_dim.begin() + perm, 0);
			for (size_t i = 0; i < rep; ++i, ++p)
			{
				out_section_sizes[p] = sections_sizes[s + i];
			}
		}
	}

	return btensor(rank, block_list_t(std::move(out_block_list)), std::move(out_section_by_dim),
	               std::move(out_section_sizes), std::move(out_c_vals), selection_rule.value);
}
btensor& btensor::permute_(torch::IntArrayRef permutation)
{
	auto new_val = permute(permutation);
	swap(new_val); //temporary. a better implementation will come.
	return *this;
}

btensor::block_list_t permute_bl(const btensor::block_list_t& block_list, torch::IntArrayRef block_permutation, torch::IntArrayRef tensor_permutation)
{
	auto out = block_list;
	auto tmp_index = std::get<0>(*out.begin());
	auto ind_l = tmp_index.size();
	for (auto& block : out)
	{
		for (auto i = 0 * ind_l; i < ind_l; ++i)
		{
			tmp_index[i] = std::get<0>(block)[block_permutation[i]];
		}
		tmp_index.swap(std::get<0>(block));
		std::get<1>(block) = std::get<1>(block).permute(tensor_permutation);
	}
	out.sort();
	return out;
}

btensor::btensor(index_list _sections_by_dim, any_quantity_vector _c_vals, index_list _section_sizes,
                 any_quantity _sel_rule)
    : selection_rule(std::move(_sel_rule)), rank(_sections_by_dim.size()), sections_by_dim(std::move(_sections_by_dim)),
      sections_sizes(std::move(_section_sizes)), blocks(), c_vals(std::move(_c_vals))
{
	std::string check_result = check_tensor(*this);
	if (check_result.size())
		throw std::invalid_argument(fmt::format("Invalid argument to construct a block tensor: \n{}", check_result));
}

std::tuple<any_quantity_vector, btensor::index_list>
compute_tdot_cval_sectSize(const btensor& left, const btensor& right, torch::IntArrayRef perm_left, torch::IntArrayRef perm_right, size_t dim_l, size_t out_l)
{
	any_quantity_vector out_cvals(0, right.selection_rule->neutral());
	out_cvals.reserve(out_l);
	btensor::index_list out_section_sizes;
	out_section_sizes.reserve(out_l);
	auto inds_from_left = perm_left.size() - dim_l;
	auto inds_from_right = perm_right.size() - dim_l;
	auto max_i = perm_left.size() - dim_l;
	for (size_t i = 0; i < max_i; ++i)
	{
		auto [sec_size_beg, sec_size_end, qtt_beg, qtt_end] = left.section_sizes_cqtts(perm_left[i]);
		out_section_sizes.insert(out_section_sizes.end(), sec_size_beg, sec_size_end);
		out_cvals.insert(out_cvals.end(), qtt_beg, qtt_end);
	}
	max_i = perm_right.size();
	for (size_t i = dim_l; i < max_i; ++i)
	{
		auto [sec_size_beg, sec_size_end, qtt_beg, qtt_end] = right.section_sizes_cqtts(perm_right[i]);
		out_section_sizes.insert(out_section_sizes.end(), sec_size_beg, sec_size_end);
		out_cvals.insert(out_cvals.end(), qtt_beg, qtt_end);
	}
	return std::make_tuple(out_cvals, out_section_sizes);
}

btensor btensor::tensordot(const btensor& other, torch::IntArrayRef dim_self, torch::IntArrayRef dims_other) const
{
	auto dim_l = dim_self.size();
	//first check that everything matches, and compute the output properties, at the block level.
	auto [t1, t2, out_btens] = [&]() {
		auto [p1, p2, out_section_by_dim] = compute_tdot_shape(*this, other, dim_self, dims_other);
		auto l = std::accumulate(out_section_by_dim.begin(), out_section_by_dim.end(), 0);
		auto out_sel_rule = selection_rule.value + other.selection_rule.value;
		auto _t1 = permute_bl(blocks, p1, p1);
		auto [out_cvals, out_section_sizes] = compute_tdot_cval_sectSize(*this, other, p1, p2, dim_l, l);
		//swap the permutation for better ordering of the loops with the algorithm.
		std::vector<int64_t> p2_prime(p2.size());
		std::copy_backward(p2.begin(), p2.begin() + dim_l, p2_prime.end());
		std::copy(p2.begin() + dim_l, p2.end(), p2_prime.begin());
		auto _t2 = permute_bl(other.blocks, p2_prime, p2);
		btensor out(out_section_by_dim, out_cvals, out_section_sizes, std::move(out_sel_rule));
		return std::make_tuple(std::move(_t1), std::move(_t2), std::move(out));
	}(); //a lambda that capture everything that we call immediatly. leave us with a somewhat clean namespace in the scope.

	fmt::print("out_btens section sizes {}\n", out_btens.sections_sizes);
	fmt::print("input section sizes {}\n", other.sections_sizes);

	auto next_index = [dim_l](const auto& iterator) {
		auto next = std::get<0>(*iterator);
		++next[dim_l - 1];
		for (auto it = next.begin() + dim_l; it != next.end(); ++it)
			*it = 0;
		return next;
	};
	{ //launch the contractions.
		auto this_curr_col_start = t1.begin();
		auto less = t1.value_comp();
		std::vector<int64_t> out_block_index(out_btens.rank);
		auto row_less = [dim_l](const auto& this_block, const auto& other_block) {
			auto this_it = this_block.end() - dim_l;
			auto other_it = other_block.end() - dim_l;
			bool result = true;
			for (; this_it != this_block.end() and result; ++this_it, ++other_it)
			{
				result &= (*this_it) < (*other_it);
			}
			return result;
		};
		//#pragma omp parallel
		//#pragma omp single //we've got a threadpool, but only one is working for now.
		while (this_curr_col_start != t1.end()) //loop over all the content of this
		{
			auto it_other_block = t2.begin();
			auto this_next_col_start = std::lower_bound(this_curr_col_start, t1.end(), next_index(this_curr_col_start), less);
			while (it_other_block != t2.end()) //loop over all the content of other
			{                                  //one pass trough this scope scan a whole column of the other tensor
				auto this_curr_block = this_curr_col_start;
				auto other_next_col_start = std::lower_bound(it_other_block, t2.end(), next_index(it_other_block), less);
				std::copy(std::get<0>(*this_curr_block).begin(), std::get<0>(*this_curr_block).end() - dim_l, out_block_index.begin());
				std::copy_backward(std::get<0>(*it_other_block).begin(), std::get<0>(*it_other_block).end() - dim_l, out_block_index.end());
				//first determine if there's a contraction to do for this output position.
				int hit = 0;
				//must first check that the first element of the two rows ain't a match.
				while (!hit and it_other_block != other_next_col_start and this_curr_block != this_next_col_start)
				{
					//compute output position
					while (row_less(std::get<0>(*this_curr_block), std::get<0>(*it_other_block)) and this_curr_block != this_next_col_start)
					{ //this loop could be replaced with a lower_bound on [this_curr_block, this_next_col_start[
						++this_curr_block;
					}
					if (this_curr_block == this_next_col_start)
						break;
					if (!row_less(std::get<0>(*it_other_block), std::get<0>(*this_curr_block)))
					{
						hit = 1;
						auto size_range = out_btens.block_sizes(out_block_index);
						fmt::print("1 at block {}\n", out_block_index);
						out_btens.block(out_block_index) = torch::zeros(std::vector<int64_t>(size_range.begin(), size_range.end()),
						                                                std::get<1>(*it_other_block).options()); //initialize the block.
						break;
					}
					while (row_less(std::get<0>(*it_other_block), std::get<0>(*this_curr_block)) and it_other_block != other_next_col_start)
					{ //this loop could be replaced with a lower_bound on [it_other_block, other_next_col_start[
						++it_other_block;
					}
					if (it_other_block == other_next_col_start)
						break;
					if (!row_less(std::get<0>(*this_curr_block), std::get<0>(*it_other_block)))
					{
						hit = 2;
						fmt::print("2 at block {}\n", out_block_index);
						auto size_range = out_btens.block_sizes(out_block_index);
						out_btens.block(out_block_index) = torch::zeros(std::vector<int64_t>(size_range.begin(), size_range.end()),
						                                                std::get<1>(*it_other_block).options()); //initialize the block.
						break;
					}
				}
				//if there is, start doing the contractions.
				// #pragma omp task //parallel section
				//Before activating openmp here, we need to make thread local copies of the iterators, block index and hit.
				if (hit)
				{
					if (hit == 2)
					{
						quantt::tensorgdot_(out_btens.block_at(out_block_index), std::get<1>(*this_curr_block), std::get<1>(*it_other_block), dim_l);
						++it_other_block;
					}
					while (it_other_block != other_next_col_start and this_curr_block != this_next_col_start)
					{
						//compute output position
						while (row_less(std::get<0>(*this_curr_block), std::get<0>(*it_other_block)) and this_curr_block != this_next_col_start)
						{ //this loop could be replaced with a lower_bound on [this_curr_block, this_next_col_start[
							++this_curr_block;
						}
						if (this_curr_block == this_next_col_start)
							break;
						if (!row_less(std::get<0>(*it_other_block), std::get<0>(*this_curr_block)) and this_curr_block != this_next_col_start)
						{
							quantt::tensorgdot_(out_btens.block_at(out_block_index), std::get<1>(*this_curr_block), std::get<1>(*it_other_block), dim_l);
							++this_curr_block;
							if (this_curr_block == this_next_col_start)
								break;
						}
						while (row_less(std::get<0>(*it_other_block), std::get<0>(*this_curr_block)) and it_other_block != other_next_col_start)
						{ //this loop could be replaced with a lower_bound on [it_other_block, other_next_col_start[
							++it_other_block;
						}
						if (it_other_block == other_next_col_start)
							break;
						if (!row_less(std::get<0>(*this_curr_block), std::get<0>(*it_other_block)) and it_other_block != other_next_col_start)
						{
							quantt::tensorgdot_(out_btens.block_at(out_block_index), std::get<1>(*this_curr_block), std::get<1>(*it_other_block), dim_l);
							++it_other_block;
						}
					}
				}
				it_other_block = other_next_col_start;
			}
			this_curr_col_start = this_next_col_start;
		}
	}
	return out_btens;
}

} // namespace quantt