/*
 * File: btensor.cpp
 * Project: quantt
 * File Created: Monday, 12th October 2020 12:20:33 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 *
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */
#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include "blockTensor/btensor.h"
#include "tensorgdot.h"
#include <ATen/ATen.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <algorithm>
#include <exception>
#include <execution>
#include <numeric>
#include <torch/torch.h>
namespace quantt
{
void throw_on_bad_arg_blocks(size_t index, size_t block, size_t rank, size_t section_size)
{
	if (index >= rank)
		throw std::invalid_argument(fmt::format("given index {} is too large for rank {}.", index, rank));
	if (block >= section_size)
		std::invalid_argument(fmt::format("there are only {} blocks along the dimension {}. block requested {}",
		                                  section_size, index, block));
}

size_t btensor::section_size(size_t index, size_t block) const
{
#ifndef NDEBUG
	throw_on_bad_arg_blocks(index, block, rank, sections_sizes[index]);
	qtt_REQUIRE(index < sections_by_dim.size());
#endif
	auto ori = std::reduce(sections_by_dim.begin(), sections_by_dim.begin() + index, 0);
	return sections_sizes[ori + block];
}

any_quantity_cref btensor::section_conserved_qtt(size_t index, size_t block) const
{
#ifndef NDEBUG
	throw_on_bad_arg_blocks(index, block, rank, sections_sizes[index]);
#endif
	auto ori = std::reduce(sections_by_dim.begin(), sections_by_dim.begin() + index, 0);
	return c_vals[ori + block];
}
std::tuple<any_quantity_vector::const_iterator, any_quantity_vector::const_iterator> btensor::
    section_conserved_qtt_range(size_t index) const
{
	if (index >= rank)
		throw std::invalid_argument(
		    fmt::format("the given index {} is greater than the rank of the tensor {}", index, rank));
	auto ori = std::reduce(sections_by_dim.begin(), sections_by_dim.begin() + index, 0);
	return std::make_tuple(c_vals.begin() + ori, c_vals.begin() + ori + sections_by_dim[index]);
}

std::tuple<size_t, any_quantity_cref> btensor::section_size_cqtt(size_t index, size_t block) const
{
#ifndef NDEBUG
	throw_on_bad_arg_blocks(index, block, rank, sections_sizes[index]);
#endif
	auto ori = std::reduce(sections_by_dim.begin(), sections_by_dim.begin() + index, 0);
	return std::make_tuple(sections_sizes[ori + block], c_vals[ori + block]);
}
std::tuple<btensor::index_list::const_iterator, btensor::index_list::const_iterator> btensor::section_sizes(
    size_t dim) const
{
	auto ori = std::reduce(sections_by_dim.begin(), sections_by_dim.begin() + dim, 0);
	return std::make_tuple(sections_sizes.begin() + ori, sections_sizes.begin() + ori + sections_by_dim[dim]);
}
std::tuple<any_quantity_vector::const_iterator, any_quantity_vector::const_iterator> btensor::section_cqtts(
    size_t dim) const
{
	auto ori = std::reduce(sections_by_dim.begin(), sections_by_dim.begin() + dim, 0);
	return std::make_tuple(c_vals.begin() + ori, c_vals.begin() + ori + sections_by_dim[dim]);
}
std::tuple<btensor::index_list::const_iterator, btensor::index_list::const_iterator,
           any_quantity_vector::const_iterator, any_quantity_vector::const_iterator>
btensor::section_sizes_cqtts(size_t dim) const
{
	auto ori = std::reduce(sections_by_dim.begin(), sections_by_dim.begin() + dim, 0);
	return std::make_tuple(sections_sizes.begin() + ori, sections_sizes.begin() + ori + sections_by_dim[dim],
	                       c_vals.begin() + ori, c_vals.begin() + ori + sections_by_dim[dim]);
}

void btensor::block_increment(btensor::index_list &block_index) const
{ // function to increment a block index
	if (block_index.size() != rank)
		throw std::invalid_argument(fmt::format(
		    "block index invalid for this tensor: it has rank {} instead of expected {}", block_index.size(), rank));
	bool cond_add = true;
	for (size_t i = 0; i < rank; ++i) // reverse the loop to have right-major incrementation.
	{
		bool cond_reset = block_index[i] < (sections_by_dim[i] - 1) or !cond_add;
		block_index[i] = (cond_reset) * (block_index[i] + 1 * cond_add);
		cond_add &= !cond_reset;
	}
};
size_t btensor::btensor_compute_max_size(const btensor &btens, size_t max)
{
	size_t block_num = 0;
	btensor::index_list block_index(btens.rank, 0);
	// update max such that we can't go over the number of block when there are no selection rule.
	max = std::min(max, std::reduce(btens.sections_by_dim.begin(), btens.sections_by_dim.end(), 1ul,
	                                [](auto &&a, auto &&b) { return a * b; })); // total number of blocks zero or not.
	for (size_t i = 0; i < max; ++i)
	{
		any_quantity qt = btens.selection_rule->neutral();
		auto qts = btens.block_quantities(block_index);
		for (const auto &q : qts)
		{
			qt += q;
		}
		block_num += (qt == btens.selection_rule); // add 1 if the selection rule is satisfied
		btens.block_increment(block_index);        // index to next block.
	}
	return block_num;
}

/**
 * @brief Increment a tensor index, right-most index are incremented first.
 *
 * for exemple: [0,0,0], with a sizes [2,3,2] will be incremented to [0,0,1] and [0,0,1] increment to [0,1,0].
 *
 * @param index index to increment, in-out argument
 * @param sizes The size of tensor along each dimension.
 * @param rank rank of the tensor, to avoid recomputing it everytime
 */
void increment_index_right(btensor::index_list &index, torch::IntArrayRef sizes, size_t rank)
{
	bool cond_add = true;
	for (size_t i = rank; i > 0; --i) // reverse the loop to have left-major incrementation.
	{
		auto k = i - 1;
		bool cond_reset = index[k] < (sizes[k] - 1) or !cond_add;
		index[k] = (cond_reset) * (index[k] + 1 * cond_add);
		cond_add &= !cond_reset;
	}
}
/**
 * @brief Increment a tensor index, left-most index are incremented first.
 *
 * for exemple: [0,0,0], with a sizes [2,3,2], will be incremented to [1,0,0] and [1,0,0] increment to [0,1,0].
 *
 * @param index index to increment, in-out argument
 * @param sizes The size of tensor along each dimension.
 * @param rank rank of the tensor, to avoid recomputing it everytime
 */
void increment_index_left(btensor::index_list &index, torch::IntArrayRef max_index, size_t rank)
{
	bool cond_add = true;
	for (size_t i = 0; i < rank; ++i) // reverse the loop to have right-major incrementation.
	{
		bool cond_reset = index[i] < (max_index[i] - 1) or !cond_add;
		index[i] = (cond_reset) * (index[i] + 1 * cond_add);
		cond_add &= !cond_reset;
	}
}
/**
 * @brief if any of the element in the range convert to true, return true.
 *
 * @return true at least one element converts to true
 * @return false no element convert to true
 */
template <class T>
bool any_truth(const T &in)
{
	bool out = false;
	for (auto &it : in)
	{
		out |= bool(it);
		if (out)
			break;
	}
	return out;
}
size_t tensor_list_size_guess(const btensor::init_list_t &list, any_quantity_cref sel_rul, size_t rank,
                              const btensor::index_list &sections_by_dims)
{
	constexpr size_t max_guess = 50;
	size_t guess = 0;
	// auto rank = list.size(); //computed the rank. will get it from the btensor instead. implies reordering of
	// members. btensor::index_list sections_by_dims(rank); for (size_t i = 0; i < rank; ++i) //computed the number of
	// section along each dims, will get it from the btensor instead. implies reordering of members.
	// {
	// 	sections_by_dims[i] = list.begin()[i].size();
	// }
	size_t block_num = std::reduce(sections_by_dims.begin(), sections_by_dims.end(), 1,
	                               [](auto &&a, auto &&b) { return a * b; }); // total number of blocks zero or not.
	btensor::index_list block_index(rank, 0);
	auto increment = [&sections_by_dims, &rank](btensor::index_list &block_index) {
		increment_index_left(block_index, sections_by_dims, rank);
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
btensor::index_list block_shapes_from_struct_list(const btensor::init_list_t &list, size_t rank)
{
	btensor::index_list sections_by_dims(rank);
	for (size_t i = 0; i < rank; ++i) // computed the number of section along each dims, will get it from the btensor
	                                  // instead. implies reordering of members.
	{
		sections_by_dims[i] = list.begin()[i].size();
	}
	return sections_by_dims;
}
btensor::index_list block_sizes_from_struct_list(const btensor::init_list_t &list, btensor::index_list sections_by_dim)
{
	btensor::index_list section_sizes(std::reduce(sections_by_dim.begin(), sections_by_dim.end(), 0));
	size_t i = 0;
	for (const auto &pair_list : list)
	{
		for (const auto &pair : pair_list)
		{
			section_sizes[i] = std::get<0>(pair);
			++i;
		}
	}
	return section_sizes;
}
any_quantity_vector c_vals_from_struct_list(const btensor::init_list_t &list, size_t size, any_quantity_cref sel_rul)
{
	any_quantity_vector c_vals(size, sel_rul.neutral());
	size_t i = 0;
	for (const auto &pair_list : list)
	{
		for (const auto &pair : pair_list)
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
    : selection_rule(std::move(_sel_rule)), rank(_rank), blocks(std::move(_blocks)),
      sections_by_dim(std::move(_sections_by_dims)), sections_sizes(std::move(_block_sizes)), c_vals(std::move(_c_vals))
{
	std::string check_result = check_tensor(*this);
	if (check_result.size())
		throw std::invalid_argument(fmt::format("Invalid argument to construct a block tensor: \n{}", check_result));
}
btensor::btensor(btensor::init_list_t dir_block_size_cqtt, any_quantity_cref selection_rule)
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
btensor::btensor(btensor::init_list_t dir_block_size_cqtt, any_quantity_cref selection_rule, size_t num_blocks)
    : selection_rule(std::move(selection_rule)), rank(dir_block_size_cqtt.size()),
      sections_by_dim(block_shapes_from_struct_list(dir_block_size_cqtt, rank)),
      sections_sizes(block_sizes_from_struct_list(dir_block_size_cqtt, sections_by_dim)), blocks(num_blocks),
      c_vals(c_vals_from_struct_list(dir_block_size_cqtt, sections_sizes.size(), selection_rule))
{
	std::string check_result = check_tensor(*this);
	if (check_result.size())
		throw std::invalid_argument(fmt::format("Invalid argument to construct a block tensor: {}", check_result));
}
bool btensor::block_conservation_rule_test(index_list block_index) const
{
	any_quantity out = selection_rule.value.neutral();
	auto bl_qt = block_quantities(std::move(block_index));
	for (const auto &qt : bl_qt)
	{
		out += qt;
	}
	return out == selection_rule;
}
torch::Tensor &btensor::block_at(const index_list &block_index) { return blocks.at(block_index); }

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
torch::Tensor &btensor::block(const index_list &block_index) // create the block if it is allowed, otherwise, return a
                                                             // freestanding sparrse tensor of zeros.
{
	if (!block_conservation_rule_test(block_index))
	{
		throw std::invalid_argument(fmt::format("block index {} not allowed by selection rule", block_index));
	}
	return blocks[block_index];
}

void btensor::throw_bad_tensor(const btensor &T)
{
	auto test_string = check_tensor(T);
	if (test_string != "")
		throw std::domain_error(test_string.c_str());
}
std::string btensor::check_tensor(const btensor &T)
{
	// things to check:
	// coherent redondent information:
	//    - the non-zero block have same sizes as stored in section_sizes
	//    - all the same rank, same as the number of block indexes.
	//    -
	// all non-zero block satisfy the conservation rule.
	std::string M = "";
	if (T.rank != T.sections_by_dim.size())
		M += fmt::format("rank ({}) incoherent with with internal sections_by_dim (size {})\n", T.rank,
		                 T.sections_by_dim.size());
	auto total_sections = std::reduce(T.sections_by_dim.begin(), T.sections_by_dim.end(), 0);
	if (total_sections != T.sections_sizes.size())
		M += fmt::format(
		    "number of section accross all dimension ({}) incoherent with number of specified section sizes ({})\n",
		    total_sections, T.sections_sizes.size());

	for (const auto &a : T.blocks)
	{
		auto &ind = std::get<0>(a);
		if (ind.size() != T.rank)
			M += fmt::format("block index {} invalid: number of index differ from rank", std::get<0>(a));
		any_quantity sel_test = T.selection_rule.value.neutral();
		{
			std::string cq = "";
			for (auto i = 0U; i < ind.size(); ++i)
			{
				if (!(ind[i] < T.sections_by_dim[i]))
					M += fmt::format(
					    "block index {} {}th element is greater than the number of section along that dimension ({})\n",
					    ind, i, T.sections_by_dim[i]);
				auto qt = T.section_conserved_qtt(i, ind[i]);
				sel_test += qt;
				cq += fmt::format("index {}: ", i) + fmt::format("{}\n", qt);
			}
			if (sel_test != T.selection_rule)
			{
				M += fmt::format("block with index {} incompatible with selection rule {}.\n conserved quantities of "
				                 "the block: \n {}",
				                 ind, T.selection_rule.value, cq);
			}
		} // destroy cq
		auto sizes = std::get<1>(a).sizes();
		if (sizes.size() != T.rank)
			M += fmt::format("block with index {} has rank ({}) incompatible with the btensor ({})\n", ind,
			                 sizes.size(), T.rank);
		else
		{
			std::string sub = "";
			for (auto i = 0U; i < T.rank; ++i)
			{
				if (T.section_size(i, ind[i]) != sizes[i])
					sub += fmt::format("\t- {}th dimension size incompatible: btensor has {} and block {}\n", i,
					                   T.section_size(i, ind[i]), sizes[i]);
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
std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>> compute_tdot_shape(
    const btensor &input1, const btensor &input2, torch::IntArrayRef dims1, torch::IntArrayRef dims2)
{
	TORCH_CHECK(dims1.size() == dims2.size(), "both dimension lists should have the same length.")
	TORCH_CHECK(input1.selection_rule->get().same_type(input2.selection_rule->get()),
	            "the two tensors have different type of conserved quantities");

	for (size_t i = 0; i < dims1.size(); ++i)
	{
		int s1 = input1.section_number(dims1[i]);
		int s2 = input2.section_number(dims2[i]);
		auto q1 = input1.section_conserved_qtt_range(dims1[i]);
		auto q2 = input1.section_conserved_qtt_range(dims2[i]);
		constexpr auto mismatch = "contracted dimensions need to match, but first "
		                          "has {} sections along dim {}  and second has {} sections along dim {}";
		constexpr auto mismatch_cons_rule =
		    "contracted conserved numbers need to sum to zero, but there "
		    "is a violation when contracting dim {} of the left tensor with dim {} of the right tensor";
		TORCH_CHECK(s1 == s2, fmt::format(mismatch, s1, dims1[i], s2, dims2[i]));
		TORCH_CHECK(std::equal(std::get<0>(q1), std::get<1>(q1), std::get<0>(q2)),
		            fmt::format(mismatch, dims1[i], dims2[i]));
		// no broadcast dimension like torch::tensordot. i can't think of a way for it to make sense
		// with the quantum number.
	}
	auto cdims1 = at::dim_list_to_bitset(dims1, input1.dim());
	auto cdims2 = at::dim_list_to_bitset(dims2, input2.dim());
	std::vector<int64_t> p1, p2,
	    out_section_by_dim; // p1, p2: input permutations, out_section_by_dim: sizes of the result
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
void btensor::swap(btensor &other)
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
	block_list_t::content_t out_block_list; // unordered.
	out_block_list.reserve(blocks.size());
	index_list out_section_by_dim;
	// permute the tensors and their position in the block matrix
	for (auto &block : blocks)
	{
		auto permute_index = [rank = this->rank](auto permutation, auto &index) {
			btensor::index_list out(rank);
			for (int i = 0; i < rank; ++i)
			{
				out[i] = index[permutation[i]];
			}
			return out;
		};
		out_block_list.emplace_back(permute_index(permutation, std::get<0>(block)),
		                            std::get<1>(block).permute(permutation));
	}
	any_quantity_vector out_c_vals = c_vals.permute(permutation.begin(), permutation.end(), sections_by_dim);
	index_list out_section_sizes(sections_sizes.size());
	{
		size_t p = 0;
		for (auto perm :
		     permutation) // to properly break this up would while reusing the s value would require a coroutine.
		{
			auto rep = sections_by_dim[perm];
			auto s = std::reduce(sections_by_dim.begin(), sections_by_dim.begin() + perm, 0);
			for (size_t i = 0; i < rep; ++i, ++p)
			{
				out_section_sizes[p] = sections_sizes[s + i];
			}
		}
	}

	return btensor(rank, block_list_t(std::move(out_block_list)), std::move(out_section_by_dim),
	               std::move(out_section_sizes), std::move(out_c_vals), selection_rule.value);
}
btensor &btensor::permute_(torch::IntArrayRef permutation)
{
	auto new_val = permute(permutation);
	swap(new_val); // TODO: temporary. a better implementation will come.
	return *this;
}

btensor::block_list_t permute_bl(const btensor::block_list_t &block_list, torch::IntArrayRef block_permutation,
                                 torch::IntArrayRef tensor_permutation)
{
	auto out = block_list;
	auto tmp_index = std::get<0>(*out.begin());
	auto ind_l = tmp_index.size();
	for (auto &block : out)
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

std::tuple<any_quantity_vector, btensor::index_list> compute_tdot_cval_sectSize(const btensor &left,
                                                                                const btensor &right,
                                                                                torch::IntArrayRef perm_left,
                                                                                torch::IntArrayRef perm_right,
                                                                                size_t dim_l, size_t out_l)
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

btensor btensor::tensordot(const btensor &other, torch::IntArrayRef dim_self, torch::IntArrayRef dims_other) const
{
	auto dim_l = dim_self.size();
	// first check that everything matches, and compute the output properties, at the block level.
	auto [t1, t2, out_btens] = [&]() {
		auto [p1, p2, out_section_by_dim] = compute_tdot_shape(*this, other, dim_self, dims_other);
		auto l = std::reduce(out_section_by_dim.begin(), out_section_by_dim.end(), 0);
		auto out_sel_rule = selection_rule.value + other.selection_rule.value;
		auto _t1 = permute_bl(blocks, p1, p1);
		auto [out_cvals, out_section_sizes] = compute_tdot_cval_sectSize(*this, other, p1, p2, dim_l, l);
		// swap the permutation for better ordering of the loops with the algorithm.
		std::vector<int64_t> p2_prime(p2.size());
		std::copy_backward(p2.begin(), p2.begin() + dim_l, p2_prime.end());
		std::copy(p2.begin() + dim_l, p2.end(), p2_prime.begin());
		auto _t2 = permute_bl(other.blocks, p2_prime, p2);
		btensor out(out_section_by_dim, out_cvals, out_section_sizes, std::move(out_sel_rule));
		return std::make_tuple(std::move(_t1), std::move(_t2), std::move(out));
	}(); // a lambda that capture everything that we call immediatly. leave us with a somewhat clean namespace in the
	     // scope.

	fmt::print("out_btens section sizes {}\n", out_btens.sections_sizes);
	fmt::print("input section sizes {}\n", other.sections_sizes);

	auto next_index = [dim_l](const auto &iterator) {
		auto next = std::get<0>(*iterator);
		++next[dim_l - 1];
		for (auto it = next.begin() + dim_l; it != next.end(); ++it)
			*it = 0;
		return next;
	};
	{ // launch the contractions.
		auto this_curr_col_start = t1.begin();
		auto less = t1.value_comp();
		std::vector<int64_t> out_block_index(out_btens.rank);
		auto row_less = [dim_l](const auto &this_block, const auto &other_block) {
			auto this_it = this_block.end() - dim_l;
			auto other_it = other_block.end() - dim_l;
			bool result = true;
			for (; this_it != this_block.end() and result; ++this_it, ++other_it)
			{
				result &= (*this_it) < (*other_it);
			}
			return result;
		};
		auto find_next_match = [](auto a_beg, const auto &a_end, auto b_beg, const auto &b_end, auto &&row_less) {
			// if we have a match both of the following boolean are false
			bool a_smaller_b = true;
			bool b_smaller_a = true;
			while ((a_beg != a_end and b_beg != b_end) and (a_smaller_b or b_smaller_a))
			{
				a_smaller_b = row_less(std::get<0>(*a_beg), std::get<0>(*b_beg));
				b_smaller_a = row_less(std::get<0>(*b_beg), std::get<0>(*a_beg));
				a_beg += a_smaller_b; // increment only the smaller one.
				b_beg += b_smaller_a;
			}
			if (a_beg == a_end or b_beg == b_end) // no match.
			{
				a_beg = a_end;
				b_beg = b_end;
			}
			return std::make_tuple(a_beg, b_beg);
		};
		//#pragma omp parallel
		//#pragma omp single //we've got a threadpool, but only one thread is working for now.
		while (this_curr_col_start != t1.end()) // loop over all the columns of this
		{
			auto other_curr_block = t2.begin();
			auto this_col_end = std::lower_bound(this_curr_col_start, t1.end(), next_index(this_curr_col_start), less);
			while (other_curr_block != t2.end()) // loop over all the columns of other
			{
				auto this_curr_block = this_curr_col_start;
				auto other_col_end = std::lower_bound(other_curr_block, t2.end(), next_index(other_curr_block), less);
				std::tie(this_curr_block, other_curr_block) =
				    find_next_match(this_curr_block, this_col_end, other_curr_block, other_col_end, row_less);
				if (this_curr_block != this_col_end and other_curr_block != other_col_end)
				{
					// compute the block index for this combination of columns of the block tensor, could be done only
					// if a match is found.
					std::copy(std::get<0>(*this_curr_block).begin(), std::get<0>(*this_curr_block).end() - dim_l,
					          out_block_index.begin());
					std::copy_backward(std::get<0>(*other_curr_block).begin(),
					                   std::get<0>(*other_curr_block).end() - dim_l, out_block_index.end());

					auto size_range = out_btens.block_sizes(out_block_index);
					out_btens.block(out_block_index) =
					    torch::zeros(std::vector<int64_t>(size_range.begin(), size_range.end()),
					                 std::get<1>(*other_curr_block).options()); // initialize the block.

					// #pragma omp task private(this_curr_block) private(other_curr_block) private(out_block_index)
					// private(this_col_end) private(other_col_end)
					do // parallelizable scope. must make thread local copies of the iterator and block index.
					{
						quantt::tensorgdot_(out_btens.block_at(out_block_index), std::get<1>(*this_curr_block),
						                    std::get<1>(*other_curr_block), dim_l);
						++this_curr_block; // break the match.
						std::tie(this_curr_block, other_curr_block) =
						    find_next_match(this_curr_block, this_col_end, other_curr_block, other_col_end, row_less);
					} while (this_curr_block != this_col_end and other_curr_block != other_col_end);
				}

				other_curr_block = other_col_end;
			}
			this_curr_col_start = this_col_end;
		}
	}
	return out_btens;
}

void btensor::add_tensor_check(const btensor &a, const btensor &b)
{
	if (!(a.c_vals == b.c_vals))
		throw std::invalid_argument("The conserved quantities of the tensors must be a perfect match");
	if (!(a.selection_rule == b.selection_rule))
		throw std::invalid_argument("The selection rules of the tensors must be the same");
	if (!(a.sections_sizes == b.sections_sizes))
		throw std::invalid_argument("The blocks of the tensors must have the same dimensions");
}

// Nasty shenanigans to read the refcount of a torch::Tensor. that thing is private with no accessor. So i make my own
// accessor here. This evil trickery allow us to modify and read private values. We must NEVER modify it.
namespace Evil
{
template <typename Tag, typename Tag::type M>
struct Rob
{
	friend typename Tag::type get(Tag) { return M; }
};
template <class stolen_type, class Victim, size_t tag = 0>
struct Thieving_tag
{
	typedef stolen_type Victim::*type;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-template-friend" // yeah i know the next function ain't a template.
	friend type get(Thieving_tag);
#pragma GCC diagnostic pop
};
using refcount_theft = Thieving_tag<std::atomic<size_t>, c10::intrusive_ptr_target, 0>;
using weakcount_theft = Thieving_tag<std::atomic<size_t>, c10::intrusive_ptr_target, 1>;
template struct Rob<refcount_theft, &c10::intrusive_ptr_target::refcount_>;
template struct Rob<weakcount_theft, &c10::intrusive_ptr_target::weakcount_>;
} // namespace Evil
/**
 * @brief Get the reference count of the tensor
 * 
 * There are some (very few) optimisation that require knowledge of the reference count to apply correctly.
 * This function expose the value of this variable. Reading this variable is an atomic load, do not poll this function uselessly.
 * if you need it multiple times, store the value locally.
 * 
 * @param tens 
 * @return size_t number of reference to this tensor.
 */
size_t get_refcount(const torch::Tensor &tens)
{
	using namespace Evil;
	auto refcounted_ptr_1 = tens.unsafeGetTensorImpl();
	auto refcount_1 = ((*refcounted_ptr_1).*get(refcount_theft())).load(); // this is an atomic load
	// auto weakcount_1 = ((*refcounted_ptr_1).*get(weakcount_theft())).load();
	// auto refcounted_ptr_2 = tens.unsafeGetTensorImpl();
	// auto weakcount_2 = ((*refcounted_ptr_2).*get(weakcount_theft())).load();
	// fmt::print("weakcount {} {}\n", weakcount_1, weakcount_2);
	return refcount_1;
}
/**
 * @brief Get the weak reference count of the tensor.
 * 
 * There are some (very few) optimisation that require knowledge of the reference count to apply correctly.
 * This function expose the value of this variable. Reading this variable is an atomic load, do not poll this function uselessly.
 * if you need it multiple times, store the value locally.
 * 
 * @param tens 
 * @return size_t  number of weak reference.
 */
size_t get_weakcount(const torch::Tensor &tens)
{
	using namespace Evil;
	auto refcounted_ptr_1 = tens.unsafeGetTensorImpl();
	auto refcount_1 = ((*refcounted_ptr_1).*get(weakcount_theft())).load();//this is an atomic load.
	// auto weakcount_1 = ((*refcounted_ptr_1).*get(weakcount_theft())).load();
	// auto refcounted_ptr_2 = tens.unsafeGetTensorImpl();
	// auto weakcount_2 = ((*refcounted_ptr_2).*get(weakcount_theft())).load();
	// fmt::print("weakcount {} {}\n", weakcount_1, weakcount_2);
	return refcount_1;
}

// end of nasty shenanigans

btensor btensor::add(const btensor &other, Scalar alpha) const
{
	add_tensor_check(*this, other); // perform compatibility check before the actual operations.
	auto out = *this;
	for (auto &a : out)
	{
		std::get<1>(a) = std::get<1>(a).clone();
	}
	out.blocks.merge(
	    other.blocks,
	    // collision : do an addition in place
	    [&alpha](torch::Tensor &a, const torch::Tensor &b) {
		    a.add_(b, alpha);
	    }, // no collision, multiply with the constant and make an independent copy
	    [&alpha](torch::Tensor &x) { x = x.mul(alpha); });
	return out;
}
/*!
 * Without the ability to check the refcount of the moved from tensor, this can't really be done efficiently.
 * But i know a trick to access private data.. tbd..
 */
btensor btensor::add(btensor &&other, Scalar alpha) const
{
	add_tensor_check(*this, other); // perform compatibility check before the actual operations.
	auto out = *this;
	out.blocks.merge(
	    other.blocks, // must not use the move merge, no way to get the nocollision case to behave correctly with that
	                  // variant. torch::tensor do shallow copy by default anyway, so it doesn't change the cost of the
	                  // operations. collision
	    [&alpha](torch::Tensor &a, const torch::Tensor &b) { a.add_(b, alpha); },
	    // no collision
	    [&alpha](torch::Tensor &x) {
		    if (get_refcount(x) > 2) // greater than 2 because we've just made a temp copy for this algorithm
		    {
			    x = x.mul(alpha);
		    }
		    else
		    {
			    x.mul_(alpha); // we are the only one with a handle to this tensor now, so we do it in place.
		    }
	    });
	return out;
}
/*!
 * In place addition. Any btensor that is a shallow copy of this (or vice versa) will not be fully updated by the
 * inplace addition. Only the blocks that where present at the moment of the copy will be affected by the inplace
 * addition. Any blocks added afterward or by this additions will not be reflected in related btensor.
 */
btensor &btensor::add_(const btensor &other, Scalar alpha)
{
	// doesn't have quite the same behavior as torch::add_: this one will create partially shared state in btensors that
	// are copy of this if the addition create a new block to the block list. the only work around i can think of is to
	// implement the state sharing logic at the level of the btensor as well. That's quite a bit of work and won't be
	// done right away.
	add_tensor_check(*this, other); // perform compatibility check before the actual operations.
	// std::for_each(b_blocks.begin(), b_blocks.end(), [](auto &x) { std::get<1>(x) = std::get<1>(x).clone(); });
	this->blocks.merge(
	    other.blocks, [&alpha](torch::Tensor &a, const torch::Tensor &b) { a.add_(b, alpha); },
	    [&alpha](torch::Tensor &x) { x = x.mul(alpha); });
	return *this;
}
/**
 * same as add(...)
 */
btensor &btensor::add_(btensor &&other, Scalar alpha)
{
	add_tensor_check(*this, other); // perform compatibility check before the actual operations.
	auto b_blocks = std::move(other.blocks);
	// std::for_each(b_blocks.begin(), b_blocks.end(), [](auto &x) { std::get<1>(x) = std::get<1>(x).clone(); });
	this->blocks.merge(
	    b_blocks, [&alpha](torch::Tensor &a, const torch::Tensor &b) { a.add_(b, alpha); },
	    [&alpha](torch::Tensor &x) {
		    if (get_refcount(x) > 2) // greater than 2 because we've just made a temp copy for this algorithm
		    {
			    x = x.mul(alpha);
		    }
		    else
		    {
			    x.mul_(alpha); // we are the only one with a handle to this tensor now, so we do it in place.
		    }
	    });
	return *this;
}
/**
 * @brief Helper functions for btensor::reshape.
 *
 * The index_group argument is modified such that there is no more implicit information to the list.
 * it must be [0,<original index group>...,rank]
 *
 */
namespace reshape_helpers
{

btensor::index_list reshape_sections_by_dim(torch::IntArrayRef index_groups, size_t new_rank,
                                            btensor::index_list sections_by_dim)
{
	btensor::index_list out_sections_by_dims(new_rank);
	std::transform(index_groups.begin(), index_groups.end() - 1, index_groups.begin() + 1, out_sections_by_dims.begin(),
	               [&sections_by_dim](auto &&a, auto &&b) {
		               return std::reduce(sections_by_dim.begin() + a, sections_by_dim.begin() + b, 1,
		                                  std::multiplies());
	               });
	// those list are not very long, might not be worth it to use a parallel execution.
	// std::transform(std::execution::par, index_groups.begin(), index_groups.end() - 1, index_groups.begin() + 1,
	//                out_sections_by_dims.begin(), [&sections_by_dim](auto &&a, auto &&b) {
	// 	               return std::reduce(std::execution::par, sections_by_dim.begin() + a,
	// 	                                  sections_by_dim.begin() + b,1,std::multiplies());
	//                });
	return out_sections_by_dims;
}
template <class T, class Init>
T reshape_block_prop(torch::IntArrayRef index_groups, const T &block_values, const Init &val, size_t out_size,
                     const btensor::index_list &in_sections_by_dims, const btensor::index_list addresses)
{
	auto out = T(out_size, val);
	auto increment = [](btensor::index_list &index, torch::IntArrayRef max_index, size_t rank) {
		increment_index_right(index, max_index, rank);
	};

	// std::transform(index_groups.begin(), index_groups.end() - 1, index_groups.begin() + 1, out.begin(),
	//    [&](auto &&index_start, auto &&index_end)
	auto out_it = out.begin();
	for (auto index_start = index_groups.begin(); index_start != index_groups.end() - 1; ++index_start)
	{
		auto rank = *(index_start + 1) - *index_start; // number of index being condensed to one.
		auto j = btensor::index_list(rank, 0);
		auto size_j = torch::ArrayRef(in_sections_by_dims.data() + *index_start, rank);
		do
		{
			for (size_t i = 0; i < rank; ++i)
			{
				(*out_it) *= block_values[addresses[i] + j[i]];
			}
			increment(j, size_j, rank)++ out_it;
		} while (any_truth(j));
	});

	return out;
}
btensor::index_list reshape_block_index(torch::IntArrayRef index_groups, const btensor::index_list &block_index,
                                        size_t out_rank, const btensor::index_list& in_sections_by_dim)
{

	btensor::index_list out(out_rank);// would be much simpler if i could zip shifts and block_index.
	std::transform(index_groups.rbegin()-1, index_groups.rend() , index_groups.rbegin(), out.rbegin(),
	               [&](auto &&a, auto &&b) {
					int64_t out = 0;
					int64_t S = 1;
					auto index_start = block_index.begin()+a;
					auto index_finish = block_index.begin()+b;
					auto dim_start = in_sections_by_dim.begin()+a;
					auto dim_end = in_sections_by_dim.begin()+b;
					while(index_start != index_finish)
					{
						--index_finish;
						--dim_end;
						out += S * (*index_finish);
						S *= *dim_end;
					}
					return out;
	               });
	return out;
}
btensor::index_list new_block_shape(torch::IntArrayRef index_groups, btensor::const_block_size_view block_sizes,size_t rank)
{
	btensor::index_list out(rank);
	std::transform(index_groups.begin(), index_groups.end() - 1, index_groups.begin() + 1, out.begin(),
	               [&block_sizes](auto &&a, auto &&b) {
		               return std::reduce(block_sizes.begin() + a, block_sizes.begin() + b, 1,
		                                  std::multiplies());
	               });
	return out;
}
} // namespace reshape_helpers
btensor btensor::reshape(torch::IntArrayRef index_groups) const
{
	using namespace reshape_helpers;
	size_t out_rank = index_groups.size() + 1;
	// make the information about the grouping explicit. The begining of the first group and the end of the last is not
	// explicitly present in the input supplied (it's always 0 and the rank respectivily)
	std::vector<int64_t> m_index_group(out_rank + 2);
	m_index_group[0] = 0;
	m_index_group.back() = rank;
	std::copy(index_groups.begin(), index_groups.end(), m_index_group.begin() + 1);
	// adresses contains the offset for section quantities for each dimensions of the tensor. perhaps i should refactor
	// such that this quantity is a class property.
	// wouldn't be too hard. require modification to the constructor to initialize this, and modification to the view
	// subclasses to make use of this. Would upgrade them from bidirectionnal to random access.
	auto addresses = btensor::index_list(sections_by_dim.size(), 0);
	std::partial_sum(sections_by_dim.begin(), sections_by_dim.end() - 1, addresses.begin() + 1);

	auto out_sections_by_dim = reshape_sections_by_dim(m_index_group, out_rank, sections_by_dim);
	auto out_size = std::reduce(out_sections_by_dim.begin(), out_sections_by_dim.end());
	auto out_sections_sizes =
	    reshape_block_prop(m_index_group, sections_sizes, 1, out_size, sections_by_dim, addresses);
	auto out_c_vals =
	    reshape_block_prop(m_index_group, c_vals, selection_rule.value.neutral(), out_size, sections_by_dim, addresses);
	std::vector<std::pair<btensor::index_list, torch::Tensor>> out_blocks(blocks.size());
	auto out_block_it = out_blocks.begin();
	auto block_it = blocks.begin();
	while (out_block_it != out_blocks.end())
	{
		auto new_block_index = reshape_block_index(m_index_group, std::get<0>(*block_it));
		auto reshaped_block =
		    std::get<1>(*block_it).reshape(new_block_shape(m_index_group, block_sizes(std::get<0>(*block_it))));
		*out_block_it = std::make_pair(std::move(new_block_index), std::move(reshaped_block));
		++out_block_it;
		++block_it;
	}
	return btensor(rank, blocks, out_sections_by_dim, out_sections_sizes, out_c_vals, selection_rule.value);
}

// btensor btensor::sub(const btensor &other, Scalar alpha) const
// {
// 	add_tensor_check(*this, other); //perform compatibility check before the actual operations.
// 	auto out = *this;
// 	for (auto &a : out)
// 	{
// 		std::get<1>(a) = std::get<1>(a).clone();
// 	}
// 	out.blocks.merge(
// 		other.blocks,
// 		// collision : do an addition in place
// 		[&alpha](torch::Tensor &a, const torch::Tensor &b) {
// 			a.sub_(b, alpha);
// 		}, //no collision, multiply with the constant and make an independent copy
// 		[&alpha](torch::Tensor &x) {
// 			x = x.mul(-alpha);
// 		});
// 	return out;
// }
// btensor &btensor::sub_(const btensor &other, Scalar alpha)
// {
// 	// doesn't have quite the same behavior as torch::add_: this one will create partially shared state in btensors that
// are copy of this if the addition create a new block to the block list.
// 	// the only work around i can think of is to implement the state sharing logic at the level of the btensor as well.
// That's quite a bit of work and won't be done right away. 	add_tensor_check(*this, other); //perform compatibility
// check before the actual operations.
// 	// std::for_each(b_blocks.begin(), b_blocks.end(), [](auto &x) { std::get<1>(x) = std::get<1>(x).clone(); });
// 	this->blocks.merge(
// 		other.blocks,
// 		[&alpha](torch::Tensor &a, const torch::Tensor &b) { a.sub_(b, alpha); },
// 		[&alpha](torch::Tensor &x) { x = x.mul(-alpha); });
// 	return *this;
// }
// btensor btensor::sub(btensor &&other, Scalar alpha ) const
// {
// 	add_tensor_check(*this, other); //perform compatibility check before the actual operations.
// 	auto out = *this;
// 	out.blocks.merge(
// 		other.blocks, //must not use the move merge, no way to get the nocollision case to behave correctly with that
// variant. torch::tensor do shallow copy by default anyway, so it doesn't change the cost of the operations.
// 					  //collision
// 		[&alpha](torch::Tensor &a, const torch::Tensor &b) { a.sub_(b, alpha); },
// 		//no collision
// 		[&alpha](torch::Tensor &x) {
// 			if (get_refcount(x) > 2) //greater than 2 because we've just made a temp copy for this algorithm
// 			{
// 				x = x.mul(-alpha);
// 			}
// 			else
// 			{
// 				x.mul_(-alpha); //we are the only one with a handle to this tensor now, so we do it in place.
// 			}
// 		});
// 	return out;
// }
// btensor &btensor::sub_(btensor &&other, Scalar alpha )
// {
// 	add_tensor_check(*this, other); //perform compatibility check before the actual operations.
// 	auto b_blocks = std::move(other.blocks);
// 	// std::for_each(b_blocks.begin(), b_blocks.end(), [](auto &x) { std::get<1>(x) = std::get<1>(x).clone(); });
// 	this->blocks.merge(
// 		b_blocks, [&alpha](torch::Tensor &a, const torch::Tensor &b) { a.sub_(b, alpha); },
// 		[&alpha](torch::Tensor &x) {
// 			if (get_refcount(x) > 2) //greater than 2 because we've just made a temp copy for this algorithm
// 			{
// 				x = x.mul(-alpha);
// 			}
// 			else
// 			{
// 				x.mul_(-alpha); //we are the only one with a handle to this tensor now, so we do it in place.
// 			}
// 		});
// 	return *this;
// }

} // namespace quantt