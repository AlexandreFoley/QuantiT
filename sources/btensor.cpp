/*
 * File: btensor.cpp
 * Project: quantt
 * File Created: Monday, 12th October 2020 12:20:33 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 *
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */
#include "blockTensor/btensor.h"
#include "tensorgdot.h"
#include <ATen/ATen.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/core/TensorBody.h>
#include <algorithm>
#include <cstdint>
#include <exception>
#include <execution>
#include <functional>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <tuple>
#include <vector>

#ifndef NDEBUG
#include <fmt/core.h>
#include <iostream>
#endif

namespace quantt
{

auto promote_types(c10::ScalarType a, c10::ScalarType b) { return at::promote_types(a, b); }
auto promote_types(caffe2::TypeMeta a, c10::ScalarType b) { return at::promote_types(at::typeMetaToScalarType(a), b); }
auto promote_types(c10::ScalarType a, caffe2::TypeMeta b) { return at::promote_types(a, at::typeMetaToScalarType(b)); }
auto promote_types(caffe2::TypeMeta a, caffe2::TypeMeta b)
{
	return at::promote_types(at::typeMetaToScalarType(a), at::typeMetaToScalarType(b));
}

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
std::tuple<any_quantity_vector::iterator, any_quantity_vector::iterator> btensor::section_conserved_qtt_range(
    size_t index)
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
	return std::tuple<size_t, any_quantity_cref>(sections_sizes[ori + block], any_quantity_cref(c_vals[ori + block]));
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
#ifndef NDEBUG
	if (block_index.size() != rank)
		throw std::invalid_argument(fmt::format(
		    "block index invalid for this tensor: it has rank {} instead of expected {}", block_index.size(), rank));
#endif
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
	if (btens.rank == 0)
		return 1;
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
	auto increment = [&sections_by_dims, &rank](btensor::index_list &block_index)
	{ increment_index_left(block_index, sections_by_dims, rank); };
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
                 any_quantity_vector _c_vals, any_quantity _sel_rule, c10::TensorOptions opt)
    : selection_rule(std::move(_sel_rule)), rank(_rank), blocks_list(std::move(_blocks)),
      sections_by_dim(std::move(_sections_by_dims)), sections_sizes(std::move(_block_sizes)),
      c_vals(std::move(_c_vals)), _options(std::move(opt))
{
	std::string check_result = check_tensor(*this);
	if (check_result.size())
		throw std::invalid_argument(fmt::format("Invalid argument to construct a block tensor: \n{}", check_result));
	_options = torch::empty({}, _options).options();
}
std::vector<int64_t> btensor::sizes() const
{
	std::vector<int64_t> out(dim());
	for (size_t i = 0; i < dim(); ++i)
	{
		auto [sections_beg, sections_end] = section_sizes(i);
		out[i] = std::reduce(sections_beg, sections_end, 0);
	}
	return out;
}
btensor::Scalar btensor::item() const
{
	auto d = std::distance(begin(), end());
	if (d == 0)
		return torch::zeros({}, options()).item();
	if (d == 1)
		return begin()->second.item();
	else
		throw std::logic_error("Only simgle block and single element tensors can be converted to scalar");
}
btensor::btensor(btensor::init_list_t dir_block_size_cqtt, any_quantity_cref selection_rule, c10::TensorOptions opt)
    : selection_rule(selection_rule), rank(dir_block_size_cqtt.size()),
      sections_by_dim(block_shapes_from_struct_list(dir_block_size_cqtt, rank)),
      sections_sizes(block_sizes_from_struct_list(dir_block_size_cqtt, sections_by_dim)),
      blocks_list(tensor_list_size_guess(dir_block_size_cqtt, selection_rule, rank, sections_by_dim)),
      c_vals(c_vals_from_struct_list(dir_block_size_cqtt, sections_sizes.size(), selection_rule)),
      _options(std::move(opt))
{
#ifndef NDEBUG
	std::string check_result = check_tensor(*this);
	if (check_result.size())
		throw std::invalid_argument(fmt::format("Invalid argument to construct a block tensor: {}", check_result));
#endif
	_options = torch::empty({}, _options).options(); // freeze the options parameter to whatever the current global
	                                                 // default is. There's probably a better way.
}
btensor::btensor(btensor::init_list_t dir_block_size_cqtt, any_quantity_cref selection_rule, size_t num_blocks,
                 c10::TensorOptions opt)
    : selection_rule(std::move(selection_rule)), rank(dir_block_size_cqtt.size()),
      sections_by_dim(block_shapes_from_struct_list(dir_block_size_cqtt, rank)),
      sections_sizes(block_sizes_from_struct_list(dir_block_size_cqtt, sections_by_dim)), blocks_list(num_blocks),
      c_vals(c_vals_from_struct_list(dir_block_size_cqtt, sections_sizes.size(), selection_rule)),
      _options(std::move(opt))
{
	std::string check_result = check_tensor(*this);
	if (check_result.size())
		throw std::invalid_argument(fmt::format("Invalid argument to construct a block tensor: {}", check_result));
	_options = torch::empty({}, _options).options(); // freeze the options parameter to whatever the current global
	                                                 // default is. There's probably a better way.
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
torch::Tensor &btensor::block_at(const index_list &block_index) { return blocks_list.at(block_index); }

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
	if (block_index.size() != rank)
		std::invalid_argument(fmt::format("block index is size {}, but size {} expected", block_index.size(), rank));
	if (!block_conservation_rule_test(block_index))
	{
		auto message = fmt::format("block index {} not allowed by selection rule. {} != {}", block_index,
		                           fmt::join(block_quantities(block_index), "*"), selection_rule.value);
		throw std::invalid_argument(message);
	}
	return blocks_list[block_index];
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

	for (const auto &a : T.blocks_list)
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
				const auto &qt = T.section_conserved_qtt(i, ind[i]);
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
 * @brief create an empty tensor from selected dimensions of this. Minimum necessary set of feature for tensor network
 * reshape.
 *
 * Basic version of index, it can only discard whole dimensions.
 *
 * @param dims List of dimensions, put -1 to keep the dimension, specify the index to keep otherwise.
 * @return btensor
 */

btensor btensor::shape_from(const std::vector<int64_t> &dims) const
{
	// The selection rule is modified by the product of the non contracted indices.
	// all of the block can receive the same .index() call
	// leave me with the computation of the outgoing shape, the sections_sizes and c_vals are just that of the
	// undropped dimensions.
	// what else?
	// sections_by_dim, plainly shorter: remove the indexed dimensions.
	// rank: the number of -1 in arguement.
	if (dims.size() != rank)
		throw std::invalid_argument("The argument lenght must match the tensor's rank");
	size_t out_rank = 0;
	index_list out_sections_by_dim(this->sections_by_dim.size());
	any_quantity out_sel_rule = this->selection_rule.value;
	auto in_secbd_it = this->sections_by_dim.begin();
	auto out_secbd_it = out_sections_by_dim.begin();
	size_t i = 0; // the dimension we're currently working on
	for (auto &a : dims)
	{
		auto is_slice = a < 0;
		out_rank += is_slice;
		*out_secbd_it = *(in_secbd_it++); // erase the previous value located at this index
		out_secbd_it += is_slice;         // advance only if we're preserving this dimension
		auto [sec_sizes_beg, sec_sizes_end] = section_sizes(i);
		auto el_index = a;
		int block_ind = std::distance(sec_sizes_beg, sec_sizes_end);
		// compute the section associated with the element index
		while (sec_sizes_beg != sec_sizes_end and el_index >= *sec_sizes_beg)
		{
			el_index -= *sec_sizes_beg;
			++sec_sizes_beg;
			// fail silently if the index is too large. require a check earlier on.
		}
		block_ind -= std::distance(sec_sizes_beg, sec_sizes_end);
		out_sel_rule.op(
		    section_conserved_qtt(i, block_ind),
		    not is_slice); // when el_index is negative, block_ind should be 0 and !is_slice false, which is always ok.
		++i;
	}
	out_sections_by_dim.resize(out_rank);
	// at this point we have the outgoing number of section per dimensions and a few other things.
	auto S_Total = std::reduce(out_sections_by_dim.begin(), out_sections_by_dim.end(), 0);
	index_list out_sections_sizes(S_Total);
	any_quantity_vector out_c_vals(S_Total, out_sel_rule.neutral());
	auto out_c_vals_it = out_c_vals.begin();
	auto out_secsize_it = out_sections_sizes.begin();
	auto out_secdim_it = out_sections_by_dim.begin();
	auto in_c_vals_it = c_vals.begin();
	auto in_secdim_it = sections_by_dim.begin();
	auto in_secsizes_it = sections_sizes.begin();
	for (auto &a : dims)
	{
		if (a == -1)
		{
			auto secsize_beg = in_secsizes_it;
			auto c_val_beg = in_c_vals_it;
			in_c_vals_it += *in_secdim_it;
			in_secsizes_it += *in_secdim_it;
			std::copy(secsize_beg, in_secsizes_it, out_secsize_it);
			// fmt::print("{}", *out_c_vals_it);
			std::copy(c_val_beg, in_c_vals_it, out_c_vals_it);
			out_secsize_it += *(in_secdim_it);
			out_c_vals_it += *(in_secdim_it);
			assert(*out_secdim_it == *in_secdim_it);
			++out_secdim_it;
		}
		else
		{
			in_secsizes_it += *(in_secdim_it);
			in_c_vals_it += *(in_secdim_it);
		}
		++in_secdim_it;
	}
	return btensor(out_rank, block_list_t(), std::move(out_sections_by_dim), std::move(out_sections_sizes),
	               std::move(out_c_vals), std::move(out_sel_rule), _options);
}

btensor rank_preserving_shape(const std::vector<int64_t> block_indices, const btensor &was_this)
{
	btensor out({}, was_this.selection_rule->neutral(), was_this.options());
	size_t r = 0;
	auto shape_spec = std::vector<int64_t>(was_this.dim(), 0);
	// build the shape one dimension at a time.
	for (auto b_it = block_indices.begin(); b_it != block_indices.end(); ++b_it, ++r)
	{
		if (*b_it == -1) // slice case
		{
			shape_spec[r] = -1;
			out = quantt::shape_from(out, was_this.shape_from(shape_spec));
			shape_spec[r] = 0;
		}
		else // single element
		{
			out = quantt::shape_from(
			    out, btensor({{{1, was_this.section_conserved_qtt(r, *b_it)}}}, was_this.selection_rule->neutral()));
		}
	}
	out.set_selection_rule_(was_this.selection_rule);
	return out;
}

std::tuple<std::vector<int64_t>, std::vector<torch::indexing::TensorIndex>> to_block_basis(
    const std::vector<int64_t> &dims, const btensor::index_list &sections_by_dim,
    const btensor::index_list &sections_sizes, int64_t rank)
{
	std::vector<int64_t> blocks(rank);
	std::vector<torch::indexing::TensorIndex> element(dims.size(), torch::indexing::Slice());
	auto blockit = blocks.begin();
	auto elementit = element.begin();
	auto section_dimit = sections_by_dim.begin();
	auto sections_sizeit = sections_sizes.begin();
	// prepare the filters for the blocks and tensors.
	// because it's a one or all situation, all the tensor have the same view filter.
	// with more complicated slices, different block would have different start and end to account for the size of the
	// preceding sections.
	for (auto &a : dims)
	{
		if (a != -1)
		{
			auto sectionsize_beg = sections_sizeit;
			sections_sizeit += *(section_dimit);
			auto index = a;
			// TODO: rework this loop, the computation of the value of blockit should be simpler.
			while (sectionsize_beg != sections_sizeit and a >= *sectionsize_beg)
			{
				index -= *sectionsize_beg;
				++sectionsize_beg;
				++(*blockit);
			}
			(*blockit) -= (*blockit) != 0; // the loop does one too many step for a correct blockit;
			*elementit = index;
		}
		else
		{
			*blockit = -1;
			*elementit = torch::indexing::Slice();
			sections_sizeit += *(section_dimit);
		}
		++blockit;
		++elementit;
		++section_dimit;
	}
	return std::make_tuple(std::move(blocks), std::move(element));
}

/**
 * @brief Create a view object on this tensor. Minimum necessary set of feature for tensor network reshape.
 *
 * Basic version of index, it can only discard whole dimensions.
 *
 * @param dims List of dimensions, put -1 to keep the dimension, specify the index to keep otherwise.
 * @return btensor&
 */
btensor btensor::basic_create_view(const std::vector<int64_t> &dims, bool preserve_rank)
{
	auto out_tensor = shape_from(dims);
	// out_tensor has all the correct information regarding the resulting block structure.
	// what's left is to filter the block of this.
	// Given the list of column that must be taken, we have to determine which of the block are in the view, then
	// apply the correct torch::index. (the index value will not be the same as supplied)
	auto [blocks, element] = to_block_basis(dims, sections_by_dim, sections_sizes, dim());
	out_tensor.blocks_list.reserve(blocks.size());
	// function like object to filter out the block to reject, and identify the block index for the output tensor while
	// we're at it
	auto filter = [rank = out_tensor.rank](auto &&index_in, auto &&filter)
	{
		index_list out_index(rank);
		bool keep = true;
		auto out_it = out_index.begin();
		auto filter_it = filter.begin();
		for (auto index_it = index_in.begin(); index_it != index_in.end() and out_it != out_index.end() and keep; ++index_it, ++filter_it)
		{
			*out_it = *index_it;
			auto sliced = *filter_it == -1;
			keep &= sliced or *index_it == *filter_it;
			out_it += sliced;
		}
		return std::make_tuple(keep, out_index);
	};
	// apply the filter
	for (const auto &index_block : this->blocks_list)
	{
		auto [keep, out_index] = filter(std::get<0>(index_block), blocks);
		if (keep)
		{
			out_tensor.blocks_list.insert(out_tensor.blocks_list.end(),
			                              {out_index, std::get<1>(index_block).index(element)});
		}
	}
	if (preserve_rank)
	{
		auto x = rank_preserving_shape(blocks, *this);
		// fmt::print("{}\n\n",to_string(x));
		// print(out_tensor);
		out_tensor = out_tensor.reshape_as<reshape_mode::overwrite_c_vals>(x);
	}
	return out_tensor;
}

std::string to_string(const btensor &x) { return fmt::format("{}", x); }
void print(const btensor &x) { fmt::print("{}\n\n", x); }

btensor &btensor::basic_index_put_(const std::vector<int64_t> &dims, const btensor &value)
{
	btensor reduced_shape = shape_from(dims);
	btensor::add_tensor_check(reduced_shape, value);
	auto [blocks, element] = to_block_basis(dims, sections_by_dim, sections_sizes, dim());

	auto output_index = [](const std::vector<int64_t> &this_index, const btensor::index_list &value_index)
	{
		auto out = this_index;
		auto itv = value_index.begin();
		for (auto ito = out.begin(); ito != out.end(); ++ito)
		{
			bool isNegative1 = (*ito) == -1;
			*ito = (*itv) * isNegative1 + (!isNegative1) * (*ito);
			itv += isNegative1;
		}
		return out;
	};
	for (const auto &index_block : value)
	{
		const auto &index = std::get<0>(index_block);
		const auto &block = std::get<1>(index_block);
		auto out_ind = output_index(blocks, index);
		if (!this->blocks_list.contains(out_ind))
		{
			auto size_view = this->block_sizes(out_ind);
			std::vector<int64_t> size(size_view.begin(),
			                          size_view.end()); // because torch factories don't accept iterator pairs.
			this->blocks_list[out_ind] = torch::zeros(size, options());
		}
		blocks_list[out_ind].index_put_(element, block);
	}
	return *this;
}

btensor btensor::neutral_shape() const
{
	btensor out = *this;
	return out.neutral_shape_();
}

btensor &btensor::neutral_shape_()
{
	if (blocks_list.size() != 0)
		throw std::logic_error("Neutral shape can only function correctly on an empty tensor.");
	selection_rule.value = selection_rule->neutral();
	for (auto &cval : c_vals)
	{
		cval = cval.neutral();
	}
	return *this;
}

template <bool Throws>
bool btensor::check_product_compat(const btensor &in1, const btensor &in2, torch::IntArrayRef dims1,
                                   torch::IntArrayRef dims2) noexcept(!Throws)
{
	bool ok_CR = dims1.size() == dims2.size();
	// fmt::print("{}\n\n {}\n\n",in1,in2);
	for (size_t i = 0; i < dims1.size() and ok_CR; ++i)
	{
		int s1 = in1.section_number(dims1[i]);
		int s2 = in2.section_number(dims2[i]);
		auto q1 = in1.section_conserved_qtt_range(dims1[i]);
		auto q2 = in2.section_conserved_qtt_range(dims2[i]);
		constexpr auto mismatch = "contracted dimensions need to match, but first "
		                          "has {} sections along dim {}  and second has {} sections along dim {}";
		constexpr auto mismatch_cons_rule =
		    "contracted conserved numbers need to sum to zero, but there "
		    "is a violation when contracting dim {} of the left tensor with dim {} of the right tensor";
		ok_CR = s1 == s2;
		if constexpr (Throws)
		{
			TORCH_CHECK(ok_CR, fmt::format(mismatch, s1, dims1[i], s2, dims2[i]));
		}
		else if (!ok_CR)
		{
			return false;
		}
		ok_CR = std::distance(std::get<0>(q1), std::get<1>(q1)) == std::distance(std::get<0>(q2), std::get<1>(q2));
		auto neut = in1.selection_rule->neutral();
		for (auto [it1, it2] = std::make_tuple(std::get<0>(q1), std::get<0>(q2)); it1 != std::get<1>(q1) and ok_CR;
		     ++it1, ++it2)
		{
			ok_CR = ((*it1) * (*it2)) == neut;
		}
		if constexpr (Throws)
		{
			const auto dims1i = dims1[i];
			const auto dims2i = dims2[i];
			TORCH_CHECK(ok_CR, fmt::format(mismatch_cons_rule, dims1i, dims2i));
		}
		// no broadcast dimension like torch::tensordot. i can't think of a way for it to make sense
		// with the quantum number.
	}
	return ok_CR;
}

template bool btensor::check_product_compat<false>(const btensor &in1, const btensor &in2, torch::IntArrayRef dims1,
                                                   torch::IntArrayRef dims2) noexcept;
template bool btensor::check_product_compat<true>(const btensor &in1, const btensor &in2, torch::IntArrayRef dims1,
                                                  torch::IntArrayRef dims2);

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

	btensor::check_product_compat<true>(input1, input2, dims1, dims2);
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
	blocks_list.swap(other.blocks_list);
	swap(selection_rule.value, other.selection_rule.value);
	swap(sections_by_dim, other.sections_by_dim);
	swap(sections_sizes, other.sections_sizes);
}
btensor &btensor::mul_(Scalar other)
{
	apply_to_all_blocks([](auto &&tensor, auto &&other_val) { tensor.mul_(other_val); }, other);
	return *this;
}
btensor btensor::mul(Scalar other) const
{
	// auto out = btensor(*this);
	// out.apply_to_all_blocks([](torch::Tensor &tensor, auto &&other_val) { tensor.mul(other_val); }, other);
	// return out;
	auto out_dtype = promote_types(_options.dtype(), other.type());
	return btensor(*this,
	               new_block_list_apply_to_all_blocks(
	                   [](const torch::Tensor &tensor, auto &&other_val) { return tensor.mul(other_val); }, other),
	               out_dtype);
}
btensor btensor::le(btensor::Scalar other) const
{
	auto out_dtype = torch::kBool;
	return btensor(*this,
	               new_block_list_apply_to_all_blocks(
	                   [](const torch::Tensor &tensor, auto &&other_val) { return tensor.le(other_val); }, other),
	               out_dtype);
}
btensor btensor::ge(btensor::Scalar other) const
{
	auto out_dtype = torch::kBool;
	return btensor(*this,
	               new_block_list_apply_to_all_blocks(
	                   [](const torch::Tensor &tensor, auto &&other_val) { return tensor.ge(other_val); }, other),
	               out_dtype);
}
btensor btensor::eq(btensor::Scalar other) const
{
	auto out_dtype = torch::kBool;
	return btensor(*this,
	               new_block_list_apply_to_all_blocks(
	                   [](const torch::Tensor &tensor, auto &&other_val) { return tensor.eq(other_val); }, other),
	               out_dtype);
}
btensor btensor::less(btensor::Scalar other) const
{
	auto out_dtype = torch::kBool;
	return btensor(*this,
	               new_block_list_apply_to_all_blocks(
	                   [](const torch::Tensor &tensor, auto &&other_val) { return tensor.less(other_val); }, other),
	               out_dtype);
}
btensor btensor::not_equal(btensor::Scalar other) const
{
	auto out_dtype = torch::kBool;
	return btensor(*this,
	               new_block_list_apply_to_all_blocks([](const torch::Tensor &tensor, auto &&other_val)
	                                                  { return tensor.not_equal(other_val); },
	                                                  other),
	               out_dtype);
}
btensor btensor::greater(btensor::Scalar other) const
{
	auto out_dtype = torch::kBool;
	return btensor(*this,
	               new_block_list_apply_to_all_blocks(
	                   [](const torch::Tensor &tensor, auto &&other_val) { return tensor.greater(other_val); }, other),
	               out_dtype);
}
btensor &btensor::div_(btensor::Scalar other)
{
	auto other_ = 1 / torch::full({}, other, options());
	apply_to_all_blocks([](auto &&tensor, auto &&other_val) { tensor.mul_(other_val); }, other_);
	return *this;
}
btensor btensor::div(btensor::Scalar other) const
{
	// auto out = btensor(*this);
	// out.apply_to_all_blocks([](torch::Tensor &tensor, auto &&other_val) { tensor.mul(other_val); }, other);
	// return out;
	auto out_dtype = promote_types(_options.dtype(), other.type());
	return btensor(*this,
	               new_block_list_apply_to_all_blocks(
	                   [](const torch::Tensor &tensor, auto &&other_val) { return tensor.div(other_val); }, other),
	               out_dtype);
}
struct mul_helpers
{

	static auto shape_compute(bool this_is_large, const size_t smaller_rank, const size_t larger_rank,
	                          const btensor &smaller_tensor, const btensor &larger_tensor)
	{
		std::vector<uint8_t> comp_mask(smaller_rank, 0);
		auto new_sections_by_dim = larger_tensor.sections_by_dim;
		{
			auto smaller_sec = smaller_tensor.sections_by_dim.end();
			auto larger_sec = larger_tensor.sections_by_dim.end();
			auto it = comp_mask.end();
			size_t n = 1;
			// detect all the dimensions to broadcast
			while (it != comp_mask.begin())
			{
				--it;
				--smaller_sec;
				--larger_sec;

				*it = (((*smaller_sec == 1) and (smaller_tensor.section_size(smaller_rank - n, 0) == 1)) +
				       2 * ((*larger_sec == 1) and (larger_tensor.section_size(larger_rank - n, 0) == 1)));
				if (*it)
					new_sections_by_dim[larger_rank - n] = std::max(*smaller_sec, *larger_sec);
				constexpr auto message = "The section and tensor size of tensor a (section size {},tensor size {}) "
				                         "must match the section size and tensor size"
				                         "of tensor b (section size {}, tensor size {}) at non-singleton dimension {}";
				if (not(*it) and *smaller_sec != *larger_sec)
				{
					if (this_is_large)
					{
						throw std::invalid_argument(fmt::format(
						    message, *larger_sec, larger_tensor.section_size(larger_rank - n, 0), *smaller_sec,
						    smaller_tensor.section_size(smaller_rank - n, 0), larger_rank - n));
					}
					else
					{
						throw std::invalid_argument(
						    fmt::format(message, *smaller_sec, smaller_tensor.section_size(smaller_rank - n, 0),
						                *larger_sec, larger_tensor.section_size(larger_rank - n, 0), smaller_rank - n));
					}
				}
				++n;
			}
		}
		// Compute the new cval and section sizes and sizes here.
		auto Nsections = std::reduce(new_sections_by_dim.begin(), new_sections_by_dim.end(), 0);
		btensor::index_list new_sections_sizes(Nsections);
		any_quantity_vector new_cvals(Nsections, larger_tensor.selection_rule->neutral());
		{
			auto new_cvals_it = new_cvals.end();
			auto new_sections_sizes_it = new_sections_sizes.end();
			auto smaller_sec = smaller_tensor.sections_by_dim.end();
			auto larger_sec = larger_tensor.sections_by_dim.end();
			auto it = comp_mask.end();
			btensor::index_list::const_iterator section_start, section_end;
			any_quantity_vector::const_iterator cval_start, cval_end, short_cval_start, short_cval_end;

			for (size_t n = 1; n <= smaller_rank; ++n)
			{
				--it;
				--larger_sec;
				--smaller_sec;
				if (*larger_sec > *smaller_sec) // one of them is one
				{
					std::tie(section_start, section_end, cval_start, cval_end) =
					    larger_tensor.section_sizes_cqtts(larger_rank - n);
					std::tie(short_cval_start, short_cval_end) =
					    smaller_tensor.section_conserved_qtt_range(smaller_rank - n);
				}
				else // or they are equal
				{
					std::tie(section_start, section_end, cval_start, cval_end) =
					    smaller_tensor.section_sizes_cqtts(smaller_rank - n);
					std::tie(short_cval_start, short_cval_end) =
					    larger_tensor.section_conserved_qtt_range(larger_rank - n);
				}
				std::copy_backward(section_start, section_end, new_sections_sizes_it);
				auto N = std::distance(section_start, section_end);
				new_sections_sizes_it -= N;
				--short_cval_end;
				while (N > 0)
				{
					--new_cvals_it;
					--cval_end;
					--N;
					*new_cvals_it = (*cval_end) * (*short_cval_end);
					short_cval_end -= (short_cval_end != short_cval_start);
				};
			}
			if (smaller_rank != larger_rank)
			{
				// fmt::print("-------------mul_helper failed-------------\n");
				// fmt::print("larger_tensor {}\n\n",larger_tensor);
				// fmt::print("smaller_tensor {}\n\n",smaller_tensor);
				// fmt::print("new_cvals {}\n\n",new_cvals);
				// fmt::print("new_cvals it delta {}\n\n",std::distance(new_cvals.begin(),new_cvals_it));
				auto rank_diff = larger_rank - smaller_rank - 1;
				auto [size_begin, size_end, cqtt_begin, cqtt_end] = larger_tensor.section_sizes_cqtts(rank_diff);
				// The range to copy is from the begining of the full list to the last element of the dimension
				// <rank_diff>.
				std::copy_backward(larger_tensor.c_vals.begin(), cqtt_end, new_cvals_it);
				std::copy_backward(larger_tensor.sections_sizes.begin(), size_end, new_sections_sizes_it);
			}
		}
		any_quantity out_selr = larger_tensor.selection_rule->get() * smaller_tensor.selection_rule->get();
		return std::make_tuple(comp_mask, new_sections_by_dim, new_sections_sizes, new_cvals, out_selr);
	}
	static auto match_index(size_t smaller_rank, const std::vector<uint8_t> comp_mask, const btensor::index_list &a,
	                        const btensor::index_list &b) -> bool
	{
		bool out = true;
		auto a_it = a.end();
		auto b_it = b.end();
		size_t i = 1;
		auto mask_it = comp_mask.end();
		while (i <= smaller_rank)
		{
			{
				--a_it;
				--b_it;
				--mask_it;
				out &= *mask_it or (*a_it == *b_it);
				++i;
			}
		}
		return out;
	};

	static auto out_index(size_t smaller_rank, const btensor::index_list &large, const btensor::index_list &small)
	    -> btensor::index_list
	{
		// assumes the index are a match.
		auto out = large;
		auto out_it = out.end();
		auto large_it = large.end();
		auto small_it = small.end();
		while (small_it != small.begin())
		{
			--out_it;
			--large_it;
			--small_it;
			*out_it = std::max(*large_it, *small_it);
		};
		return out;
	};
	static auto check_bmm_compatibility(const btensor &a, const btensor &b)
	{
		// first N dimensions must be the same.
		// last 2 must be matrix compatible
		bool good = a.rank == b.rank;
		if (not good)
			throw std::invalid_argument(fmt::format("incompatible ranks {} and {}", a.rank, b.rank));
		if (a.rank < 2)
			throw std::invalid_argument("rank of tensors must be greater or equal to 2");
		{ // check for compatible number of section in batch dimensions
			auto a_sbd = a.sections_by_dim.begin();
			auto b_sbd = b.sections_by_dim.begin();
			for (size_t i = 0; a_sbd != a.sections_by_dim.end() - 2; ++a_sbd, ++b_sbd, ++i)
			{
				if (*a_sbd != *b_sbd)
					throw std::invalid_argument(
					    fmt::format("input tensors have different number of section ({} and {}) in batch dimension {}",
					                *a_sbd, *b_sbd, i));
			}
		}
		{ // check for compatible section sizes in batch dimensions
			auto [a_start, a_end] = a.section_sizes(a.rank - 3);
			auto [b_start, b_end] = b.section_sizes(b.rank - 3);
			a_start = a.sections_sizes.begin();
			b_start = b.sections_sizes.begin();
			for (size_t i = 0; a_start != a_end; ++a_start, ++b_start, ++i)
			{
				if (*a_start != *b_start)
				{
					auto sbd_it = a.sections_by_dim.begin();
					size_t dim = 0;
					while (sbd_it != a.sections_by_dim.end() and *sbd_it > i)
					{
						++dim;
						i -= *sbd_it;
						++sbd_it;
					}
					throw std::invalid_argument(fmt::format(
					    "input tensors have different section sizes ({} and {}) in batch dimension {} section {}",
					    *a_start, *b_start, dim, i));
				}
			}
		}
		{ // check for compatible matrix dimensions. section number, size and cvals
			if (a.sections_by_dim[a.rank - 1] != b.sections_by_dim[a.rank - 2])
				throw std::invalid_argument("input tensors have incompatible section numbers on the matrix dimensions");

			{
				auto [a_start, a_end] = a.section_sizes(a.rank - 1);
				auto [b_start, b_end] = b.section_sizes(a.rank - 2);
				for (size_t i = 0; a_start != a_end; ++a_start, ++b_start, ++i)
				{
					if (*a_start != *b_start)
						throw std::invalid_argument(
						    fmt::format("input tensors matrix section {} have incompatible sizes ({} and {})", i,
						                *a_start, *b_start));
				}
			}
			auto [a_start, a_end] = a.section_conserved_qtt_range(a.rank - 1);
			auto [b_start, b_end] = b.section_conserved_qtt_range(a.rank - 2);

			auto neutral = a_start->neutral();
			for (size_t i = 0; a_start != a_end; ++a_start, ++b_start, ++i)
			{
				if ((*a_start) * (*b_start) != neutral)
					throw std::invalid_argument(fmt::format(
					    "input tensors matrix section {} have incompatible conserved quantities ({} and {})", i,
					    *a_start, *b_start));
			}
		}
	}
};
/**
 * @brief inplace generic implementation of broadcasting operation, for any function that always map 0 to 0. incorrect
 * for addition and substraction
 *
 * @param f out of place variant of the operation, its usually not possible to to the operation in place for every block
 * @param f_ in place variant of the operation, will be used as much as feasible
 * @param other secondary input of the operation.
 * @return btensor&
 */
template <class F, class F_>
btensor &btensor::broadcast_operation_(const btensor &other, F &&f, F_ &&f_)
{
	// broadcasting, multiplies the dimensions from last to first.
	// size one dimensions are treated as "scalar", meaning it is multiplied with every elements of that dimensions. As
	// far as conserved value are concerned, this correspond to a cval shift. for a block tensor a size one dimensions
	// as one section of size one.

	// The larger and smaller identification buisiness can be mostly illiminated: the output MUST be the larger one in
	// rank for the inplace case. This is a bit of a silly decision on torch's part, since it allow reallocation if the
	// tensor has a large enough rank.

	const auto &smaller_rank = std::min(rank, other.rank);
	const auto &larger_rank = std::max(rank, other.rank);
	const btensor &larger_tensor = rank == larger_rank ? *this : other;
	const btensor &smaller_tensor = other.rank == smaller_rank ? other : *this;
	bool this_is_large = rank == larger_rank;
	auto [comp_mask, new_sections_by_dim, new_sections_sizes, new_cvals, out_selr] =
	    mul_helpers::shape_compute(this_is_large, smaller_rank, larger_rank, smaller_tensor, larger_tensor);

	block_list_t out_blocks;
	if (std::any_of(comp_mask.begin(), comp_mask.end(), [](auto &&a) { return bool(a); }))
	{
		out_blocks.reserve(blocks_list.size() * other.blocks_list.size()); // lazy upper bound.
	}
	else
	{
		out_blocks.reserve(std::min(blocks_list.size(), other.blocks_list.size())); // tight upper bound.
	}
	// if there are no broadcast dimensions, the number of output block is smaller or equal to the smaller of the
	// two block_list. if all index are broadcast with the other tensor (e.g. sizes [x,1,y]*[1,z,1]), then the
	// number of blocks will be the product. This is case is basically a tensor product.

	// two indices are a match if all the values are identical, but 0 matches anything if it's the only possible
	// value of that dimension. we only consider a number of value equal to the smaller of the two indices, and
	// start the comparison from the last. We accomplish this by applying a mask to the two indices during the
	// comparison. If mul_ is not used, this breaks the shared state between this and copies of this. The only way
	// to restore exactly the same behavior as torch would be to store the block list and other properties in a shared
	// structure.

	bool can_do_inplace =
	    std::none_of(comp_mask.begin(), comp_mask.end(),
	                 [this_is_large](auto &&x)
	                 {
		                 bool out = (x >> this_is_large) & 1; // whether the current dimensions is 1 in this.
		                 out ^= ((x >> !this_is_large) & 1) &&
		                        out; // exclusive or with the value in the other if it's currently true.
		                 return out;
	                 });

	// when there are no broadcast, there's no need for nested loop, we can increment the two iterator in lockstep
	// When there are broadcast, there's a more complicated, shorter loop. But it probably trigger many more rollback
	// from the branch prediction. This one will almost never match, in which case there's nothing to do. When there's a
	// rollback, it's because there is some work. should be pretty good.
	for (auto other_it = other.blocks_list.begin(); other_it != other.blocks_list.end(); ++other_it)
	{
		for (auto this_it = blocks_list.begin(); this_it != blocks_list.end(); ++this_it)
		{
			auto &this_index = std::get<0>(*this_it);
			auto &other_index = std::get<0>(*other_it);
			auto &this_tensor = std::get<1>(*this_it);
			auto &other_tensor = std::get<1>(*other_it);
			// to get the correct output, we can only do the actual inplace mul_ at the block level only when there's a
			// single output block associated with the input block.
			// at best, we could try to do it inplace for the last call.
			// There's a single output when only this tensor has no size 1 dimensions, or the size 1 dimension are
			// matched with a size 1 in other.
			if (mul_helpers::match_index(smaller_rank, comp_mask, this_index, other_index))
			{
				if (can_do_inplace)
					out_blocks.emplace(out_blocks.end(), mul_helpers::out_index(smaller_rank, this_index, other_index),
					                   f_(this_tensor, other_tensor));
				else
					out_blocks.emplace(out_blocks.end(), mul_helpers::out_index(smaller_rank, this_index, other_index),
					                   f(this_tensor, other_tensor));
			}
		}
	}
	// stuff the new data in this.
	blocks_list = std::move(out_blocks);
	c_vals = std::move(new_cvals);
	sections_sizes = std::move(new_sections_sizes);
	sections_by_dim = std::move(new_sections_by_dim);
	rank = larger_rank;
	selection_rule.value = std::move(out_selr);

	return *this; // btensor(larger_rank, out_blocks, new_sections_by_dim, new_sections_sizes, new_cvals, out_selr);
}
btensor &btensor::mul_(const btensor &other)
{
	return broadcast_operation_(
	    other, [](const torch::Tensor &A, const torch::Tensor &B) { return A.mul(B); },
	    [](torch::Tensor &A, const torch::Tensor &B) { return A.mul_(B); });
}
btensor &btensor::div_(const btensor &other)
{
	return broadcast_operation_(
	    other, [](const torch::Tensor &A, const torch::Tensor &B) { return A.div(B); },
	    [](torch::Tensor &A, const torch::Tensor &B) { return A.div_(B); });
}
btensor btensor::bmm(const btensor &mat) const
{
	// The loop is not proper for parallisation preallocation of output container necessary.
	// The need for syncronisation can be eliminated by reordering the block of mat such that all the output that have
	// the same index are done in loop order.
	mul_helpers::check_bmm_compatibility(*this, mat);
	auto batch_shape_inds = std::vector<int64_t>(dim(), -1);
	batch_shape_inds.back() = 0;
	*(batch_shape_inds.end() - 2) = 0;
	auto this_inds = std::vector<int64_t>(mat.dim(), 0);
	*(this_inds.end() - 2) = -1;
	auto mat_inds = std::vector<int64_t>(mat.dim(), 0);
	mat_inds.back() = -1;
	auto [comp_mask, new_sections_by_dim, new_sections_sizes, new_cvals, out_selr] = mul_helpers::shape_compute(
	    false, rank - 2, rank - 2, this->shape_from(batch_shape_inds), mat.shape_from(batch_shape_inds));
	btensor batch_shape(new_sections_by_dim, new_cvals, new_sections_sizes, selection_rule->neutral());
	auto out = quantt::shape_from(
	    batch_shape, this->shape_from(this_inds),
	    mat.shape_from(mat_inds)); // ambiguity because of the in-class context lifted by the quantt namespace;

	block_list_t::content_t mat_blocks(mat.blocks_list.begin(), mat.blocks_list.end());
	// sort mat in an order ideal for parallelism.
	// stable sort internally work out of place if it can, so it has extra space complexity relative to quicksort
	// quicksort can be used with a more complicated comparator.
	std::sort(mat_blocks.begin(), mat_blocks.end(),
	          [](auto &&a_pair, auto &&b_pair)
	          {
		          const auto &a = std::get<0>(a_pair);
		          const auto &b = std::get<0>(b_pair);
		          // standard loop for lexicographical comparison
		          for (auto [first1, first2] = std::tuple(a.begin(), b.begin());
		               (first1 != a.end() - 2) && (first2 != b.end() - 2); ++first1, ++first2)
		          {
			          if (*first1 < *first2)
				          return true;
			          if (*first2 < *first1)
				          return false;
		          }
		          // swap the order of the last two element of the index.
		          if (*(a.end() - 1) < *(b.end() - 1))
			          return true;
		          if (*(b.end() - 1) < *(a.end() - 1))
			          return false;
		          if (*(a.end() - 2) < *(b.end() - 2))
			          return true;
		          if (*(b.end() - 2) < *(a.end() - 2))
			          return false;
		          return false; // they're equal if we get here. can't happen really.
	          });

	out.selection_rule.value = this->selection_rule.value * mat.selection_rule.value;
	auto batch_equal = [](const index_list &a, const index_list &b) -> bool
	{
		for (auto [it_a, it_b] = std::tuple(a.begin(), b.begin()); it_a != a.end() - 2; ++it_a, ++it_b)
		{
			if (*it_a != *it_b)
				return false;
		}
		return true;
	};
	auto batch_lesser = [](const index_list &a, const index_list &b) -> bool
	{ return std::lexicographical_compare(a.begin(), a.end() - 2, b.begin(), b.end() - 2); };
	auto out_index = [](block_list_t::const_iterator it1,
	                    block_list_t::content_t::const_iterator it2) -> btensor::index_list
	{
		auto &l1 = std::get<0>(*it1);
		auto &l2 = std::get<0>(*it2);
		index_list out(l1.size());
		std::copy(l1.begin(), l1.end() - 1, out.begin());
		out.back() = l2.back();
		return out;
	};
	auto matrix_match = [](const index_list &a, const index_list &b) { return a.back() == *(b.end() - 2); };
	auto out_index_unchanged = [](const index_list &out_index, const index_list &fast_changing)
	{ return out_index.back() == fast_changing.back(); };
	auto this_row_start = begin();
	auto mat_batch_start = mat_blocks.begin();
	auto this_region_start = this_row_start; // begining of a "region" of index with the same batch indices for this.
	auto mat_region_start = mat_batch_start; // idem but for mat.
	out.reserve_space_(btensor_size::max); // we reserve the maximum size for the conservation rule, instead of counting
	                                       // the number of output tensors.
	auto get_ind = [](auto &&x) { return std::get<0>(*x); };
	// create threadpool no later than here.
	while (this_row_start != end())
	{
		// advance the iterator until the batch and matrix indices are a match
		while (this_row_start != end() and mat_batch_start != mat_blocks.end() and
		       (not batch_equal(std::get<0>(*this_row_start), std::get<0>(*mat_batch_start))))
		{
			auto &this_ind = std::get<0>(*this_row_start);
			auto &mat_ind = std::get<0>(*mat_batch_start);
			bool this_less = batch_lesser(this_ind, mat_ind);
			// batch less takes precedence over matrix lesser.
			this_row_start += this_less;
			mat_batch_start += !this_less;
		}
		if (this_row_start == end() or mat_batch_start == mat_blocks.end())
			break;
		// andvance until the output index would change
		auto this_row_end = this_row_start;
		while (this_row_end != end() and batch_equal(get_ind(this_row_start), get_ind(this_row_end)) and
		       get_ind(this_row_end)[rank - 2] == get_ind(this_row_start)[rank - 2])
		{
			++this_row_end;
		}
		auto mat_it = mat_batch_start;
		// do all the products for this_row.
		while (mat_it != mat_blocks.end() and
		       batch_equal(get_ind(mat_it), get_ind(mat_batch_start))) // until batch end in mat
		{
			auto mat_col_end = mat_it;
			while (mat_col_end != mat_blocks.end() and batch_equal(get_ind(mat_batch_start), get_ind(mat_col_end)) and
			       get_ind(mat_it)[rank - 1] == get_ind(mat_col_end)[rank - 1])
			{
				++mat_col_end;
			}
			auto this_it = this_row_start;
			// advance within this column until we find a matrix match if there is one
			while (this_it != this_row_end and mat_it != mat_col_end)
			{
				if (get_ind(this_it)[rank - 1] == get_ind(mat_it)[rank - 2])
					break;
				bool inc = get_ind(this_it)[rank - 1] < get_ind(mat_it)[rank - 2];
				this_it += inc;
				mat_it += !inc;
			}
			if (this_it == this_row_end or mat_it == mat_col_end)
			{
				mat_it = mat_col_end;
				continue; // there isn't a match for this output index, we skip to the next.
			}
			auto ind = out_index(this_it, mat_it);
			// mat_it and this_it are ordered such that all the combination that create the same output index are next
			// to each other. this while loop can be parallelized without syncronization for the output, if we add a
			// preparatory step on the output.
			// The loop is in an order such that the result newest tensor is place at the end of the list.
			// Since enough room has been reserved, no pointer invalidation or reordering happens!
			int64_t reduced;
			std::vector<int64_t> ind_shape;
			{
				auto block_sizes = out.block_sizes(ind);
				ind_shape = std::vector<int64_t>(block_sizes.begin(), block_sizes.end());
				reduced = std::reduce(ind_shape.begin(), ind_shape.end() - 2, 1, std::multiplies());
				std::array<int64_t, 3> ind_shape2 = {reduced, *(ind_shape.end() - 2), *(ind_shape.end() - 1)};
				out.block(ind) = torch::zeros(ind_shape2, this->blocks_list.begin()->second.options());
			}
			// launch a worker here, must have a private copy of the iterators, ind, ind_shape and reduced
			{
				while (this_it != this_row_end and mat_it != mat_col_end)
				{
					if (get_ind(this_it)[rank - 1] == get_ind(mat_it)[rank - 2])
					{
						auto AA = std::get<1>(*this_it);
						auto BA = std::get<1>(*mat_it);
						// work around limitation of baddbmm to rank 3 tensor...
						// reshape the tensor to whatever rank it is to rank 3.
						// TODO: Need to make sure rank 1 tensor are rejected and rank 2 tensor treated correctly.
						auto A_shape = AA.sizes();
						std::array<int64_t, 3> A_shape2 = {reduced, *(A_shape.end() - 2), *(A_shape.end() - 1)};
						auto B_shape = BA.sizes();
						std::array<int64_t, 3> B_shape2 = {reduced, *(B_shape.end() - 2), *(B_shape.end() - 1)};

						auto A = AA.reshape(A_shape2);
						auto B = BA.reshape(B_shape2);
						out.block(ind).baddbmm_(A, B);
						++this_it;
						++mat_it;
					}
					else
					{
						// increment the iterator with the smaller contracted index. we know they're not equal, so one
						// comparison give us the full info.
						bool inc_this = get_ind(this_it)[rank - 1] < get_ind(mat_it)[rank - 2];
						this_it += inc_this;
						mat_it += !inc_this;
					}
				}
				// reshape the output back to its proper shape, workaround baddbmm_ limitation to rank 3.
				out.block_at(ind) = out.block_at(ind).reshape(ind_shape);
			}
			mat_it = mat_col_end;
		}
		// batch end are the start of the next batch.
		this_row_start = this_row_end;
		// mat_batch_start = mat_col_end;
	}
	return out;
}

btensor btensor::sum() const
{
	auto out = btensor({}, selection_rule.value);
	auto list = new_block_list_apply_to_all_blocks([](const torch::Tensor &t) { return t.sum(); });
	torch::Tensor out_val = torch::zeros({}, options());
	for (auto &a : list)
	{
		out_val += std::get<1>(a);
	}
	return btensor(out, btensor::block_list_t({{{}, out_val}}));
}

btensor btensor::sqrt() const
{
	auto out_list = new_block_list_apply_to_all_blocks(torch::sqrt);
	auto opt = options();
	if (out_list.size())
	{
		opt = out_list.begin()->second.options();
	}
	return btensor(*this, std::move(out_list), opt);
}

btensor &btensor::sqrt_()
{
	apply_to_all_blocks(torch::sqrt_);
	if (blocks_list.size())
	{
		_options = blocks_list.begin()->second.options();
	}
	return *this;
}
btensor &btensor::abs_()
{
	apply_to_all_blocks(torch::abs_);
	if (blocks_list.size())
	{
		_options = blocks_list.begin()->second.options();
	}
	return *this;
}

btensor btensor::abs() const
{
	auto out_list = new_block_list_apply_to_all_blocks(torch::abs);
	auto opt = options();
	if (out_list.size())
	{
		opt = out_list.begin()->second.options();
	}
	return btensor(*this, std::move(out_list), opt);
}

btensor btensor::pow(btensor::Scalar exponent) const
{
	auto out_list = new_block_list_apply_to_all_blocks(
	    [](const torch::Tensor &x, btensor::Scalar exponent) { return x.pow(exponent); }, exponent);
	auto opt = options();
	if (out_list.size())
	{
		opt = out_list.begin()->second.options();
	}
	return btensor(*this, std::move(out_list), opt);
}

btensor &btensor::pow_(btensor::Scalar exponent)
{
	auto X = torch::zeros({5, 5});
	// torch::pow_(X,exponent);
	// X.pow_(exponent);
	apply_to_all_blocks([](torch::Tensor &x, btensor::Scalar exponent) { return x.pow_(exponent); }, exponent);
	if (blocks_list.size())
	{
		_options = blocks_list.begin()->second.options();
	}
	return *this;
}
/**
 * @brief out of place generic implementation of broadcasting operation, for any function that always map 0 to 0.
 * incorrect for addition and substraction
 *
 * @param f out of place variant of the operation
 * @param other secondary input of the operation.
 * @return btensor&
 */
template <class F, bool promote>
btensor btensor::broadcast_operation(const btensor &other, F &&f) const
{
	// broadcasting, multiplies the dimensions from last to first.
	// size one dimensions are treated as "scalar", meaning it is multiplied with every elements of that dimensions. As
	// far as conserved value are concerned, this correspond to a cval shift. for a block tensor a size one dimensions
	// as one section of size one.
	const auto &smaller_rank = std::min(rank, other.rank);
	const auto &larger_rank = std::max(rank, other.rank);
	const btensor &larger_tensor = rank == larger_rank ? *this : other;
	const btensor &smaller_tensor = other.rank == smaller_rank ? other : *this;
	const bool this_is_large = rank == larger_rank;
	auto [comp_mask, new_sections_by_dim, new_sections_sizes, new_cvals, out_selr] =
	    mul_helpers::shape_compute(this_is_large, smaller_rank, larger_rank, smaller_tensor, larger_tensor);

	block_list_t out_blocks;
	if (std::any_of(comp_mask.begin(), comp_mask.end(), [](auto &&a) { return bool(a); }))
	{
		out_blocks.reserve(blocks_list.size() * other.blocks_list.size()); // lazy upper bound.
	}
	else
	{
		out_blocks.reserve(std::min(blocks_list.size(), other.blocks_list.size())); // tight upper bound.
	}
	// if there are no broadcast dimensions, the number of output block is smaller or equal to the smaller of the
	// two block_list. if all index are broadcast with the other tensor (e.g. sizes [x,1,y]*[1,z,1]), then the
	// number of blocks will be the product. This is case is basically a tensor product.

	// two indices are a match if all the values are identical, but 0 matches anything if it's the only possible
	// value of that dimension. we only consider a number of value equal to the smaller of the two indices, and
	// start the comparison from the last. We accomplish this by applying a mask to the two indices during the
	// comparison.

	// when there are no broadcast, there's no need for nested loop, we can increment the two iterator in lockstep
	// When there are broadcast, there's a more complicated, shorter loop. But it probably trigger many more rollback
	// from the branch prediction. This one will almost never match, in which case there's nothing to do. When there's a
	// rollback, it's because there is some work. should be pretty good.
	for (auto other_it = other.blocks_list.begin(); other_it != other.blocks_list.end(); ++other_it)
	{
		for (auto this_it = blocks_list.begin(); this_it != blocks_list.end(); ++this_it)
		{
			auto &large_index = this_is_large ? std::get<0>(*this_it) : std::get<0>(*other_it);
			auto &small_index =
			    this_is_large ? std::get<0>(*other_it)
			                  : std::get<0>(*this_it); // out_index care about ordering by size of the input index.
			auto &this_tensor = std::get<1>(*this_it);
			auto &other_tensor = std::get<1>(*other_it);
			if (mul_helpers::match_index(smaller_rank, comp_mask, large_index, small_index))
				out_blocks.emplace(out_blocks.end(), mul_helpers::out_index(smaller_rank, large_index, small_index),
				                   f(this_tensor, other_tensor));
		}
	}
	auto dtype = torch::typeMetaToScalarType(options().dtype());
	if constexpr (promote)
	{
		dtype = promote_types(options().dtype(), other.options().dtype());
	}
	// fmt::print("new_sections_by_dim {}\n", new_sections_by_dim);
	// fmt::print("new_cvals {}\n", new_cvals);
	return btensor(larger_rank, out_blocks, new_sections_by_dim, new_sections_sizes, new_cvals, out_selr,
	               options().merge_in(dtype));
}

btensor btensor::ge(const btensor &other) const
{
	// This should be a broadcasting operation? yes.
	auto out = broadcast_operation(other, [](const torch::Tensor &A, const torch::Tensor &B) { return A.ge(B); });
	out._options = out._options.merge_in(torch::kBool);
	return out;
}
btensor btensor::le(const btensor &other) const
{
	// This should be a broadcasting operation? yes.
	auto out = broadcast_operation(other, [](const torch::Tensor &A, const torch::Tensor &B) { return A.le(B); });
	out._options = out._options.merge_in(torch::kBool);
	return out;
}
btensor btensor::less(const btensor &other) const
{
	// This should be a broadcasting operation? yes.
	auto out = broadcast_operation(other, [](const torch::Tensor &A, const torch::Tensor &B) { return A.less(B); });
	out._options = out._options.merge_in(torch::kBool);
	return out;
}
btensor btensor::greater(const btensor &other) const
{
	// This should be a broadcasting operation? yes.
	auto out = broadcast_operation(other, [](const torch::Tensor &A, const torch::Tensor &B) { return A.greater(B); });
	out._options = out._options.merge_in(torch::kBool);
	return out;
}
btensor btensor::not_equal(const btensor &other) const
{
	// This should be a broadcasting operation? yes.
	auto out =
	    broadcast_operation(other, [](const torch::Tensor &A, const torch::Tensor &B) { return A.not_equal(B); });
	out._options = out._options.merge_in(torch::kBool);
	return out;
}
btensor btensor::eq(const btensor &other) const
{
	// This should be a broadcasting operation? yes.
	auto out = broadcast_operation(other, [](const torch::Tensor &A, const torch::Tensor &B) { return A.eq(B); });
	out._options = out._options.merge_in(torch::kBool);
	return out;
}
btensor btensor::div(const btensor &other) const
{
	return broadcast_operation(other, [](const torch::Tensor &A, const torch::Tensor &B) { return A.div(B); });
}
btensor btensor::pow(const btensor &exponent) const
{
	return broadcast_operation(exponent, [](const torch::Tensor &A, const torch::Tensor &B) { return A.pow(B); });
}
btensor btensor::mul(const btensor &other) const
{
	return broadcast_operation(other, [](const torch::Tensor &A, const torch::Tensor &B) { return A.mul(B); });
}
btensor btensor::permute(torch::IntArrayRef in_permutation) const
{
	// fmt::print("========PERMUTATION========\ninput:\tthis{}\n\n\tperm{}\n", *this, in_permutation);
	block_list_t::content_t out_block_list; // unordered.
	out_block_list.reserve(blocks_list.size());
	index_list out_section_by_dim(rank);
	assert(in_permutation.size() == rank);
	std::vector<int64_t> permutation(rank);
	// turn it into a positive indexed permutation.
	std::transform(in_permutation.begin(), in_permutation.end(), permutation.begin(),
	               [rank = this->rank](auto &&x) { return rank * (x < 0) + x; });
	// permute the tensors and their position in the block matrix
	for (size_t i = 0; i < rank; ++i)
	{
		out_section_by_dim[i] = sections_by_dim[permutation[i]];
	}
	for (auto &block : blocks_list)
	{
		auto permute_index = [rank = this->rank](auto permutation, auto &index)
		{
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
	any_quantity_vector out_c_vals = c_vals.permute(&*permutation.begin(), &*permutation.end(), sections_by_dim);
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
	// fmt::print("=>=>outbound blocks_list<=<=\n{}", fmt::join(out_block_list, "\n\n"));
	return btensor(rank, block_list_t(std::move(out_block_list)), std::move(out_section_by_dim),
	               std::move(out_section_sizes), std::move(out_c_vals), selection_rule.value, _options);
}
btensor &btensor::permute_(torch::IntArrayRef permutation)
{
	auto new_val = permute(permutation);
	swap(new_val); // TODO: temporary. a better implementation will come. Or will it?
	return *this;
}

btensor::block_list_t permute_bl(const btensor::block_list_t &block_list, torch::IntArrayRef block_permutation,
                                 torch::IntArrayRef tensor_permutation)
{//profiler shows that too much time is spent here.
	auto out = block_list;
	if (out.begin() != out.end())
	{
		auto tmp_index = std::get<0>(*out.begin());
		auto ind_l = tmp_index.size();
		for (auto &block : out)
		{
			for (decltype(ind_l) i = 0; i < ind_l; ++i)
			{
				tmp_index[i] = std::get<0>(block)[block_permutation[i]];
			}
			tmp_index.swap(std::get<0>(block));
			std::get<1>(block) = std::get<1>(block).permute(tensor_permutation);
		}
		out.sort();
	}
	return out;
}

btensor::btensor(index_list _sections_by_dim, any_quantity_vector _c_vals, index_list _section_sizes,
                 any_quantity _sel_rule, c10::TensorOptions opt)
    : selection_rule(std::move(_sel_rule)), rank(_sections_by_dim.size()), sections_by_dim(std::move(_sections_by_dim)),
      sections_sizes(std::move(_section_sizes)), blocks_list(), c_vals(std::move(_c_vals)), _options(std::move(opt))
{
	std::string check_result = check_tensor(*this);
	if (check_result.size())
		throw std::invalid_argument(fmt::format("Invalid argument to construct a block tensor: \n{}", check_result));
	_options = torch::empty({}, _options).options(); // freeze the options parameter to whatever the current global
	                                                 // default is. There's probably a better way.
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

btensor btensor::tensor_product_shape(const btensor &other) const
{
	auto [p1, p2, out_section_by_dim] = compute_tdot_shape(*this, other, {}, {});
	auto l = std::reduce(out_section_by_dim.begin(), out_section_by_dim.end(), 0);
	auto out_sel_rule = selection_rule.value + other.selection_rule.value;
	auto [out_cvals, out_section_sizes] = compute_tdot_cval_sectSize(*this, other, p1, p2, 0, l);
	auto dtype = promote_types(options().dtype(), other.options().dtype());
	return btensor(out_section_by_dim, out_cvals, out_section_sizes, std::move(out_sel_rule),
	               options().merge_in(dtype));
}

btensor btensor::transpose(int64_t dim0, int64_t dim1) const
{
	// convert to a permutation? That's the simplest option for me.
	dim0 = rank * (dim0 < 0) + dim0;
	dim1 = rank * (dim1 < 0) + dim1;
	std::vector<int64_t> permutation(rank);
	std::iota(permutation.begin(), permutation.end(), 0);
	permutation[dim0] = dim1;
	permutation[dim1] = dim0;
	return permute(permutation);
}
btensor &btensor::transpose_(int64_t dim0, int64_t dim1)
{
	// convert to a permutation? That's the simplest option for me.
	dim0 = rank * (dim0 < 0) + dim0;
	dim1 = rank * (dim1 < 0) + dim1;
	std::vector<int64_t> permutation(rank);
	std::iota(permutation.begin(), permutation.end(), 0);
	permutation[dim0] = dim1;
	permutation[dim1] = dim0;
	return permute_(permutation);
}

btensor btensor::tensordot(const btensor &other, torch::IntArrayRef dim_self, torch::IntArrayRef dims_other) const
{
	const auto dim_l = dim_self.size();
	// first check that everything matches, and compute the output properties, at the block level.
	btensor out_btens;
	btensor::block_list_t t1, t2;
	std::tie(t1, t2, out_btens) = [&]()
	{
		auto out_scalar_type = promote_types(options().dtype(), other.options().dtype());
		auto [p1, p2, out_section_by_dim] = compute_tdot_shape(*this, other, dim_self, dims_other);
		auto l = std::reduce(out_section_by_dim.begin(), out_section_by_dim.end(), 0);
		auto out_sel_rule = selection_rule.value + other.selection_rule.value;
		auto _t1 = permute_bl(blocks_list, p1, p1);
		auto [out_cvals, out_section_sizes] = compute_tdot_cval_sectSize(*this, other, p1, p2, dim_l, l);
		// swap the permutation for better ordering of the loops with the algorithm.
		std::vector<int64_t> p2_prime(p2.size());
		std::copy_backward(p2.begin(), p2.begin() + dim_l, p2_prime.end());
		std::copy(p2.begin() + dim_l, p2.end(), p2_prime.begin());
		auto _t2 = permute_bl(other.blocks_list, p2_prime, p2);
		btensor out(out_section_by_dim, out_cvals, out_section_sizes, std::move(out_sel_rule),this->options().dtype(out_scalar_type));
		return std::make_tuple(std::move(_t1), std::move(_t2), std::move(out));
	}(); // a lambda that capture everything that we call immediatly. leave us with a somewhat clean namespace in
	     // the scope.
	auto next_index = [dim_l](const auto &iterator)
	{
		// compute the smallest possible block index that correspond to a different output index than the input
		// The last dim_l dimension do not contribute to the output block index.
		auto next = std::get<0>(*iterator);
		if (next.begin() != next.end())
		{
			bool null_dim_l = dim_l == 0;
			const auto rank = std::distance(next.begin(), next.end());
			const auto x = static_cast<int>(rank) - static_cast<int>(dim_l) - null_dim_l;
			constexpr auto max_value = (std::numeric_limits<std::remove_reference_t<decltype(*next.begin())>>::max());
			(*(next.begin() + x)) = max_value*!null_dim_l + null_dim_l*(*(next.begin() + x)+1);
		}
		return next;
	};
	auto find_next_match = [dim_l](auto a_beg, const auto &a_end, auto b_beg, const auto &b_end)
	{
		// if we have a match both of the following boolean are false
		bool a_smaller_b = true;
		bool b_smaller_a = true;
		while ((a_beg != a_end and b_beg != b_end) and (a_smaller_b or b_smaller_a))
		{
			auto &a_l = std::get<0>(*a_beg);
			auto &b_l = std::get<0>(*b_beg);
			// we could use the lexicographical three way comparison instead of those two calls.
			a_smaller_b = std::lexicographical_compare(a_l.end() - dim_l, a_l.end(), b_l.end() - dim_l, b_l.end());
			b_smaller_a = std::lexicographical_compare(b_l.end() - dim_l, b_l.end(), a_l.end() - dim_l, a_l.end());
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
	auto cpt_output_block = [dim_l, r = out_btens.rank](auto this_current_block_iter, auto other_current_block_iter)
	{
		std::vector<int64_t> out_block_index(r);
		std::copy(std::get<0>(*this_current_block_iter).begin(), std::get<0>(*this_current_block_iter).end() - dim_l,
		          out_block_index.begin());
		std::copy_backward(std::get<0>(*other_current_block_iter).begin(),
		                   std::get<0>(*other_current_block_iter).end() - dim_l, out_block_index.end());
		return out_block_index;
	};

	// fmt::print("out_btens section sizes {}\n", out_btens.sections_sizes);
	// fmt::print("input section sizes {}\n", other.sections_sizes);

	{ // launch the contractions.
		auto this_col_start = t1.begin();
		auto less = t1.value_comp();
		//#pragma omp parallel
		//#pragma omp single 
		while (this_col_start != t1.end()) // loop over all the columns of this, parallel tasks must synchronize at the end of this loop.
		{
			auto other_curr_block = t2.begin();
			auto this_col_end = std::lower_bound(this_col_start, t1.end(), next_index(this_col_start), less); //also the next column start if it's not the end.
			while (other_curr_block != t2.end()) // loop over all the columns of other
			{
				auto this_curr_block = this_col_start;
				auto other_col_end = std::lower_bound(other_curr_block, t2.end(), next_index(other_curr_block), less);
				auto out_block_index = cpt_output_block(this_curr_block, other_curr_block);
				// if (!out_btens.block_conservation_rule_test(out_block_index)) 
				// {//skip to next if we don't satisfy the conservation rule: we know that there will be no match.
				// 	other_curr_block = other_col_end;
				// 	continue;
				// }
				std::tie(this_curr_block, other_curr_block) =
				    find_next_match(this_curr_block, this_col_end, other_curr_block, other_col_end);
				if (this_curr_block != this_col_end and
				    other_curr_block != other_col_end) // TODO: getting a false match, in dmrg environement prep.
				{
					// compute the block index for this combination of columns of the input block tensors

					auto size_range = out_btens.block_sizes(out_block_index);
					auto& curr_block_tens = out_btens.blocks_list[out_block_index];//current block output.
					curr_block_tens =
					    torch::zeros(std::vector<int64_t>(size_range.begin(), size_range.end()),
					                 std::get<1>(*other_curr_block).options()); // initialize the block.

					// #pragma omp task private(this_curr_block) private(other_curr_block) private(out_block_index)
					// private(this_col_end) private(other_col_end)
					do // this loop can be executed by a single independant thread. further parallelism is possible at the cost of extra memory.
					{
						quantt::tensorgdot_(curr_block_tens, std::get<1>(*this_curr_block),
						                    std::get<1>(*other_curr_block), dim_l);
						++this_curr_block; // break the match.
						std::tie(this_curr_block, other_curr_block) =
						    find_next_match(this_curr_block, this_col_end, other_curr_block, other_col_end);
					} while (this_curr_block != this_col_end and other_curr_block != other_col_end);
				}

				other_curr_block = other_col_end;
			}
			this_col_start = this_col_end;
		}
	}
	return out_btens;
}
btensor &btensor::squeeze_(int64_t dim)
{
	if (section_number(dim) == 1 and section_size(dim, 0) == 1)
	{ // do the squeeze
		std::vector<int64_t> res(this->dim(), -1);
		res[dim] = 0;
		*this = reshape(res);
	}
	return *this;
}
btensor btensor::squeeze(int64_t dim) const
{
	btensor out = *this;
	return out.squeeze_(dim);
}
btensor &btensor::squeeze_()
{
	std::vector<int64_t> res(this->dim(), -1);
	for (size_t dim = 0; dim < this->dim(); ++dim)
	{
		res[dim] *= !(section_number(dim) == 1 and section_size(dim, 0) == 1);
	}
	*this = reshape_as(shape_from(res));
	return *this;
}
btensor btensor::squeeze() const
{
	btensor out = *this;
	return out.squeeze_();
}

/**
 * @brief compute the complex conjugate of the tensor and inverse the conserved values.
 *
 * @return btensor
 */
btensor btensor::conj() const { return conj_only().inverse_cvals_(); }
btensor btensor::conj_only() const
{
	return btensor(*this, new_block_list_apply_to_all_blocks([](auto &&tens) { return tens.conj(); }));
}

btensor btensor::inverse_cvals() const { return btensor(*this).inverse_cvals_(); }

btensor &btensor::inverse_cvals_()
{
	for (auto &val : c_vals)
	{
		val.inverse_();
	}
	selection_rule.value.inverse_();
	return *this;
}

btensor btensor::cval_shift(any_quantity_cref shift, int64_t dim) const
{
	btensor out = *this;
	return out.cval_shift_(shift, dim);
}

btensor &btensor::cval_shift_(any_quantity_cref shift, int64_t dim)
{
	shift_impl(shift, dim);
	selection_rule.value *= shift.inverse();
	return *this;
}

btensor &btensor::non_conserving_cval_shift_(any_quantity_cref shift, int64_t dim)
{
	if (blocks_list.size() != 0)
		throw std::logic_error("This transformation can only be applied to empty btensors");
	shift_impl(shift, dim);
	return *this;
}

btensor &btensor::shift_selection_rule_(any_quantity_cref shift)
{
	if (blocks_list.size() != 0)
		throw std::logic_error("This transformation can only be applied to empty btensors");
	selection_rule.value *= shift;
	return *this;
}
btensor &btensor::set_selection_rule_(any_quantity_cref value)
{
	if (blocks_list.size() != 0)
		throw std::logic_error("This transformation can only be applied to empty btensors");
	selection_rule.value = value;
	return *this;
}

void btensor::reserve_space_(size_t N) { blocks_list.reserve(N); }

void btensor::reserve_space_(btensor_size) { reserve_space_(btensor_compute_max_size(*this)); }

void btensor::shift_impl(any_quantity_cref shift, int64_t dim)
{
	auto [c_vals_it, c_vals_end] = section_conserved_qtt_range(dim);
	for (; c_vals_it != c_vals_end; ++c_vals_it)
	{
		*c_vals_it *= shift;
	}
}


bool btensor::test_same_shape(const btensor &a, const btensor &b)
{
	if (!(a.c_vals == b.c_vals) or !(a.selection_rule == b.selection_rule) or !(a.sections_sizes == b.sections_sizes))
		return false;
	return true;
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
#if defined(__GNUC__) and not defined(__clang__) // clang doesn't understand this, and defines __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-template-friend" // yeah i know the next function ain't a template.
#endif
	friend type get(Thieving_tag);
#if defined(__GNUC__) and not defined(__clang__)
#pragma GCC diagnostic pop
#endif
};
using refcount_theft = Thieving_tag<std::atomic<size_t>, c10::intrusive_ptr_target, 0>;
using weakcount_theft = Thieving_tag<std::atomic<size_t>, c10::intrusive_ptr_target, 1>;
template struct Rob<refcount_theft, &c10::intrusive_ptr_target::refcount_>;
template struct Rob<weakcount_theft, &c10::intrusive_ptr_target::weakcount_>;
} // namespace Evil
torch_shape shape_from(std::initializer_list<torch_shape> shapes)
{
	size_t rank = 0;
	torch::TensorOptions opt;
	if (std::distance(shapes.begin(), shapes.end()) != 0)
		opt = shapes.begin()->opt;
	for (auto &shape : shapes)
	{
		rank += shape._sizes.size();
	}
	std::vector<int64_t> sizes(rank);
	auto it = sizes.begin();
	for (auto &shape : shapes)
	{
		std::copy(shape._sizes.begin(), shape._sizes.end(), it);
		it += std::distance(shape._sizes.begin(), shape._sizes.end());
	}
	return torch_shape(std::move(sizes), std::move(opt));
}
torch_shape shape_from(const torch_shape &shape, const std::vector<int64_t> inds)
{
	size_t rank = 0;
	for (auto &ind : inds)
	{
		rank += ind == -1;
	}
	std::vector<int64_t> sizes(rank);
	auto it = sizes.begin();
	auto shape_it = shape._sizes.begin();
	if (rank)
	{
		for (auto &ind : inds)
		{
			*it = *shape_it;
			it += ind == -1;
			++shape_it;
		}
	}
	return {sizes, shape.opt};
}
/**
 * @brief Get the reference count of the tensor
 *
 * There are some (very few) optimisation that require knowledge of the reference count to apply correctly.
 * This function expose the value of this variable. Reading this variable is an atomic load, do not poll this function
 * uselessly. if you need it multiple times, store the value locally.
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
 * @brief common code for the implementation of factory function based on the factory function of pytorch
 *
 * Just about every factory function of pytorch have more than one overload.
 * This means that it often (always?) is necessary to wrap the call in a lambda that resolve which overload we want.
 * See ones and ones_like implementation for an exemple.
 *
 * @tparam Factory
 * @param out
 * @param factory
 */
template <class Factory>
void factory_wrap(btensor &out, Factory &&factory)
{
	out.reserve_space_(btensor_size::max);
	auto index = btensor::index_list(out.dim());
	do
	{
		if (out.block_conservation_rule_test(index))
		{
			auto shape_view = out.block_sizes(index);
			out.block(index) = factory(std::vector<int64_t>(shape_view.begin(), shape_view.end()), out.options());
		}
		out.block_increment(index);
	} while (any_truth(index));
}

btensor zeros(btensor::init_list_t shape_spec, any_quantity selection_rule, c10::TensorOptions opt)
{
	auto out = quantt::sparse_zeros(shape_spec, std::move(selection_rule), opt);
	factory_wrap(out, [](torch::IntArrayRef size, c10::TensorOptions options) { return torch::zeros(size, options); });
	return out;
}
btensor zeros_like(const btensor &tens, c10::TensorOptions opt)
{
	auto out = quantt::sparse_zeros_like(tens, opt);
	factory_wrap(out, [](torch::IntArrayRef size, c10::TensorOptions options) { return torch::zeros(size, options); });
	return out;
}

btensor ones(btensor::init_list_t shape_spec, any_quantity selection_rule, c10::TensorOptions opt)
{
	auto out = quantt::sparse_zeros(shape_spec, std::move(selection_rule), opt);
	factory_wrap(out, [](torch::IntArrayRef size, c10::TensorOptions options) { return torch::ones(size, options); });
	return out;
}
btensor ones_like(const btensor &tens, c10::TensorOptions opt)
{
	auto out = quantt::sparse_zeros_like(tens, opt);
	factory_wrap(out, [](torch::IntArrayRef size, c10::TensorOptions options) { return torch::ones(size, options); });
	return out;
}
btensor empty(btensor::init_list_t shape_spec, any_quantity selection_rule, c10::TensorOptions opt)
{
	auto out = quantt::sparse_zeros(shape_spec, selection_rule, opt);
	factory_wrap(out, [](torch::IntArrayRef size, c10::TensorOptions options) { return torch::empty(size, options); });
	return out;
}
btensor empty_like(const btensor &tens, c10::TensorOptions opt)
{
	auto out = quantt::sparse_zeros_like(tens, opt);
	factory_wrap(out, [](torch::IntArrayRef size, c10::TensorOptions options) { return torch::empty(size, options); });
	return out;
}
btensor rand(btensor::init_list_t shape_spec, any_quantity selection_rule, c10::TensorOptions opt)
{
	auto out = quantt::sparse_zeros(shape_spec, selection_rule, opt);
	factory_wrap(out, [](torch::IntArrayRef size, c10::TensorOptions options) { return torch::rand(size, options); });
	return out;
}
btensor rand_like(const btensor &tens, c10::TensorOptions opt)
{
	auto out = quantt::sparse_zeros_like(tens, opt);
	factory_wrap(out, [](torch::IntArrayRef size, c10::TensorOptions options) { return torch::rand(size, options); });
	return out;
}
btensor full(btensor::init_list_t shape_spec, any_quantity selection_rule, btensor::Scalar fill_value,
             c10::TensorOptions opt)
{
	auto out = quantt::sparse_zeros(shape_spec, selection_rule, opt);
	factory_wrap(out, [fill_value](torch::IntArrayRef size, c10::TensorOptions options)
	             { return torch::full(size, fill_value, options); });
	return out;
}
btensor full_like(const btensor &tens, btensor::Scalar fill_value, c10::TensorOptions opt)
{
	auto out = quantt::sparse_zeros_like(tens, opt);
	factory_wrap(out, [fill_value](torch::IntArrayRef size, c10::TensorOptions options)
	             { return torch::full(size, fill_value, options); });
	return out;
}
btensor randint(int64_t low, int64_t high, btensor::init_list_t shape_spec, any_quantity selection_rule,
                c10::TensorOptions opt)
{
	auto out = quantt::sparse_zeros(shape_spec, selection_rule, opt);
	factory_wrap(out, [high, low](torch::IntArrayRef size, c10::TensorOptions options)
	             { return torch::randint(low, high, size, options); });
	return out;
}
btensor randint_like(int64_t low, int64_t high, const btensor &tens, c10::TensorOptions opt)
{
	auto out = quantt::sparse_zeros_like(tens, opt);
	factory_wrap(out, [high, low](torch::IntArrayRef size, c10::TensorOptions options)
	             { return torch::randint(low, high, size, options); });
	return out;
}
btensor randn(btensor::init_list_t shape_spec, any_quantity selection_rule, c10::TensorOptions opt)
{
	auto out = quantt::sparse_zeros(shape_spec, selection_rule, opt);
	factory_wrap(out, [](torch::IntArrayRef size, c10::TensorOptions options) { return torch::randn(size, options); });
	return out;
}
btensor randn_like(const btensor &tens, c10::TensorOptions opt)
{
	auto out = quantt::sparse_zeros_like(tens, opt);
	factory_wrap(out, [](torch::IntArrayRef size, c10::TensorOptions options) { return torch::randn(size, options); });
	return out;
}

std::vector<torch::indexing::TensorIndex> btensor::full_slice(const btensor &tensor, const btensor::index_list &block)
{

	std::vector<torch::indexing::TensorIndex> out(tensor.dim(), torch::indexing::Slice());
	// fmt::print("full slice\n");
	for (size_t i = 0; i < tensor.dim(); ++i)
	{
		auto [ss_start, ss_end] = tensor.section_sizes(i);
		// fmt::print("dim {}, sizes {}\n",i,fmt::join(ss_start,ss_end,std::string(",") ));
		if (block[i] >= std::distance(ss_start, ss_end))
			throw std::invalid_argument(fmt::format("the {}th block index ({}) exceed maximum allowed value ({}).", i,
			                                        block[i], std::distance(ss_start, ss_end)));
		auto ori = std::reduce(ss_start, ss_start + block[i], 0);
		auto slice_end = ori + *(ss_start + block[i]);
		// fmt::print("slice: {} {}\n\n",ori,slice_end);
		out[i] = torch::indexing::Slice(ori, slice_end);
	}
	return out;
}
torch::Tensor btensor::to_dense() const
{

	auto out = torch::zeros(sizes(), options());
	for (const auto &index_block : this->blocks_list)
	{
		auto &index = std::get<0>(index_block);
		auto &tens = std::get<1>(index_block);
		auto Slices = full_slice(*this, index);
		out.index_put_(Slices, tens);
	}
	return out;
}

void from_basic_impl(btensor &out, const torch::Tensor &values)
{
	if (out.dim() != values.dim())
		throw std::invalid_argument("input arguments have incompatible rank!");
	out.reserve_space_(btensor_size::max);
	auto index = btensor::index_list(out.dim());
	// fmt::print("construction of btensor from full basic tensor\n==============\n");
	do
	{
		if (out.block_conservation_rule_test(index))
		{
			auto shape_view = out.block_sizes(index);
			auto S = btensor::full_slice(out, index);
			out.block(index) = values.index(torch::ArrayRef(S));
			// fmt::print("\tindex {}\n\tSlice {}\n\t block {}\n================\n",index,S,out.block(index));
		}
		out.block_increment(index);
	} while (any_truth(index));
}
btensor from_basic_tensor(btensor::init_list_t shape_spec, any_quantity selection_rul, const torch::Tensor &values,
                          c10::TensorOptions opt)
{
	auto shape = quantt::sparse_zeros(shape_spec, selection_rul, opt);
	from_basic_impl(shape, values);
	return shape;
}
btensor from_basic_tensor_like(const btensor &shape, const torch::Tensor &values, c10::TensorOptions opt)
{
	auto out = quantt::sparse_zeros_like(shape, opt);
	from_basic_impl(out, values);
	return out;
}
bool allclose(const btensor &a, const btensor &b, double rtol, double atol, bool equal_nan)
{
	if (!btensor::test_same_shape(a, b))
		return false;
	auto a_it = a.begin();
	auto b_it = b.begin();
	bool is_allclose = true;
	while (a_it != a.end() and b_it != b.end() and is_allclose)
	{
		const auto &a_index = std::get<0>(*a_it);
		const auto &b_index = std::get<0>(*b_it);
		const auto &a_tens = std::get<1>(*a_it);
		const auto &b_tens = std::get<1>(*b_it);
		if (a_index < b_index)
		{
			is_allclose &= torch::allclose(a_tens, torch::zeros_like(a_tens), rtol, atol, equal_nan);
			++a_it;
		}
		else if (a_index > b_index)
		{
			is_allclose &= torch::allclose(b_tens, torch::zeros_like(b_tens), rtol, atol, equal_nan);
			++b_it;
		}
		else // equal index
		{
			is_allclose &= torch::allclose(a_tens, b_tens, rtol, atol, equal_nan);
			++a_it;
			++b_it;
		}
	}
	// maybe one of the two has blocks of zeros packed at the end.
	while (is_allclose and a_it != a.end())
	{
		const auto &a_tens = std::get<1>(*a_it);
		const auto &a_index = std::get<0>(*a_it);
		is_allclose &= torch::allclose(a_tens, torch::zeros_like(a_tens), rtol, atol, equal_nan);
		++a_it;
	}
	while (is_allclose and b_it != b.end())
	{
		const auto &b_index = std::get<0>(*b_it);
		const auto &b_tens = std::get<1>(*b_it);
		is_allclose &= torch::allclose(b_tens, torch::zeros_like(b_tens), rtol, atol, equal_nan);
		++b_it;
	}
	return is_allclose;
}
btensor squeeze(btensor tens, int64_t dim) { return tens.squeeze_(dim); }
any_quantity find_selection_rule(const torch::Tensor &tens, const btensor &shape, btensor::Scalar cutoff)
{
	if (tens.dim() != 2)
		throw std::invalid_argument("the input tensor must be rank 2");
	if (shape.dim() != 2)
		throw std::invalid_argument("the shape specifying btensor must be rank 2");
	auto state = torch::zeros(tens.sizes()[tens.dim() - 1], tens.options());
	auto quantitiesi = [&shape](size_t element_pos, size_t dim)
	    -> const vquantity & { // return type must be specified, because return by value isn't an option here.
		size_t block = 0;
		auto [sect_size_beg, sect_size_end, sect_cqtt_beg, sec_cqtt_end] = shape.section_sizes_cqtts(dim);
		while (sect_size_beg != sect_size_end and dim > *sect_size_beg)
		{
			dim -= *(sect_size_beg);
			++sect_size_beg;
			++sect_cqtt_beg;
			++block;
		}
		// dereferencing the the vquantities iterator might fail... it has many time before for unknown reasons.
		return *sect_cqtt_beg;
		// in that case use:
		// return shape.section_conserved_qtt(dim,block));
	};
	auto quantities1 = [&quantitiesi, &shape](size_t element_pos) -> const vquantity & {
		return quantitiesi(element_pos, 0);
	};
	auto quantities2 = [&quantitiesi, &shape](size_t element_pos) -> const vquantity & {
		return quantitiesi(element_pos, 1);
	};
	any_quantity out_sel_rule;
	bool first_hit = true;
	for (auto i = 0; i < state.sizes()[0]; ++i)
	{
		state.index_put_({i}, 1);

		auto out_state = abs(tens.matmul(state)) > cutoff;
		// out_state = out_state > cutoff; //so we can deal with floating points values
		for (auto j = 0; j < out_state.sizes()[0]; ++j)
		{
			if (out_state.index({j}).item().to<bool>())
			{
				if (first_hit)
				{ // first non-negligible element in the output, sets the selection rule, all other iteration are
				  // testing for correctness.
					first_hit = false;
					out_sel_rule = quantities1(j) * quantities2(i);
				}
				else if (out_sel_rule != quantities1(j) * quantities2(i))
					throw std::logic_error("input tensor doesn't have a well defined selection rule");
			}
		}
		state.index_put_({i}, 0);
	}
	return out_sel_rule;
}
/**
 * @brief Get the weak reference count of the tensor.
 *
 * There are some (very few) optimisation that require knowledge of the reference count to apply correctly.
 * This function expose the value of this variable. Reading this variable is an atomic load, do not poll this function
 * uselessly. if you need it multiple times, store the value locally.
 *
 * @param tens
 * @return size_t  number of weak reference.
 */
size_t get_weakcount(const torch::Tensor &tens)
{
	using namespace Evil;
	auto refcounted_ptr_1 = tens.unsafeGetTensorImpl();
	auto refcount_1 = ((*refcounted_ptr_1).*get(weakcount_theft())).load(); // this is an atomic load.

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
	out.blocks_list.merge(
	    other.blocks_list,
	    // collision : do an addition in place
	    [&alpha](torch::Tensor &a, const torch::Tensor &b)
	    { a.add_(b, alpha); }, // no collision, multiply with the constant and make an independent copy
	    [&alpha](torch::Tensor &x) { x = x.mul(alpha); });
	out._options = std::get<1>(*out.begin()).options();
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
	out.blocks_list.merge(
	    other.blocks_list, // must not use the move merge, no way to get the nocollision case to behave correctly
	                       // with that variant. torch::tensor do shallow copy by default anyway, so it doesn't
	                       // change the cost of the operations. collision
	    [&alpha](torch::Tensor &a, const torch::Tensor &b) { a.add_(b, alpha); },
	    // no collision
	    [&alpha](torch::Tensor &x)
	    {
		    if (get_refcount(x) > 2) // greater than 2 because we've just made a temp copy for this algorithm
		    {
			    x = x.mul(alpha);
		    }
		    else
		    {
			    x.mul_(alpha); // we are the only one with a handle to this tensor now, so we do it in place.
		    }
	    });
	out._options = std::get<1>(*out.begin()).options();
	return out;
}
/*!
 * In place addition. Any btensor that is a shallow copy of this (or vice versa) will not be fully updated by the
 * inplace addition. Only the blocks that where present at the moment of the copy will be affected by the inplace
 * addition. Any blocks added afterward or by this additions will not be reflected in related btensor.
 */
btensor &btensor::add_(const btensor &other, Scalar alpha)
{
	// doesn't have quite the same behavior as torch::add_: this one will create partially shared state in btensors
	// that are copy of this if the addition create a new block to the block list. the only work around i can think
	// of is to implement the state sharing logic at the level of the btensor as well. That's quite a bit of work
	// and won't be done right away.
	add_tensor_check(*this, other); // perform compatibility check before the actual operations.
	// std::for_each(b_blocks.begin(), b_blocks.end(), [](auto &x) { std::get<1>(x) = std::get<1>(x).clone(); });
	this->blocks_list.merge(
	    other.blocks_list, [&alpha](torch::Tensor &a, const torch::Tensor &b) { a.add_(b, alpha); },
	    [&alpha](torch::Tensor &x) { x = x.mul(alpha); });
	return *this;
}
/**
 * same as add(...)
 */
btensor &btensor::add_(btensor &&other, Scalar alpha)
{
	add_tensor_check(*this, other); // perform compatibility check before the actual operations.
	auto b_blocks = std::move(other.blocks_list);
	// std::for_each(b_blocks.begin(), b_blocks.end(), [](auto &x) { std::get<1>(x) = std::get<1>(x).clone(); });
	this->blocks_list.merge(
	    b_blocks, [&alpha](torch::Tensor &a, const torch::Tensor &b) { a.add_(b, alpha); },
	    [&alpha](torch::Tensor &x)
	    {
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
	std::transform(
	    index_groups.begin(), index_groups.end() - 1, index_groups.begin() + 1, out_sections_by_dims.begin(),
	    [&sections_by_dim](auto &&a, auto &&b)
	    { return std::reduce(sections_by_dim.begin() + a, sections_by_dim.begin() + b, 1, std::multiplies()); });
	// those list are not very long, might not be worth it to use a parallel execution.
	// std::transform(std::execution::par, index_groups.begin(), index_groups.end() - 1, index_groups.begin() +
	// 1,
	//                out_sections_by_dims.begin(), [&sections_by_dim](auto &&a, auto &&b) {
	// 	               return std::reduce(std::execution::par, sections_by_dim.begin() + a,
	// 	                                  sections_by_dim.begin() + b,1,std::multiplies());
	//                });
	return out_sections_by_dims;
}
/**
 * @brief multiplies the block property of the grouped indices.
 *
 * @param index_groups
 * @param block_values
 * @param val
 * @param out_size
 * @param in_sections_by_dims
 * @param addresses
 * @return T Block Property for the reduced rank tensor.
 */
template <class T, class Init>
T reshape_block_prop(torch::IntArrayRef index_groups, const T &block_values, const Init &val, size_t out_size,
                     const btensor::index_list &in_sections_by_dims, const btensor::index_list addresses)
{
	auto out = T(out_size, val);
	auto increment = [](btensor::index_list &index, torch::IntArrayRef max_index, size_t rank)
	{ increment_index_right(index, max_index, rank); };

	auto out_it = out.begin();
	for (auto index_start = index_groups.begin(); index_start != index_groups.end() - 1; ++index_start)
	{
		auto rank = *(index_start + 1) - *index_start; // number of index being condensed to one.
		auto j = btensor::index_list(rank, 0);
		// fmt::print("index bundle [{}, {}[\n", *index_start, *(index_start+1));
		auto size_j = torch::ArrayRef(in_sections_by_dims.data() + *index_start, rank);
		// fmt::print("sizes {}\n", size_j);
		do
		{
			// fmt::print("	j {}\n",j);
			for (size_t i = 0; i < rank; ++i)
			{
				// fmt::print("		value {}\n", block_values[addresses[i+*index_start] + j[i]]);
				(*out_it) *= block_values[addresses[i + *index_start] + j[i]];
				// fmt::print("		accumulator {}\n", *out_it);
			}
			increment(j, size_j, rank);
			++out_it;
		} while (any_truth(j));
	}

	return out;
}
/**
 * @brief
 * Note: this whole buisiness of computing the new block index could be avoided if we used the vectorized index
 * always for ordering. Drawback: permute then becomes much harder.
 * @param index_groups
 * @param block_index
 * @param out_rank
 * @param in_sections_by_dim
 * @return btensor::index_list
 */
btensor::index_list reshape_block_index(torch::IntArrayRef index_groups, const btensor::index_list &block_index,
                                        size_t out_rank, const btensor::index_list &in_sections_by_dim)
{

	btensor::index_list out(out_rank);
	auto ig_it = index_groups.end() - 1;
	auto out_it = out.end();
	while (ig_it != index_groups.begin())
	{
		--out_it;
		int64_t S = 1;
		auto index_finish = block_index.begin() + *ig_it;
		auto dim_end = in_sections_by_dim.begin() + *ig_it;
		--ig_it;
		auto index_start = block_index.begin() + *ig_it;
		auto dim_start = in_sections_by_dim.begin() + *ig_it;
		while (index_start != index_finish)
		{
			--index_finish;
			--dim_end;
			*out_it += S * (*index_finish);
			S *= *dim_end;
		}
	}
	return out;
}
btensor::index_list new_block_shape(torch::IntArrayRef index_groups, btensor::const_block_size_view block_sizes,
                                    size_t rank)
{
	btensor::index_list out(rank, 1);
	auto old_rank = index_groups.back();
	auto size_it = block_sizes.begin();
	auto out_it = out.begin();
	auto index_it = index_groups.begin() + 1;
	for (auto i = index_groups.front(); i < old_rank; ++i)
	{
		if (i >= (*index_it))
		{
			++out_it;
			++index_it;
		}
		*out_it *= *size_it;
		++size_it;
	}
	// implementation if i updgrade the block_size_view to random access.
	// std::transform(index_groups.begin(), index_groups.end() - 1, index_groups.begin() + 1, out.begin(),
	//                [&block_sizes](auto &&a, auto &&b) {
	// 	               return std::reduce(block_sizes.begin() + a, block_sizes.begin() + b, 1,
	// std::multiplies());
	//                });
	return out;
}
/**
 * @brief compute the new block index of a block
 * Note: this whole buisiness of computing the new block index could be avoided if we used the vectorized index
 * always for ordering. Drawback: permute then becomes much harder. consequence: functions that access a block
 * would first compute the vectorized index. time cost perhaps O(rank), we have to factor in the cost of the
 * search with the index_list instead of a single integer to be sure... The performance consideration are likely
 * to be irrelevent.
 * @param in_block_index old block index
 * @param out_rank output rank (lenght of out_block_index)
 * @param out_sections_by_dim number of section along each dimension of the output
 * @param in_rank current rank
 * @param sections_by_dim current number of section along each dimension
 * @return btensor::index_list out_block_index
 */
btensor::index_list reshape_block_index(btensor::index_list in_block_index, size_t out_rank,
                                        btensor::index_list out_sections_by_dim, btensor::index_list sections_by_dim)
{
	// i can't think of a more clever algorithm than this right now.
	auto flatten = [](auto &&index, auto &&sizes)
	{
		size_t out = 0;
		size_t s = 1;

		auto size_it = sizes.end();
		auto index_it = index.end();
		while (index_it != index.begin())
		{
			--size_it;
			--index_it;
			out += s * (*index_it);
			s *= *size_it;
		}
		return out;
	};
	auto unflatten = [out_rank](size_t flat, auto &&sizes)
	{
		auto out = btensor::index_list(out_rank);
		auto out_it = out.end();
		auto size_it = sizes.end();
		while (out_it != out.begin())
		{
			--out_it;
			--size_it;
			*out_it = flat % (*size_it);
			flat /= *size_it;
		}
		return out;
	};
	size_t flat_index = flatten(in_block_index, sections_by_dim);
	return unflatten(flat_index, out_sections_by_dim);
}
bool compatible_sections_by_dim(const btensor::index_list &lhs_sections_by_dim,
                                const btensor::index_list &rhs_sections_by_dim)
{
	auto lhs = std::reduce(lhs_sections_by_dim.begin(), lhs_sections_by_dim.end(), 1, std::multiplies());
	auto rhs = std::reduce(rhs_sections_by_dim.begin(), rhs_sections_by_dim.end(), 1, std::multiplies());
	return lhs == rhs;
}
// each of those possible block have the same associated flux. (product of all the conserved quantity associated
// with the block)
bool compatible_c_vals(const btensor &lhs, const btensor &rhs)
{
	bool out = true;
	auto lhs_index = btensor::index_list(lhs.dim(), 0);
	auto rhs_index = btensor::index_list(rhs.dim(), 0);
	auto increment = [](btensor::index_list &index, torch::IntArrayRef max_index, size_t rank)
	{ increment_index_right(index, max_index, rank); };
	do
	{
		auto lhs_c_vals = lhs.block_quantities(lhs_index);
		auto rhs_c_vals = rhs.block_quantities(rhs_index);
		auto lhs_f =
		    std::accumulate(lhs_c_vals.begin(), lhs_c_vals.end(), lhs.selection_rule->neutral(), std::multiplies());
		auto rhs_f =
		    std::accumulate(rhs_c_vals.begin(), rhs_c_vals.end(), rhs.selection_rule->neutral(), std::multiplies());
		out &= lhs_f == rhs_f;
		increment(lhs_index, lhs.section_numbers(), lhs.dim());
		increment(rhs_index, rhs.section_numbers(), rhs.dim());
	} while (any_truth(lhs_index) and out);
	return out;
}
// each of the possible block have the same total number of elements.
bool compatible_block_size(const btensor &lhs, const btensor &rhs)
{
	bool out = true;
	auto lhs_index = btensor::index_list(lhs.dim(), 0);
	auto rhs_index = btensor::index_list(rhs.dim(), 0);
	auto increment = [](btensor::index_list &index, torch::IntArrayRef max_index, size_t rank)
	{ increment_index_right(index, max_index, rank); };
	do
	{
		auto lhs_vals = lhs.block_sizes(lhs_index);
		auto rhs_vals = rhs.block_sizes(rhs_index);
		auto lhs_f = std::accumulate(lhs_vals.begin(), lhs_vals.end(), 1, std::multiplies());
		auto rhs_f = std::accumulate(rhs_vals.begin(), rhs_vals.end(), 1, std::multiplies());
		out &= lhs_f == rhs_f;
		increment(lhs_index, lhs.section_numbers(), lhs.dim());
		increment(rhs_index, rhs.section_numbers(), rhs.dim());
	} while (any_truth(lhs_index) and out);
	return out;
}
} // namespace reshape_helpers
btensor btensor::reshape(torch::IntArrayRef index_groups) const
{
	using namespace reshape_helpers;
	size_t out_rank = index_groups.size() + 1;
	// make the information about the grouping explicit. The begining of the first group and the end of the last is
	// not explicitly present in the input supplied (it's always 0 and the rank respectivily)
	std::vector<int64_t> m_index_group(out_rank + 1);
	m_index_group[0] = 0;
	m_index_group.back() = rank;
	std::copy(index_groups.begin(), index_groups.end(), m_index_group.begin() + 1);
	// adresses contains the offset for section quantities for each dimensions of the tensor. perhaps i should
	// refactor such that this quantity is a class property. wouldn't be too hard. require modification to the
	// constructor to initialize this, and modification to the view subclasses to make use of this. Would upgrade
	// them from bidirectionnal to random access.
	auto addresses = btensor::index_list(sections_by_dim.size(), 0);
	std::partial_sum(sections_by_dim.begin(), sections_by_dim.end() - 1, addresses.begin() + 1);

	auto out_sections_by_dim = reshape_sections_by_dim(m_index_group, out_rank, sections_by_dim);
	auto out_size = std::reduce(out_sections_by_dim.begin(), out_sections_by_dim.end());
	auto out_sections_sizes =
	    reshape_block_prop(m_index_group, sections_sizes, 1, out_size, sections_by_dim, addresses);
	auto out_c_vals =
	    reshape_block_prop(m_index_group, c_vals, selection_rule.value.neutral(), out_size, sections_by_dim, addresses);
	// fmt::print("{}",out_c_vals);
	std::vector<std::pair<btensor::index_list, torch::Tensor>> out_blocks(blocks_list.size());
	auto out_block_it = out_blocks.begin();
	auto block_it = blocks_list.begin();
	while (out_block_it != out_blocks.end())
	{
		auto new_block_index = reshape_block_index(m_index_group, std::get<0>(*block_it), out_rank, sections_by_dim);
		auto reshaped_block = std::get<1>(*block_it).reshape(
		    new_block_shape(m_index_group, block_sizes(std::get<0>(*block_it)), out_rank));
		*out_block_it = std::make_pair(std::move(new_block_index), std::move(reshaped_block));
		++out_block_it;
		++block_it;
	}
	return btensor(out_rank, out_blocks, out_sections_by_dim, out_sections_sizes, out_c_vals, selection_rule.value,
	               _options);
}

template <reshape_mode Mode>
btensor btensor::reshape_as(const btensor &other) const
{
	static_assert(Mode == reshape_mode::dims_only or Mode == reshape_mode::overwrite_c_vals, "invalid reshape mode");
	using namespace reshape_helpers;
	// Check compatibility
	// total possible number of block must be the same. (product of the number of section along each dimension)
	if (not compatible_sections_by_dim(other.sections_by_dim, sections_by_dim))
		throw std::invalid_argument(
		    fmt::format("incompatible sections layouts {} and {}", sections_by_dim, other.sections_by_dim));
	// each of those possible block have the same associated flux. (product of all the conserved quantity associated
	// with the block)
	any_quantity sel_rul(selection_rule->neutral());
	if constexpr (Mode == reshape_mode::dims_only)
	{
		if (not compatible_c_vals(*this, other))
			throw std::invalid_argument(
			    "incompatible conserved quantities"); // hard to offer a more detailed human readable diagnostic,
			                                          // without having the throw in the test loop.
		sel_rul = this->selection_rule.value;
	}
	else
	{
		sel_rul = other.selection_rule.value;
	}

	// each of the possible block have the same total number of elements.
	if (not compatible_block_size(*this, other))
		throw std::invalid_argument("incompatible block dimensions"); // idem to c_vals
	// ok

	std::vector<std::pair<btensor::index_list, torch::Tensor>> out_blocks(blocks_list.size());
	auto out_block_it = out_blocks.begin();
	auto block_it = blocks_list.begin();
	while (out_block_it != out_blocks.end())
	{
		auto new_block_index =
		    reshape_block_index(std::get<0>(*block_it), other.rank, other.sections_by_dim, sections_by_dim);
		if constexpr (Mode == reshape_mode::overwrite_c_vals)
		{
			if (not other.block_conservation_rule_test(new_block_index))
			{
				throw std::invalid_argument(fmt::format(
				    "block at {} in orginal btensor not allowed by the new selection rule", std::get<0>(*block_it)));
			}
		};

		auto new_shape_view = other.block_sizes(new_block_index);
		std::vector<int64_t> new_shape(new_shape_view.begin(), new_shape_view.end());
		auto reshaped_block = std::get<1>(*block_it).reshape(new_shape);
		*out_block_it = std::make_pair(std::move(new_block_index), std::move(reshaped_block));
		++out_block_it;
		++block_it;
	}

	return btensor(other.rank, std::move(out_blocks), other.sections_by_dim, other.sections_sizes, other.c_vals,
	               std::move(sel_rul), options());
}
// with those explicit instantiation, having the template definition visible should be unnecessary.
template btensor btensor::reshape_as<reshape_mode::overwrite_c_vals>(
    const btensor &other) const;                                                           // explicit instantiation
template btensor btensor::reshape_as<reshape_mode::dims_only>(const btensor &other) const; // explicit instantiation

const btensor::block_list_t &btensor::blocks() const { return blocks_list; }
/**
 * @brief make a full torch::tensor from a btensor
 *
 * @param tensor
 * @return torch::Tensor
 */
// torch::Tensor to_dense(btensor &tensor) { return tensor.to_dense(); }

/**
 * @brief split a full tensor into a block tensor with the specified shape
 *
 * regions of the input tensor that are not allowed by the conservation rules are ignored.
 *
 * @param tensor
 * @param shape
 * @return btensor
 */
// btensor split(torch::Tensor &tensor, btensor &shape) // missing argument: tensor to block specification.
// {
// 	// check that the sizes are compatible.

// 	// create a block_list with all allowed element assigned.

// 	// assign a view on the tensor for each block

// 	// return!
// }
/**
 * @brief Create a btensor from the torch tensor supplied and the shape supplied in arguement
 *
 * The position specification is going to be more complicated than merely a shape.
 *
 *
 * @param tensor
 * @param shape
 * @return btensor
 */

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
// 	// doesn't have quite the same behavior as torch::add_: this one will create partially shared state in btensors
// that are copy of this if the addition create a new block to the block list.
// 	// the only work around i can think of is to implement the state sharing logic at the level of the btensor as
// well. That's quite a bit of work and won't be done right away. 	add_tensor_check(*this, other); //perform
// compatibility check before the actual operations.
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
// 		other.blocks, //must not use the move merge, no way to get the nocollision case to behave correctly with
// that variant. torch::tensor do shallow copy by default anyway, so it doesn't change the cost of the operations.
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