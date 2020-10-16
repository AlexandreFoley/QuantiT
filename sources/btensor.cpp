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
namespace quantt
{
void throw_on_bad_arg_blocks(size_t index, size_t block, size_t rank, size_t block_size)
{
	if (index >= rank)
		throw std::invalid_argument(fmt::format("given index {} is too large for rank {}.", index, rank));
	if (block >= block_size)
		std::invalid_argument(fmt::format("there are only {} blocks along the dimension {}. block requested {}", block_size, index, block));
}

size_t btensor::block_size(size_t index, size_t block) const
{
#ifndef NDEBUG
	throw_on_bad_arg_blocks(index, block, rank, sections_sizes[index]);
#endif
	auto ori = std::accumulate(sections_by_dim.begin(), sections_by_dim.begin() + index - 1, 0);
	return sections_sizes[ori + block];
}

any_quantity_cref btensor::block_conserved_qtt(size_t index, size_t block) const
{
#ifndef NDEBUG
	throw_on_bad_arg_blocks(index, block, rank, sections_sizes[index]);
#endif
	auto ori = std::accumulate(sections_by_dim.begin(), sections_by_dim.begin() + index - 1, 0);
	return c_vals[ori + block];
}

std::tuple<size_t, any_quantity_cref> btensor::block_size_cqtt(size_t index, size_t block) const
{
#ifndef NDEBUG
	throw_on_bad_arg_blocks(index, block, rank, sections_sizes[index]);
#endif
	auto ori = std::accumulate(sections_by_dim.begin(), sections_by_dim.begin() + index - 1, 0);
	return std::make_tuple(sections_sizes[ori + block], c_vals[ori + block]);
}
size_t tensor_list_size_guess(const std::initializer_list<std::initializer_list<std::tuple<size_t&, any_quantity_cref>>>& list,
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
btensor::index_list block_shapes_from_struct_list(const std::initializer_list<std::initializer_list<std::tuple<size_t&, any_quantity_cref>>>& list, size_t rank)
{
	btensor::index_list sections_by_dims(rank);
	for (size_t i = 0; i < rank; ++i) //computed the number of section along each dims, will get it from the btensor instead. implies reordering of members.
	{
		sections_by_dims[i] = list.begin()[i].size();
	}
	return sections_by_dims;
}
btensor::index_list block_sizes_from_struct_list(const std::initializer_list<std::initializer_list<std::tuple<size_t&, any_quantity_cref>>>& list, btensor::index_list sections_by_dim)
{
	btensor::index_list section_sizes(std::accumulate(sections_by_dim.begin(), sections_by_dim.end(), 1, [](auto&& a, auto&& b) { return a * b; }));
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
any_quantity_vector c_vals_from_struct_list(const std::initializer_list<std::initializer_list<std::tuple<size_t&, any_quantity_cref>>>& list, size_t size, any_quantity_cref sel_rul)
{
	any_quantity_vector c_vals(size, sel_rul.neutral());
	size_t i = 0;
	for (const auto& pair_list : list)
	{
		for (const auto& pair : pair_list)
		{
			c_vals[i] = std::get<1>(pair);
			++i;
		}
	}
	return c_vals;
}

btensor::btensor(size_t _rank, block_list_t _blocks, index_list _block_shapes, index_list _block_sizes,
                 any_quantity_vector _c_vals, any_quantity _sel_rule)
    : selection_rule(std::move(_sel_rule)), rank(rank), blocks(std::move(_blocks)), sections_by_dim(std::move(sections_by_dim)), sections_sizes(std::move(_block_sizes)),
      c_vals(std::move(_c_vals))
{
	std::string check_result = check_tensor(*this);
	if (check_result.size())
		throw std::invalid_argument(fmt::format("Invalid argument to construct a block tensor: {}", check_result));
}
btensor::btensor(std::initializer_list<std::initializer_list<std::tuple<size_t&, any_quantity_cref>>> dir_block_size_cqtt,
                 any_quantity_cref selection_rule)
    : selection_rule(std::move(selection_rule)), rank(dir_block_size_cqtt.size()),
      sections_by_dim(block_shapes_from_struct_list(dir_block_size_cqtt, rank)),
      sections_sizes(block_sizes_from_struct_list(dir_block_size_cqtt, sections_by_dim)),
      blocks(tensor_list_size_guess(dir_block_size_cqtt, selection_rule, rank, sections_by_dim)),
      c_vals(c_vals_from_struct_list(dir_block_size_cqtt, sections_by_dim.size(), selection_rule))
{
	// blocks(tensor_list_size_guess(dir_block_size_cqtt, selection_rule, rank, sections_by_dims)),
	std::string check_result = check_tensor(*this);
	if (check_result.size())
		throw std::invalid_argument(fmt::format("Invalid argument to construct a block tensor: {}", check_result));
}
btensor::btensor(std::initializer_list<std::initializer_list<std::tuple<size_t&, any_quantity_cref>>> dir_block_size_cqtt,
                 any_quantity_cref selection_rule, size_t num_blocks)
    : selection_rule(std::move(selection_rule)), rank(dir_block_size_cqtt.size()),
      sections_by_dim(block_shapes_from_struct_list(dir_block_size_cqtt, rank)),
      sections_sizes(block_sizes_from_struct_list(dir_block_size_cqtt, sections_by_dim)), blocks(num_blocks),
      c_vals(c_vals_from_struct_list(dir_block_size_cqtt, sections_by_dim.size(), selection_rule))
{
	std::string check_result = check_tensor(*this);
	if (check_result.size())
		throw std::invalid_argument(fmt::format("Invalid argument to construct a block tensor: {}", check_result));
}
} // namespace quantt