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
#endif
	auto ori = std::accumulate(sections_by_dim.begin(), sections_by_dim.begin() + index - 1, 0);
	return sections_sizes[ori + block];
}

any_quantity_cref btensor::section_conserved_qtt(size_t index, size_t block) const
{
#ifndef NDEBUG
	throw_on_bad_arg_blocks(index, block, rank, sections_sizes[index]);
#endif
	auto ori = std::accumulate(sections_by_dim.begin(), sections_by_dim.begin() + index - 1, 0);
	return c_vals[ori + block];
}

std::tuple<size_t, any_quantity_cref> btensor::section_size_cqtt(size_t index, size_t block) const
{
#ifndef NDEBUG
	throw_on_bad_arg_blocks(index, block, rank, sections_sizes[index]);
#endif
	auto ori = std::accumulate(sections_by_dim.begin(), sections_by_dim.begin() + index - 1, 0);
	return std::make_tuple(sections_sizes[ori + block], c_vals[ori + block]);
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

btensor::btensor(size_t _rank, block_list_t _blocks, index_list _sections_by_dim, index_list _block_shapes, index_list _block_sizes,
                 any_quantity_vector _c_vals, any_quantity _sel_rule)
    : selection_rule(std::move(_sel_rule)), rank(_rank), blocks(std::move(_blocks)), sections_by_dim(std::move(_sections_by_dim)), sections_sizes(std::move(_block_sizes)),
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
torch::Tensor btensor::block_at(const index_list& block_index)
{
	return blocks.at(block_index);
}

torch::Tensor btensor::block(const index_list& block_index) //create the block if it is allowed.
{
	return blocks[block_index];
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
				sel_test += T.section_conserved_qtt(i, ind[i]);
				cq += fmt::format("index {}: {}\n", i, T.section_conserved_qtt(i, ind[i]));
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
btensor::const_block_size_view btensor::block_size(index_list block_index) const
{
	auto a = sections_sizes.begin();
	return const_block_size_view(sections_sizes.begin(), sections_sizes.end(), sections_by_dim, std::move(block_index));
}

} // namespace quantt