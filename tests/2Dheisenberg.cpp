/*
 * File: 2Dheisenberg.cpp
 * Project: QuantiT
 * File Created: Thursday, 14th October 2021 11:27:07 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * Copyright (c) 2021 Alexandre Foley
 * Licensed under GPL v3
 */
#define DOCTEST_CONFIG_IMPLEMENT

#include "blockTensor/btensor.h"
#include "dmrg.h"
#include <charconv>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <sstream>
#include <torch/torch.h>
#include <tuple>
#include <vector>

constexpr std::string_view mpo1 =
#include "2dHeisenberg/mpotensor_1.txt"
    ;
constexpr std::string_view mpo2 =
#include "2dHeisenberg/mpotensor_2.txt"
    ;
constexpr std::string_view mpo3 =
#include "2dHeisenberg/mpotensor_3.txt"
    ;
constexpr std::string_view mpo4 =
#include "2dHeisenberg/mpotensor_4.txt"
    ;
constexpr std::string_view mpo5 =
#include "2dHeisenberg/mpotensor_5.txt"
    ;
constexpr std::string_view mpo6 =
#include "2dHeisenberg/mpotensor_6.txt"
    ;
constexpr std::string_view mpo7 =
#include "2dHeisenberg/mpotensor_7.txt"
    ;
constexpr std::string_view mpo8 =
#include "2dHeisenberg/mpotensor_8.txt"
    ;
constexpr std::string_view mpo9 =
#include "2dHeisenberg/mpotensor_9.txt"
    ;
constexpr std::string_view mpo10 =
#include "2dHeisenberg/mpotensor_10.txt"
    ;
constexpr std::string_view mpo11 =
#include "2dHeisenberg/mpotensor_11.txt"
    ;
constexpr std::string_view mpo12 =
#include "2dHeisenberg/mpotensor_12.txt"
    ;
constexpr std::string_view mpo13 =
#include "2dHeisenberg/mpotensor_13.txt"
    ;
constexpr std::string_view mpo14 =
#include "2dHeisenberg/mpotensor_14.txt"
    ;
constexpr std::string_view mpo15 =
#include "2dHeisenberg/mpotensor_15.txt"
    ;
constexpr std::string_view mpo16 =
#include "2dHeisenberg/mpotensor_16.txt"
    ;
constexpr std::string_view mpo17 =
#include "2dHeisenberg/mpotensor_17.txt"
    ;
constexpr std::string_view mpo18 =
#include "2dHeisenberg/mpotensor_18.txt"
    ;
constexpr std::string_view mpo19 =
#include "2dHeisenberg/mpotensor_19.txt"
    ;
constexpr std::string_view mpo20 =
#include "2dHeisenberg/mpotensor_20.txt"
    ;
constexpr std::string_view mpo21 =
#include "2dHeisenberg/mpotensor_21.txt"
    ;
constexpr std::string_view mpo22 =
#include "2dHeisenberg/mpotensor_22.txt"
    ;
constexpr std::string_view mpo23 =
#include "2dHeisenberg/mpotensor_23.txt"
    ;
constexpr std::string_view mpo24 =
#include "2dHeisenberg/mpotensor_24.txt"
    ;
constexpr std::string_view mpo25 =
#include "2dHeisenberg/mpotensor_25.txt"
    ;
constexpr std::string_view mpo26 =
#include "2dHeisenberg/mpotensor_26.txt"
    ;
constexpr std::string_view mpo27 =
#include "2dHeisenberg/mpotensor_27.txt"
    ;
constexpr std::string_view mpo28 =
#include "2dHeisenberg/mpotensor_28.txt"
    ;
constexpr std::string_view mpo29 =
#include "2dHeisenberg/mpotensor_29.txt"
    ;
constexpr std::string_view mpo30 =
#include "2dHeisenberg/mpotensor_30.txt"
    ;
constexpr std::string_view mpo31 =
#include "2dHeisenberg/mpotensor_31.txt"
    ;
constexpr std::string_view mpo32 =
#include "2dHeisenberg/mpotensor_32.txt"
    ;

constexpr std::array<std::string_view, 32> mpo_strings = {
    mpo1,  mpo2,  mpo3,  mpo4,  mpo5,  mpo6,  mpo7,  mpo8,  mpo9,  mpo10, mpo11, mpo12, mpo13, mpo14, mpo15, mpo16,
    mpo17, mpo18, mpo19, mpo20, mpo21, mpo22, mpo23, mpo24, mpo25, mpo26, mpo27, mpo28, mpo29, mpo30, mpo31, mpo32};

/**
 * @brief Convert the string of the 2D heisenberg model continained in this source file into a structure containing bond
 * dimensions and key,value pairs.
 *
 *
 * @param mpo_txt
 * @return std::tuple<int, int, std::vector<std::pair<std::array<int, 4>, double>>>
 */
std::tuple<int, int, std::vector<std::pair<std::array<int, 4>, double>>> string2structure(
    const std::string_view &mpo_txt)
{
	size_t i = 0;
	size_t l = mpo_txt.length();
	std::vector<std::pair<std::array<int, 4>, double>> list(0);
	int rbond_size = 0;
	int lbond_size = 0;
	while (i < l)
	{
		std::array<int, 4> ind;
		for (int s = 0; s < 4; ++s)
		{
			auto x = mpo_txt.find_first_of(' ', i);
			auto substr = mpo_txt.substr(i, x);
			std::from_chars(substr.data(), substr.data() + substr.size(), ind[s]);
			i = x + 1;
		}
		auto x = mpo_txt.find_first_of('\n', i);
		auto substr = mpo_txt.substr(i, x);
		double coeff = std::atof(std::string(substr).c_str());
		i = x == std::numeric_limits<decltype(x)>::max() ? l : x + 1;
		lbond_size = std::max(ind[0], lbond_size);
		rbond_size = std::max(ind[3], rbond_size);
		list.emplace_back(ind, coeff);
	}
	return std::make_tuple(lbond_size, rbond_size, list);
}

torch::Tensor make_tensor(std::tuple<int, int, std::vector<std::pair<std::array<int, 4>, double>>> descriptor)
{
	auto lbond = std::get<0>(descriptor);
	auto rbond = std::get<1>(descriptor);
	auto &list = std::get<2>(descriptor);
	auto out = torch::zeros({lbond, 2, rbond, 2});
	for (auto &ind_coeff : list)
	{
		auto &ind = std::get<0>(ind_coeff);
		double coeff = std::get<1>(ind_coeff);
		out.index_put_({ind[0] - 1, ind[1] - 1, ind[3] - 1, ind[2] - 1}, coeff);
	}
	return out;
}

std::vector<torch::indexing::TensorIndex> Tindexing(const quantit::btensor::index_list &in)
{
	std::vector<torch::indexing::TensorIndex> out(in.size(), 0);
	for (size_t i = 0; i < in.size(); ++i)
	{
		out[i] = in[i];
	}
	return out;
}

/**
 * @brief guess the missing section sizes and conserved quantity of one dimension of the torch tensor given in input,
 * assuming the first M dimensions and the last N-M-1 are characterized by the input arguments The selection_rule of the
 * output tensor is assumed to be the product of the two input btensor.
 *
 * @param before_missing
 * @param after_missing
 * @param descriptor
 * @return quantit::btensor
 */
quantit::btensor guess_btensor(const torch::Tensor &tens, const quantit::btensor &before_missing,
                              const quantit::btensor &after_missing, torch::Scalar cutoff = 1e-16)
{
	using namespace quantit;
	auto mask = tens.abs() > cutoff;
	assert(before_missing.selection_rule->get().same_type(after_missing.selection_rule->get()));
	quantit::any_quantity neutral = before_missing.selection_rule->neutral();
	quantit::any_quantity out_sel_rule = before_missing.selection_rule->get() * after_missing.selection_rule;
	quantit::btensor::index_list index(tens.dim(), 0);
	auto sizes = tens.sizes();
	auto rank = tens.dim();
	int64_t missing_dim = tens.dim() - after_missing.dim() - 1;
	std::map<std::tuple<int64_t, quantit::any_quantity>, int64_t>
	    missing_section_sizes; // any repeat of the int in the tuple key means we've got a non-conserving tensor on our
	                           // hands
	// fmt::print("{}\n\n", mask);
	do
	{
		// fmt::print("{}", index);
		if (mask.index(Tindexing(index)).item().toBool())
		{
			auto [block_ind_before, subelement_ind_before] = before_missing.element_index_decompose(
			    btensor::index_list(index.begin(), index.begin() + before_missing.dim()));
			auto [block_ind_after, subelement_ind_after] = after_missing.element_index_decompose(
			    btensor::index_list(index.end() - after_missing.dim(), index.end()));
			// fmt::print("\n\tblock_ind_before {} block ind after {}\n", block_ind_before, block_ind_after);
			// fmt::print("\t subelements before {}  subelements after {}\n", subelement_ind_before,
			// subelement_ind_after);
			auto cval_before = before_missing.block_quantities(block_ind_before);
			auto cval_after = after_missing.block_quantities(block_ind_after);
			// fmt::print("\tcval before {}\n cval after {}", fmt::join(cval_before, ","), fmt::join(cval_after, ","));
			any_quantity cval = std::accumulate(cval_before.begin(), cval_before.end(), neutral, std::multiplies()) *
			                    std::accumulate(cval_after.begin(), cval_after.end(), neutral, std::multiplies());
			cval.inverse_();
			cval *= out_sel_rule;
			// fmt::print("\tbond index {} has conserved value {}\n", index[missing_dim], cval);
			missing_section_sizes[std::make_tuple(index[missing_dim], cval)] += 1;
		}
		// else
		// 	fmt::print("\n");
		increment_index_right(index, sizes, rank);
	} while (quantit::any_truth(index));
	// build the description of the missing dimension. also check for non-conservation
	quantit::btensor::index_list out_section_sizes;
	quantit::any_quantity_vector out_cvals(0, neutral);
	auto max_sect_size = tens.size(missing_dim);
	out_section_sizes.reserve(max_sect_size);
	out_cvals.reserve(max_sect_size);
	if (missing_section_sizes.size())
	{
		auto next = missing_section_sizes.begin();

		auto it = next++;
		int section_size = 0;
		for (; it != missing_section_sizes.end(); ++it, ++next)
		{
			auto &key = std::get<0>(*it);
			if (next != missing_section_sizes.end())
			{
				auto &next_key = std::get<0>(*next);
				auto &next_cval = std::get<1>(*next);
				if (std::get<0>(next_key) == std::get<0>(key) and next_key != key)
					throw std::logic_error(fmt::format(
					    "the input tensor cannot be conserving with the candidate shape.\n{}\nincompatible with\n{}",
					    key, next_key));
				bool same_section =
				    std::get<1>(next_key) == std::get<1>(key) and !(std::get<0>(next_key) == std::get<0>(key));
				section_size += same_section;
				if (!same_section)
				{
					out_section_sizes.emplace_back(section_size + 1);
					out_cvals.push_back((std::get<1>(key)));
					section_size = 0;
				}
			}
			else
			{
				out_section_sizes.emplace_back(section_size + 1);
				out_cvals.push_back((std::get<1>(key)));
			}
		}
	}
	auto missing_side = quantit::btensor({static_cast<int64_t>(out_section_sizes.size())}, out_cvals, out_section_sizes,
	                                    neutral, before_missing.options());
	return from_basic_tensor_like(shape_from(before_missing, missing_side, after_missing), tens, cutoff,
	                              before_missing.options());
}

int main()
{
	torch::set_num_threads(2);
	at::init_num_threads();
	torch::InferenceMode Inference_guard;
	torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kFloat64));
	doctest::Context doctest_context;
	doctest_context.addFilter("test-case-exclude",
	                          "**"); // don't run the tests. with this qtt_CHECKS, qtt_REQUIRES, etc. should work
	                                 // outside test context. not that i want to do that.
	using namespace quantit;
	quantit::MPO heis(32);
	int i = 0;
	for (auto &tens : heis)
	{
		tens = make_tensor(string2structure(mpo_strings[i]));
		++i;
	}
	quantit::bMPO bheis(32);
	using cval = quantit::conserved::Z;
	i = 0;
	auto phys = quantit::btensor({{{1, cval(-1)}, {1, cval(1)}}}, any_quantity(cval(0)));
	auto physdag = phys.conj();
	auto leftbond = quantit::btensor({{{1, cval(0)}}}, any_quantity(cval(0)));
	for (auto &tens : bheis)
	{
		// fmt::print("site {}\n\tleft bond {}\n\n",i, leftbond);
		auto before_missing = shape_from(leftbond, phys);
		tens = guess_btensor(heis[i], before_missing, physdag, 1e-4);
		auto rightbond = tens.shape_from({0, 0, -1, 0}).set_selection_rule_(any_quantity(cval(0)));
		// fmt::print("\tright bond {}\n\n", rightbond);
		leftbond = rightbond.conj();
		++i;
	}
	bheis.coalesce();
	quantit::dmrg_options dmrg_opt;
	dmrg_opt.maximum_bond = 1000;
	dmrg_opt.maximum_iterations = 50;
	{
		quantit::dmrg_log_simple logger;
		quantit::bMPS state = quantit::random_bMPS(4, bheis, any_quantity(cval(0)), 0);
		auto start = std::chrono::steady_clock::now();
		auto E0 = quantit::dmrg(bheis, state, dmrg_opt,logger);
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		fmt::print("btensor 4x8 heisenberg cylinder E0 {}, time {}\n", E0.item().toDouble(), elapsed_seconds.count());
		fmt::print("Obtained in {} iterations. Bond dimension at middle of MPS: {}.\n", logger.it_num,
		           logger.middle_bond_dim);
	}
	{
		auto size = 32;
		auto local_tens = torch::rand({4, 2, 4});
		std::string print_string = "{} sites AFM heisenberg Energy per sites {:.15}. obtained in {} seconds\n";
		quantit::dmrg_log_simple logger;
		quantit::MPS state(size, local_tens);
		{
			using namespace torch::indexing;
			state[0] = state[0].index({Slice(0, 1), Ellipsis});
			state[size - 1] = state[size - 1].index({Ellipsis, Slice(0, 1)});
		}
		// fmt::print("MPS square norm {}\n", contract(state,state));
		state[0] /= sqrt(tensordot(state[0], state[0].conj(), {0, 1, 2}, {0, 1, 2}));
		state.move_oc(state.size() - 1);
		state.move_oc(0);
		state[0] /= sqrt(tensordot(state[0], state[0].conj(), {0, 1, 2}, {0, 1, 2}));
		auto start = std::chrono::steady_clock::now();
		auto E0 = quantit::dmrg(heis, state, dmrg_opt,logger);
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		fmt::print("torch tensor 4x8 heisenberg cylinder E0 {}, time {}\n", E0.item().toDouble(),
		           elapsed_seconds.count());
		fmt::print("Obtained in {} iterations. Bond dimension at middle of MPS: {}.\n", logger.it_num,
		           logger.middle_bond_dim);
	}
}