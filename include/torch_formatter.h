/*
 * File: tens_formatter.h
 * Project: QuantiT
 * File Created: Tuesday, 21st July 2020 6:07:08 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Thursday, 23rd July 2020 10:47:14 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */
#ifndef FD137415_7533_4BFF_A130_17B4315FE304
#define FD137415_7533_4BFF_A130_17B4315FE304


#include <torch/torch.h>
#include <fmt/ostream.h>
#include <fmt/format.h>
#include <charconv>
namespace quantit
{
	void print(const torch::Tensor& X);
}
/**
 * @brief fmt::formatter for torch::Tensor
 * 
 */
template <>
struct fmt::formatter<torch::Tensor>
{
	uint linelenght = 80;

	constexpr auto parse(format_parse_context& ctx)
	{
		auto it = ctx.begin(), end = ctx.end();
		if (it and *it != '}')
		{
			++it;
			auto first_pos = it;
			while (it != end && *it != '}')
			{
				++it;
			}
			auto last_pos = it - 1;
			uint width = 80;
			if (last_pos > first_pos)
			{
				auto code = std::from_chars(first_pos, last_pos, width);
				if (code.ptr != last_pos || std::errc::invalid_argument == code.ec)
					throw format_error("invalid format, only a single integer supported for torch::Tensor");
				else
				{
					linelenght = width;
				}
			}
		}
		if (it and *it != '}')
			throw format_error("invalid format,closing brace missing");

		// Return an iterator past the end of the parsed range:
		return it;
	}

	template <typename FormatContext>
	auto format(const torch::Tensor &tens, FormatContext &ctx)
	{
		std::stringstream strstr;
		at::print(strstr, tens, linelenght);
		std::string tens_string(strstr.str());
		auto end = tens_string.length();
		// find occurences of {} and replace with ()
		// done because fmt use {} to mark format section which can lead to weird interactions...
		auto it = tens_string.find("{", 0);
		while (it < end)
		{
			tens_string[it] = '(';
			++it;
			it = tens_string.find("{", it);
		}
		it = tens_string.find("}", 0);
		while (it < end)
		{
			tens_string[it] = ')';
			++it;
			it = tens_string.find("}", it);
		}

		return format_to(ctx.out(), tens_string);
	}
};

#endif /* FD137415_7533_4BFF_A130_17B4315FE304 */
