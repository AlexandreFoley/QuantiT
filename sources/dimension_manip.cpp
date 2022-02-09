/*
 * File: dimension_manip.cpp
 * Project: QuantiT
 * File Created: Thursday, 6th August 2020 11:53:28 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Thursday, 6th August 2020 11:53:28 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */


#include "dimension_manip.h"

namespace quantit{

std::vector<int64_t> concat(std::initializer_list<torch::IntArrayRef> in)
{
	size_t size = 0;
	for (torch::IntArrayRef a:in)//compute the size
	{
		size += a.size();
	}
	std::vector<int64_t> out;
	out.reserve(size);
	for (auto a:in) //fill the thing.
	{
		out.insert(out.end(),a.begin(),a.end());
	}
	return out;
}

int64_t prod(torch::IntArrayRef A_dims)
{
	int64_t out = 1; 
	for (const auto& a: A_dims)
	{
		out *= a;
	}
	return out;
}

int64_t prod(torch::IntArrayRef dims, size_t start, size_t n)
{
	return prod(dims.slice(start,n));
}

std::vector<int64_t> reverse(torch::IntArrayRef dims)
{
	std::vector<int64_t> out(dims.size());
	out.insert(out.begin(),dims.rbegin(),dims.rend());
	return out;
}


torch::IntArrayRef slice(const std::vector<int64_t>& in, size_t start, size_t n)
{
	return torch::IntArrayRef(in).slice(start,n);
}

}//QuantiT