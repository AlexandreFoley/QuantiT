/*
 * File: dimension_manip.h
 * Project: quantt
 * File Created: Thursday, 6th August 2020 11:49:57 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Thursday, 6th August 2020 11:49:58 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#ifndef CA372053_0A7E_4364_91C5_BE90784DCF42
#define CA372053_0A7E_4364_91C5_BE90784DCF42

#include <torch/torch.h>
#include <vector>

#include <fmt/core.h>
#include <fmt/ostream.h>

#include "cond_doctest.h"

namespace quantt{
/**
 * concatenate into a single list multiple tensor dimension lists.
 * concat({a,b,c},{d,e,f},{g,h,j}) -> {a,b,c,d,e,f,g,h,j}
 */
std::vector<int64_t> concat(std::initializer_list<torch::IntArrayRef> in);
template<class... Args> 
std::vector<int64_t> concat(Args... args)//allow us to call concat without putting braces.
{
	// auto x = {torch::IntArrayRef(args)...};//make sure it's a list of intarrayrefs.
	return concat({args...}); // use the implementation for lists of intarrayrefs
}



/**
 * compute the product of the dimensions given.
 * 
 * With start and n, compute the product of a slice of the dimensions, using the same rules as torch::IntArrayRef::slice;
 * It compute the product of of the slice spaning the interval [start,n[.
 */
int64_t prod(torch::IntArrayRef dims, size_t start, size_t n);
int64_t prod(torch::IntArrayRef dims);

/**
 * Reverse the order of the dimension.
 */
std::vector<int64_t> reverse(torch::IntArrayRef dims);

torch::IntArrayRef slice(const std::vector<int64_t>& in, size_t start, size_t n);
torch::IntArrayRef slice(torch::IntArrayRef in, size_t start, size_t n);


TEST_CASE("dimension manipulation tools")
{
	std::vector<int64_t> dim1({2,3,4});
	std::vector<int64_t> dim2({5,6});
	std::vector<int64_t> dim3({7,8});

	CHECK(prod(dim1)==24);
	CHECK(prod(dim2)==30);
	CHECK(prod(dim3)==56);
	CHECK(prod(dim1,0,3)==24);
	CHECK(prod(dim2,0,2)==30);
	CHECK(prod(dim3,0,2)==56);


	CHECK_NOTHROW(auto ref1 = slice(dim1,0,1));
	std::vector<int64_t> concat12 = {2,3,4,5,6};
	std::vector<int64_t> concat21 = {5,6,2,3,4} ;
	CHECK(concat12 == concat(dim1,dim2));
	CHECK(concat21 == concat(dim2,dim1));
	std::vector<int64_t> concat123 = {2,3,4,5,6,7,8};
	CHECK(concat123 == concat(dim1,dim2,dim3));

	CHECK(prod(concat(dim1,dim2,dim3)) == 40320);
}




}//quantt



#endif /* CA372053_0A7E_4364_91C5_BE90784DCF42 */
