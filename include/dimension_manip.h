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

namespace quantt{
/**
 * concatenate into a single list multiple tensor dimension lists.
 * concat({a,b,c},{d,e,f},{g,h,j}) -> {a,b,c,d,e,f,g,h,j}
 */
std::vector<int64_t> concat(std::initializer_list<torch::IntArrayRef> in);
template<class... Args> 
std::vector<int64_t> concat(Args... args)//allow us to call concat without putting braces.
{// this should not cause any code bloat because the optimizer has no reason not to inline that sort of function.
	return concat({args...});
}
/**
 * compute the product of the dimensions given.
 * 
 * With start and n, compute the product of a slice of the dimensions, using the same rules as torch::IntArrayRef::slice;
 * It compute the product of of the slice spaning the interval [start,n].
 */
int64_t prod(torch::IntArrayRef dims, size_t start, size_t n);
int64_t prod(torch::IntArrayRef dims);

/**
 * Reverse the order of the dimension.
 */
std::vector<int64_t> reverse(torch::IntArrayRef dims);

}//quantt



#endif /* CA372053_0A7E_4364_91C5_BE90784DCF42 */
