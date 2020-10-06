/*
 * File: blocktensor.h
 * Project: quantt
 * File Created: Thursday, 1st October 2020 10:54:53 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Thursday, 1st October 2020 10:54:53 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#ifndef D49FFA60_85C4_431A_BA62_9B1D30D67E86
#define D49FFA60_85C4_431A_BA62_9B1D30D67E86

#include "Conserved/Composite/quantity.h"
#include "Conserved/Composite/quantity_vector.h"
#include <torch/torch.h>
#include <vector>
namespace quantt
{

class btensor
{
	using index_list = std::vector<size_t>;

	std::vector<std::pair<index_list, torch::Tensor>> blocks; //
	index_list block_shape;
	index_list block_sizes; //for non-empty slices, this is strictly redundent: the information could be found by inspecting the blocks
	//truncation should remove any and all empty slices, but user-written tensor could have empty slices.
	any_quantity_vector c_vals;  //dmrjulia equiv: qsum in the QTensor class
	any_quantity selection_rule; //dmrjulia equiv: the flux.

public:
};

} // namespace quantt

#endif /* D49FFA60_85C4_431A_BA62_9B1D30D67E86 */
