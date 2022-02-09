/*
 * File: tens_formatter.cpp
 * Project: QuantiT
 * File Created: Tuesday, 21st July 2020 6:10:10 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Thursday, 23rd July 2020 10:46:48 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

// nothing in here... the parser of the tensor formatter could be move in here, but that would not lead to significant reduction of dependency
// most of them stem from the format function which is a template

#include "torch_formatter.h"

void quantit::print(const torch::Tensor& X)
{
	fmt::print("{}\n\n", X);
}
