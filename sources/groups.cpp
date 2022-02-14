/*
 * File: groups.cpp
 * Project: QuantiT
 * File Created: Tuesday, 15th September 2020 12:27:31 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Tuesday, 15th September 2020 12:27:31 pm
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * Licensed under GPL v3
 */
#include "Conserved/quantity.h"
#include <fmt/core.h>
#include <ostream>
namespace quantit
{
namespace conserved
{

std::ostream &operator<<(std::ostream &out, const Z &c)
{
	out << fmt::format("grp::Z({})", c.val);
	return out;
}


} // namespace groups
} // namespace quantit