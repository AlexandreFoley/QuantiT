/*
 * File: groups.cpp
 * Project: quantt
 * File Created: Tuesday, 15th September 2020 12:27:31 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Tuesday, 15th September 2020 12:27:31 pm
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */
#include "groups.h"
#include <ostream>
namespace quantt
{
namespace groups
{

std::ostream &operator<<(std::ostream &out, const Z &c)
{
	out << "grp::Z(" << c.val << ')';
	return out;
}


} // namespace groups
} // namespace quantt