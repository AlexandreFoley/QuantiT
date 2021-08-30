/*
 * File: TensorGroup.cpp
 * Project: quantt
 * File Created: Tuesday, 1st September 2020 2:21:57 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Tuesday, 1st September 2020 2:21:57 pm
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */
#include "Conserved/Composite/cquantity.h"
#include "Conserved/Composite/quantity_vector.h"
#include "Conserved/quantity.h"
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
quantt::any_quantity::any_quantity() : impl(std::make_unique<quantity<conserved::C<1>>>()) {}
using namespace quantt::conserved;
template class quantt::quantity<Z>;          //spin or particle
template class quantt::quantity<Z, Z>;       //spin and particle
template class quantt::quantity<Z, Z, C<2>>; //spin and particle + spatial inversion
template class quantt::quantity<Z, Z, C<4>>; //spin and particle + rotation of the square
template class quantt::quantity<Z, Z, C<6>>; //spin and particle + rotation of the hexagon
template class quantt::quantity<C<2>>;       // parity or spatial inversion
template class quantt::quantity<C<2>, C<2>>; //parity and spatial inversion
template class quantt::quantity<C<2>, C<4>>; //parity and rotation of the square
template class quantt::quantity<C<2>, C<6>>; //parity and rotation of the hexagon
template class quantt::quantity<C<4>>;       //rotation of the square
template class quantt::quantity<C<6>>;       //rotation of the hexagon
                                             // template class fmt::v6::formatter<quantt::conserved::C<2>>;

std::string to_string(quantt::any_quantity_cref cqtt)
{
	return fmt::format("{}",cqtt);
}
