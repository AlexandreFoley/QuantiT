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

#include "composite_group.h"
#include "groups.h"
quantt::cgroup::cgroup() : impl(std::make_unique<conc_cgroup_impl<groups::C<1>>>()) {}
using namespace quantt::groups;
template class quantt::conc_cgroup_impl<Z>;          //spin or particle
template class quantt::conc_cgroup_impl<Z, Z>;       //spin and particle
template class quantt::conc_cgroup_impl<Z, Z, C<2>>; //spin and particle + spatial inversion
template class quantt::conc_cgroup_impl<Z, Z, C<4>>; //spin and particle + rotation of the square
template class quantt::conc_cgroup_impl<Z, Z, C<6>>; //spin and particle + rotation of the hexagon
template class quantt::conc_cgroup_impl<C<2>>;       // parity or spatial inversion
template class quantt::conc_cgroup_impl<C<2>, C<2>>; //parity and spatial inversion
template class quantt::conc_cgroup_impl<C<2>, C<4>>; //parity and rotation of the square
template class quantt::conc_cgroup_impl<C<2>, C<6>>; //parity and rotation of the hexagon
template class quantt::conc_cgroup_impl<C<4>>;       //rotation of the square
template class quantt::conc_cgroup_impl<C<6>>;       //rotation of the hexagon

