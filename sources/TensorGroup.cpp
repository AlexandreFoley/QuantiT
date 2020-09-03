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

#include "TensorGroup.h"

using namespace quantt::groups;
template class quantt::conc_cgroup_impl<U1>;
template class quantt::conc_cgroup_impl<U1,U1>;
template class quantt::conc_cgroup_impl<U1,U1,Z<2> >;
template class quantt::conc_cgroup_impl<Z<2> >;