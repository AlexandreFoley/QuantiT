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

using namespace quantt::groups;
template class quantt::conc_cgroup_impl<Z>;
template class quantt::conc_cgroup_impl<Z, Z>;
template class quantt::conc_cgroup_impl<Z, Z, C<2>>;
template class quantt::conc_cgroup_impl<C<2>>;