/*
 * File: catch_test.cpp
 * Project: QuanTT
 * File Created: Friday, 31st July 2020 11:04:58 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Friday, 31st July 2020 11:04:58 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
// with the preceding define, doctest autogenerate a main() function that run the test found in the sources included.

#include "MPT.h"
#include "dimension_manip.h"
#include "LinearAlgebra.h"
#include "dmrg.h"
#include "operators.h"
#include "models.h"
#include "TensorGroup.h"
