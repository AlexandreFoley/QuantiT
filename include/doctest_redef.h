/*
 * File: doctest_redef.h
 * Project: include
 * File Created: Sunday, 2nd August 2020 12:06:35 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Sunday, 2nd August 2020 12:06:35 pm
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#ifndef C8CCE3D5_354C_49A9_B03C_3B444488A93D
#define C8CCE3D5_354C_49A9_B03C_3B444488A93D

//eliminate macro redef warning due to pytorch and doctest defining macros with the same name.

#undef CHECK_LE
#undef CHECK_GE
#undef CHECK_GE
#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_GT
#undef CHECK_LT
#undef CHECK_LE
#undef CHECK
#endif /* C8CCE3D5_354C_49A9_B03C_3B444488A93D */
