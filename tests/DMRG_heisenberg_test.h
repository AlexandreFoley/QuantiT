/*
 * File: DMRG_heisenberg_test.h
 * Project: quantt
 * File Created: Wednesday, 19th August 2020 11:39:47 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Wednesday, 19th August 2020 11:41:35 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#ifndef E0106B85_7787_42FD_9E7E_47803E425A61
#define E0106B85_7787_42FD_9E7E_47803E425A61

#include "dmrg.h"
#include "models.h"
#include "cond_doctest.h"
#include "torch_formatter.h"

TEST_CASE("solving the heisenberg model")
{
	torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kFloat64));
	SUBCASE("10 sites AFM")
	{
		auto hamil = quantt::Heisenberg(1,10);
		quantt::dmrg_options options;
		auto [E0,state] = quantt::dmrg(hamil,options);
		fmt::print("10 sites AFM heisenberg Energy {}\n",E0.to<double>());
	}
	SUBCASE("20 sites AFM")
	{
		auto hamil = quantt::Heisenberg(1,20);
		quantt::dmrg_options options;
		auto [E0,state] = quantt::dmrg(hamil,options);
		fmt::print("20 sites AFM heisenberg Energy {}\n",E0.to<double>());
	}
	SUBCASE("50 sites AFM")
	{
		auto hamil = quantt::Heisenberg(1,50);
		quantt::dmrg_options options;
		auto [E0,state] = quantt::dmrg(hamil,options);
		fmt::print("50 sites AFM heisenberg Energy {}\n",E0.to<double>());
	}
}


#endif /* E0106B85_7787_42FD_9E7E_47803E425A61 */
