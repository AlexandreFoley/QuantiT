/*
 * File: benchmark.cpp
 * Project: QuantiT
 * File Created: Monday, 20th September 2021 12:59:11 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * Copyright (c) 2021 Alexandre Foley
 * Licensed under GPL v3
 */
// #if E_PROFILER
// #define DOCTEST_CONFIG_DISABLE
// #else
#define DOCTEST_CONFIG_IMPLEMENT
// #endif
#define ANKERL_NANOBENCH_IMPLEMENT

#include "DMRG_heisenberg_test.h"
#include "blockTensor/btensor.h"
#include "nanobench.h"
#include "tensorgdot.h"
#ifdef E_PROFILER
#include <gperftools/profiler.h>
#endif
#include <random>
#include <torch/torch.h>
	// #include <chrono>
	// #include <fmt/core.h>
	// #include <fmt/chrono.h>

	template <class T>
	void DNO(T &&t)
	{
		ankerl::nanobench::doNotOptimizeAway(t);
	};

// // the basic case, more option are available without using this macro.
// // use inside main()
#define BENCHMARK(name, function) ankerl::nanobench::Bench().run(name, function)

// TODO: performance test on the trivial group
// TODO: performance test on a two dimensionnal latice? no easy way to generate that right away.

int main()
{
	doctest::Context doctest_context;
	doctest_context.addFilter("test-case-exclude",
	                          "**"); // don't run the tests. with this qtt_CHECKS, qtt_REQUIRES, etc. should work
	                                 // outside test context. not that i want to do that.

	torch::set_num_threads(1);
	torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kFloat64));
	torch::InferenceMode _GUARD;
	// {	using cval = quantit::quantity<quantit::conserved::Z>;
	// 	auto r = std::mt19937(std::random_device()());
	// 	auto size_generator = std::uniform_int_distribution(10, 40); // size generator
	// 	auto cval_generator = std::uniform_int_distribution(-5, 5);  // conserved value generator
	// 	auto sg = [&]() { return size_generator(r); };
	// 	auto cg = [&]() { return cval_generator(r); };
	// 	auto shapeA =
	// 	    quantit::btensor({{{sg(), cval(cg())}, {sg(), cval(cg())}, {sg(), cval(cg())}, {sg(), cval(cg())}}},
	// cval(0)); 	auto shapeB = quantit::btensor({{{1, cval(cg())}, {1, cval(cg())}, {1, cval(cg())}}}, cval(0)); 	auto
	// shapeC = quantit::shape_from(shapeA, shapeB).reshape({}).conj(); 	auto shapeX = shape_from(shapeA, shapeB, shapeC);
	// 	auto tX = torch::rand({200, 200});
	// 	auto X = quantit::rand_like(shapeX);
	// 	auto Y = X.conj();

	// quantit::btensor Z;}
	// DNO(Z = X.tensordot(Y, {0}, {0}));
	// auto tZ = torch::tensordot(tX,tX,{0},{0});
	// fmt::print("{}\n",Z.begin());
	// ankerl::nanobench::Bench().minEpochIterations(1000).run("tensordot",
	//                                                         [&]()
	//                                                         {
	// 	                                                        DNO(Z = X.tensordot(Y, {0}, {0}));
	// 	                                                        DNO(&(*(Z.begin())));
	//                                                         });
	// auto O=torch::zeros({1,1});
	// auto X=torch::rand({1,1});
	// auto Y=torch::rand({1,1});
	// auto oo = O.clone();
	// using namespace ankerl::nanobench;
	// Bench().minEpochIterations(5000).run("addmm single value",[&]()
	// {
	// 	// oo *= O;
	// 	DNO(oo.addmm_(X,Y));
	// });
	// Bench().minEpochIterations(5000).run("mult-add value",[&]()
	// {
	// 	// oo *= O;
	// 	DNO(oo.addcmul_(X,Y));
	// });
	// ankerl::nanobench::Bench().minEpochIterations(1000).run("tensordot",[&]()
	// {
	// 	DNO(O += torch::tensordot(X,Y,{2},{0}));//in principle, tensorgdot should be an optimization of this. I suspect it's currently slower.
	// 	DNO(O*=0);//zeros it out
	// });
	// ankerl::nanobench::Bench().minEpochIterations(1000).run("tensorgdot",[&]()
	// {
	// 	DNO(quantit::tensorgdot(O,X,Y,{2},{0}));
	// 	DNO(O*=0);//zeros it out
	// });
	#ifdef E_PROFILER
	(ProfilerStop());
	(ProfilerStart("btensor.out"));
	#endif
	
	{
		Heisen_afm_test_bt(50);
		Heisen_afm_test_bt(50);
		Heisen_afm_test_bt(50);
	#ifdef E_PROFILER
	(ProfilerStop());
	(ProfilerStart("torch.out"));
	#endif
		Heisen_afm_test_tt(50);
		Heisen_afm_test_tt(50);
		Heisen_afm_test_tt(50);
		// Heisen_afm_test_tt(50);
		// Heisen_afm_test_tt(50);
		// Heisen_afm_test_tt(50);
	}
#ifdef E_PROFILER
	(ProfilerStop());
#endif
}