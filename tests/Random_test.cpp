// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE RandomTest

#include "Linx/Base/Random.h"
#include "Linx/Data/Sequence.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

LINX_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(generate_uniform_test)
{
  Linx::Sequence<int, 100> zero;
  auto a = Linx::generate<100>("a", Linx::UniformRng(Linx::Slice(0, 1000), 42));
  auto b = Linx::generate("b", Linx::UniformRng({0, 1000}, 42), 100);
  auto c = Linx::generate("c", Linx::UniformRng({0, 1000}, 43), 100);
  BOOST_TEST((a != zero));
  BOOST_TEST((b == a));
  BOOST_TEST((c != zero));
  BOOST_TEST((c != a));
  // FIXME test range
}

BOOST_AUTO_TEST_CASE(apply_uniform_test)
{
  auto signal = Linx::generate<100>("signal", Linx::Constant(1.));
  auto noise = Linx::generate<100>("noise", Linx::UniformRng({0., 1.}, 3));
  auto data = signal * Linx::UniformRng({0., 1.}, 3);
  BOOST_TEST((data == signal * noise));
  signal *= noise;
  BOOST_TEST((signal == data));
}

BOOST_AUTO_TEST_CASE(generate_gaussian_test)
{
  auto a = Linx::generate<100>("a", Linx::GaussianRng({100, 15}, 42));
  auto b = Linx::generate("b", Linx::GaussianRng({100, 15}, 42), 100);
  auto c = Linx::generate("c", Linx::GaussianRng({100, 15}, 43), 100);
  BOOST_TEST((b == a));
  BOOST_TEST((c != a));
  // FIXME test stats
}

BOOST_AUTO_TEST_CASE(apply_gaussian_test)
{
  auto signal = Linx::generate<100>("signal", Linx::Constant(1.));
  auto noise = Linx::generate<100>("noise", Linx::GaussianRng({0., 1.}, 3));
  auto data = signal + Linx::GaussianRng({0., 1.}, 3);
  BOOST_TEST((data == signal + noise));
  signal += noise;
  BOOST_TEST((signal == data));
}

BOOST_AUTO_TEST_CASE(poisson_stability_test)
{
  auto a = Linx::generate<100>("a", Linx::Constant(2));
  auto b = Linx::generate<100>("b", Linx::Constant(2.0));
  auto c = +a;
  Linx::for_each(
      "perturbate",
      c.domain(),
      KOKKOS_LAMBDA(Linx::Index i) { c[i] = (i % 2) * a[i]; });
  a.apply("seed 1", Linx::PoissonNoise(1));
  b.apply("seed 2", Linx::PoissonNoise(2));
  c.apply("seed 1", Linx::PoissonNoise(1));
  BOOST_TEST((b != a));
  auto diff = Linx::Sequence<int, 100>("diff"); // FIXME norm breaks when T = bool
  Linx::for_each(
      "assess stability",
      diff.domain(),
      KOKKOS_LAMBDA(Linx::Index i) { diff[i] = (i % 2) ? (c[i] == a[i]) : true; });
  BOOST_TEST(Linx::norm<0>(diff) == diff.size());
}

BOOST_AUTO_TEST_SUITE_END()
