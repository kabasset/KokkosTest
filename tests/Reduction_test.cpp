// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE DistributionTest

#include "Linx/Data/Image.h"
#include "Linx/Run/ProgramContext.h"
#include "Linx/Transforms/Reduction.h"

#include <boost/test/unit_test.hpp>

using Linx::ProgramContext;
BOOST_TEST_GLOBAL_FIXTURE(ProgramContext);
BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE);

BOOST_AUTO_TEST_CASE(sum_test)
{
  const int width = 4;
  const int height = 3;
  Linx::Image<int, 2> a("a", width, height);

  a.domain().iterate(
      "range",
      KOKKOS_LAMBDA(int i, int j) { a(i, j) = i + j * width; });

  auto sum = Linx::sum("sum", a);

  BOOST_TEST(sum == a.size() * (a.size() - 1) / 2);
}

BOOST_AUTO_TEST_CASE(dot_test)
{
  const int width = 2;
  const int height = 2;
  Linx::Image<int, 2> a("a", width, height);

  a.domain().iterate(
      "range",
      KOKKOS_LAMBDA(int i, int j) { a(i, j) = i + j * width; });

  auto norm = Linx::dot("norm", a, a);

  BOOST_TEST(norm == 14);
}

BOOST_AUTO_TEST_CASE(dot_1d_test)
{
  Linx::Image<int, 1> a("a", 4);

  for (std::size_t i = 0; i < a.size(); ++i) {
    a(i) = i;
  }

  auto norm = Linx::dot("norm", a, a);

  BOOST_TEST(norm == 14);
}

BOOST_AUTO_TEST_SUITE_END();
