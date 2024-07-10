// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE ImagePixelwiseTest

#include "Linx/Data/Image.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

using Linx::ProgramContext;
BOOST_TEST_GLOBAL_FIXTURE(ProgramContext);
BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE);

BOOST_AUTO_TEST_CASE(sum_test)
{
  const int width = 4;
  const int height = 3;
  using Image = Linx::Image<int, 2>;

  Image a("a", width, height);
  a.fill(1);
  BOOST_TEST(a.contains_only(1));

  auto b = +a;
  BOOST_TEST(b.contains_only(1));
  ++b;
  BOOST_TEST(a.contains_only(1));
  BOOST_TEST(b.contains_only(2));

  auto c = a + b;
  BOOST_TEST(c.contains_only(3));
}

BOOST_AUTO_TEST_CASE(exp_test)
{
  const int width = 4;
  const int height = 3;
  using Image = Linx::Image<double, 2>;

  Image a("a", width, height);
  BOOST_TEST(a.contains_only(0.));

  auto b = exp(a); // FIXME add name as first argument
  BOOST_TEST(a.contains_only(0.));
  BOOST_TEST(b.contains_only(1.));

  a.exp();
  BOOST_TEST(a.contains_only(1.));
  BOOST_TEST(b.contains_only(1.));
}

BOOST_AUTO_TEST_SUITE_END();
