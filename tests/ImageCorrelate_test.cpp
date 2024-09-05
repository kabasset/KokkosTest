// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE ImageCorrelateTest

#include "Linx/Run/ProgramContext.h"
#include "Linx/Transforms/Correlation.h"

#include <boost/test/unit_test.hpp>

using Linx::ProgramContext;
BOOST_TEST_GLOBAL_FIXTURE(ProgramContext);
BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE);

BOOST_AUTO_TEST_CASE(crop_to_test)
{
  const int width = 4;
  const int height = 3;
  const int kernel = 2;
  using Image = Linx::Image<int, 2>;
  Image a("a", width, height);
  Image k("k", kernel, kernel);
  Image b("b", width - kernel + 1, height - kernel + 1);
  a.fill(1);
  k.fill(1);
  Kokkos::fence();

  correlate_to(a, k, b);

  const auto& b_on_host = Linx::on_host(b);
  for (int j = 0; j < height - kernel; ++j) {
    for (int i = 0; i < width - kernel; ++i) {
      BOOST_TEST(b_on_host(i, j) == kernel * kernel);
    }
  }
}

BOOST_AUTO_TEST_CASE(crop_test)
{
  const int width = 4;
  const int height = 3;
  const int kernel = 2;
  using Image = Linx::Image<int, 2>;
  Image a("a", width, height);
  Image k("k", kernel, kernel);
  a.fill(1);
  k.fill(1);
  Kokkos::fence();

  auto b = correlate("mean", a, k);

  const auto& b_on_host = Linx::on_host(b);
  for (int j = 0; j < height - kernel; ++j) {
    for (int i = 0; i < width - kernel; ++i) {
      BOOST_TEST(b_on_host(i, j) == kernel * kernel);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();
