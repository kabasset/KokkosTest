// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE "KokkosTest"

#include "Linx/Run/ProgramContext.h"
#include "Linx/Transforms/Correlation.h"

#include <boost/test/unit_test.hpp>

using Linx::ProgramContext;
BOOST_TEST_GLOBAL_FIXTURE(ProgramContext);

BOOST_AUTO_TEST_SUITE(ImageCorrelateTest);

BOOST_AUTO_TEST_CASE(crop_test)
{
  const int width = 4;
  const int height = 3;
  const int kernel = 2;
  using Image = Linx::Image<int, 2>;
  Image a("a", width, height);
  Image k("k", kernel, kernel);
  a.generate(
      "init a",
      KOKKOS_LAMBDA() { return 1; });
  k.generate(
      "init k",
      KOKKOS_LAMBDA() { return 1; });
  Kokkos::fence();

  auto b = correlate("mean", a, k);

  for (int j = 0; j < height - kernel; ++j) {
    for (int i = 0; i < width - kernel; ++i) {
      BOOST_TEST(b(i, j) == kernel * kernel);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();
