// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE ImageApplyTest

#include "Linx/Data/Image.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

using Linx::ProgramContext;
BOOST_TEST_GLOBAL_FIXTURE(ProgramContext);
BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE);

BOOST_AUTO_TEST_CASE(apply_test)
{
  const int width = 4;
  const int height = 3;
  using Image = Linx::Image<int, 2>;
  Image a("a", width, height);
  auto shape = a.shape();
  auto domain = a.domain();
  Image b("b", shape);

  domain.iterate(
      "init",
      KOKKOS_LAMBDA(int i, int j) {
        a(i, j) = i + 2 * j;
        b(i, j) = 3;
      });

  a.apply(
      "eval",
      [](auto ai, auto bi) {
        return ai * ai + bi;
      },
      b);
  Kokkos::fence();

  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      BOOST_TEST(a(i, j) == i * i + 4 * i * j + 4 * j * j + 3);
    }
  }
}

BOOST_AUTO_TEST_CASE(reduce_test)
{
  const int width = 4;
  const int height = 3;
  using Image = Linx::Image<int, 2>;
  Image a("a", width, height);

  a.domain().iterate(
      "init",
      KOKKOS_LAMBDA(int i, int j) { a(i, j) = 2; });

  auto sum = a.reduce("sum", [](auto e, auto f) {
    return e + f;
  });
  Kokkos::fence();

  BOOST_TEST(sum == 2 * width * height);
}

BOOST_AUTO_TEST_SUITE_END();
