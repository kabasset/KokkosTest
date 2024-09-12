// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE ImageApplyTest

#include "Linx/Data/Image.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

LINX_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(apply_test)
{
  const int width = 4;
  const int height = 3;
  using Image = Linx::Image<int, 2>;
  Image a("a", width, height);
  auto shape = a.shape();
  auto domain = a.domain();
  Image b("b", shape);

  for_each(
      "init",
      domain,
      KOKKOS_LAMBDA(int i, int j) {
        a(i, j) = i + 2 * j;
        b(i, j) = 3;
      });

  a.apply(
      "eval",
      KOKKOS_LAMBDA(int ai, int bi) { return ai * ai + bi; },
      b);
  Kokkos::fence();

  const auto& a_on_host = Linx::on_host(a);
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      BOOST_TEST(a_on_host(i, j) == i * i + 4 * i * j + 4 * j * j + 3);
    }
  }
}

BOOST_AUTO_TEST_CASE(copy_test)
{
  const int width = 4;
  const int height = 3;
  using Right = Linx::Image<int, 2, Linx::DefaultContainer<int, 2, Kokkos::LayoutRight>::Image>;
  using Left = Linx::Image<int, 2, Linx::DefaultContainer<int, 2, Kokkos::LayoutLeft>::Image>;
  auto right = Right("right", width, height).fill_with_offsets();
  auto left = Left("left", width, height).copy_from(right);
  Kokkos::fence();

  const auto& left_on_host = Linx::on_host(left);
  const auto& right_on_host = Linx::on_host(right);
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      BOOST_TEST(left_on_host(i, j) == right_on_host(i, j));
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
