// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE TilingTest

#include "Linx/Data/Image.h"
#include "Linx/Data/Tiling.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

LINX_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(rows_test)
{
  auto image = Linx::Image<int, 3>("image", 16, 9, 4);
  Linx::for_each<Kokkos::DefaultHostExecutionSpace>(
      "fill",
      image.domain(),
      KOKKOS_LAMBDA(auto i, auto j, auto k) { image(i, j, k) = i; });
  Linx::Position<16> sum;
  for (const auto& row : Linx::rows(image)) {
    BOOST_TEST(row.size() == image.extent(0));
    BOOST_TEST(row.size() == sum.size());
    for (int i = 0; i < row.size(); ++i) {
      sum[i] += row.local(i);
    }
  }
  for (std::size_t i = 0; i < sum.size(); ++i) {
    BOOST_TEST(sum[i] == 9 * 4 * i);
  }
}

BOOST_AUTO_TEST_CASE(profiles_test)
{
  auto image = Linx::Image<int, 3>("image", 16, 9, 4);
  Linx::for_each<Kokkos::DefaultHostExecutionSpace>(
      "fill",
      image.domain(),
      KOKKOS_LAMBDA(auto i, auto j, auto k) { image(i, j, k) = j; });
  Linx::Position<9> sum;
  for (const auto& column : Linx::profiles<1>(image)) {
    BOOST_TEST(column.size() == image.extent(1));
    BOOST_TEST(column.size() == sum.size());
    for (int i = 0; i < column.size(); ++i) {
      sum[i] += column.local(i);
    }
  }
  for (std::size_t i = 0; i < sum.size(); ++i) {
    BOOST_TEST(sum[i] == 16 * 4 * i);
  }
}

BOOST_AUTO_TEST_SUITE_END()
