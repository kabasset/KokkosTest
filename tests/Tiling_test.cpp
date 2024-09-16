// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE TilingTest

#include "Linx/Data/Raster.h"
#include "Linx/Data/Tiling.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>
#include <sstream>

LINX_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(rows_test)
{
  auto image = Linx::HostRaster<int, 3>("image", 16, 9, 4);
  image.fill(1);
  Linx::Sequence<int, 16> sum;
  for (const auto& row : Linx::rows(image)) {
    for (std::size_t i = 0; i < row.size(); ++i) {
      sum[i] += row.local(i);
    }
  }
  for (auto e : sum) {
    BOOST_TEST(e == 9 * 4);
  }
}

BOOST_AUTO_TEST_CASE(profiles_test)
{
  auto image = Linx::HostRaster<int, 3>("image", 16, 9, 4);
  image.fill(1);
  Linx::Sequence<int, 9> sum;
  for (const auto& column : Linx::profiles<1>(image)) {
    for (std::size_t i = 0; i < column.size(); ++i) {
      sum[i] += column.local(i);
    }
  }
  for (auto e : sum) {
    BOOST_TEST(e == 16 * 4);
  }
}

BOOST_AUTO_TEST_SUITE_END()
