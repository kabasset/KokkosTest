// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE ImageMedianFilterTest

#include "Linx/Run/ProgramContext.h"
#include "Linx/Transforms/RankFiltering.h"

#include <boost/test/unit_test.hpp>

LINX_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(crop_test)
{
  const int width = 16;
  const int height = 9;
  Linx::Image<int, 2> a("a", width, height);
  a.fill_with_offsets();

  const int radius = 1;
  auto b = Linx::median_filter("median", radius, a);
  BOOST_TEST(b.extent(0) == width - 2 * radius);
  BOOST_TEST(b.extent(1) == height - 2 * radius);

  const auto& a_on_host = Linx::on_host(a);
  const auto& b_on_host = Linx::on_host(b);
  for (int j = 0; j < height - 2 * radius; ++j) {
    for (int i = 0; i < width - 2 * radius; ++i) {
      std::vector<int> neighbors;
      for (int l = 0; l <= 2 * radius; ++l) {
        for (int k = 0; k <= 2 * radius; ++k) {
          neighbors.push_back(a_on_host(i + k, j + l));
        }
      }
      BOOST_TEST(b_on_host(i, j) == Linx::median(neighbors));
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
