// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE ImageMedianFilterTest

#include "Linx/Run/ProgramContext.h"
#include "Linx/Transforms/RankFiltering.h"

#include <boost/test/unit_test.hpp>

LINX_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(crop_test)
{
  const int width = 4;
  const int height = 3;
  Linx::Image<int, 2> a("a", width, height);
  a.fill_with_offsets();

  const int radius = 1;
  auto b = Linx::median_filter("median", a, radius);

  const auto& a_on_host = Linx::on_host(a);
  const auto& b_on_host = Linx::on_host(b);
  for (int j = 0; j < height - 2 * radius; ++j) {
    for (int i = 0; i < width - 2 * radius; ++i) {
      std::cout << i << ", " << j << std::endl;
      std::vector<int> neighbors;
      for (int l = 0; l <= 2 * radius; ++l) {
        std::cout << "  ";
        for (int k = 0; k <= 2 * radius; ++k) {
          neighbors.push_back(a_on_host(i + k, j + l));
          std::cout << neighbors.back() << " ";
        }
        std::cout << std::endl;
      }
      std::cout << b_on_host(i, j) << " =?= " << Linx::median(neighbors) << std::endl;
      BOOST_TEST(b_on_host(i, j) == Linx::median(neighbors));
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
