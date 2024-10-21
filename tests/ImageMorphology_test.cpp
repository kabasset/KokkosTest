// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE ImageMorphologyTest

#include "Linx/Run/ProgramContext.h"
#include "Linx/Transforms/Morphology.h"

#include <boost/test/unit_test.hpp>

LINX_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(crop_test)
{
  const int width = 16;
  const int height = 9;
  const auto zero = Linx::Image<bool, 2>("0", width, height);
  const auto one = Linx::Image<bool, 2>("1", width, height).fill(true);

  const int radius = 1;
  auto min_0 = Linx::erode("min", radius, zero);
  auto max_0 = Linx::dilate("max", radius, zero);
  auto min_1 = Linx::erode("min", radius, one);
  auto max_1 = Linx::dilate("max", radius, one);

  const auto& min_0_on_host = Linx::on_host(min_0);
  const auto& max_0_on_host = Linx::on_host(max_0);
  const auto& min_1_on_host = Linx::on_host(min_1);
  const auto& max_1_on_host = Linx::on_host(max_1);
  for (int j = 0; j < height - 2 * radius; ++j) {
    for (int i = 0; i < width - 2 * radius; ++i) {
      BOOST_TEST(min_0_on_host(i, j) == false);
      BOOST_TEST(max_0_on_host(i, j) == false);
      BOOST_TEST(min_1_on_host(i, j) == true);
      BOOST_TEST(max_1_on_host(i, j) == true);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
