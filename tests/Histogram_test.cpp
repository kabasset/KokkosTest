// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE HistogramTest

#include "Linx/Data/Distribution.h"
#include "Linx/Data/Raster.h"
#include "Linx/Run/ProgramContext.h"
#include "Linx/Transforms/Reduction.h"

#include <boost/test/unit_test.hpp>

using Linx::ProgramContext;
BOOST_TEST_GLOBAL_FIXTURE(ProgramContext);
BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE);

BOOST_AUTO_TEST_CASE(min_max_test)
{
  const int size = 10;
  using Image = Linx::Raster<int, 1>;
  const auto a = Image("a", size).fill_with_offsets();
  const auto bins = Linx::Sequence<int, 2> {Linx::min(a), Linx::max(a) + 1};
  const auto histogram = Linx::histogram(a, bins);
  BOOST_TEST(histogram.size() == 1);
  BOOST_TEST(histogram[0] == size);
}

BOOST_AUTO_TEST_CASE(min_maxm1_max_test)
{
  const int size = 10;
  using Image = Linx::Raster<int, 1>;
  const auto a = Image("a", size).fill_with_offsets();
  const auto bins = Linx::Sequence<int, 3> {Linx::min(a), Linx::max(a), Linx::max(a) + 1};
  const auto histogram = Linx::histogram(a, bins);
  BOOST_TEST(histogram.size() == 2);
  BOOST_TEST(histogram[0] == size - 1);
  BOOST_TEST(histogram[1] == 1);
}

BOOST_AUTO_TEST_CASE(inner_test)
{
  const int size = 10;
  using Image = Linx::Raster<int, 1>;
  const auto a = Image("a", size).fill_with_offsets();
  const auto bins = Linx::Sequence<int, 2> {Linx::min(a) + 1, Linx::max(a)};
  const auto histogram = Linx::histogram(a, bins);
  BOOST_TEST(histogram.size() == 1);
  BOOST_TEST(histogram[0] == size - 2);
}

BOOST_AUTO_TEST_SUITE_END();
