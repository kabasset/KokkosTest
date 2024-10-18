// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE AlgorithmSort

#include "Linx/Base/Algorithm.h"
#include "Linx/Data/Sequence.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>
#include <ranges>

LINX_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(odd_increasing_test)
{
  Linx::Sequence a({0, 1, 10, 100, 1000}); // FIXME on host
  BOOST_TEST(std::ranges::is_sorted(a));
  BOOST_TEST(Linx::median(a) == 10);
  BOOST_TEST(Linx::sort_n(a, 0) == 0);
  BOOST_TEST(Linx::sort_n(a, 1) == 1);
  BOOST_TEST(Linx::sort_n(a, 2) == 10);
  BOOST_TEST(Linx::sort_n(a, 3) == 100);
  BOOST_TEST(Linx::sort_n(a, 4) == 1000);
}

BOOST_AUTO_TEST_CASE(odd_decreasing_test)
{
  Linx::Sequence a({1000, 100, 10, 1, 0}); // FIXME on host
  BOOST_TEST(not std::ranges::is_sorted(a));
  BOOST_TEST(Linx::sort_n(a, 0) == 0);
  BOOST_TEST(Linx::sort_n(a, 1) == 1);
  BOOST_TEST(not std::ranges::is_sorted(a));
  BOOST_TEST(Linx::sort_n(a, 2) == 10);
  BOOST_TEST(Linx::sort_n(a, 3) == 100);
  BOOST_TEST(Linx::sort_n(a, 4) == 1000);
  BOOST_TEST(std::ranges::is_sorted(a));
  BOOST_TEST(Linx::median(a) == 10);
}

BOOST_AUTO_TEST_CASE(even_random_test)
{
  Linx::Sequence a({1, 100, 0, 10, 10000, 1000}); // FIXME on host
  BOOST_TEST(Linx::sort_n(a, 1) == 1);
  BOOST_TEST(not std::ranges::is_sorted(a));
  BOOST_TEST(std::ranges::is_sorted_until(a.data(), a.data() + 2));
  BOOST_TEST(Linx::median(a) == 550);
  BOOST_TEST(std::ranges::is_sorted_until(a.data(), a.data() + 4));
  BOOST_TEST(Linx::sort_n(a, 5));
  BOOST_TEST(std::ranges::is_sorted(a));
}

BOOST_AUTO_TEST_SUITE_END()
