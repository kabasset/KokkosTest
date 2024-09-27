// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE PositionArithmeticTest

#include "Linx/Data/Box.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

LINX_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(scalar_additive_test)
{
  Linx::Position<4> in {0, 1, 2, 3};
  auto plus = in + 1;
  auto minus = in - 1;
  for (std::size_t i = 0; i < in.size(); ++i) {
    BOOST_TEST(plus[i] == i + 1);
    BOOST_TEST(minus[i] == i - 1);
  }
}

BOOST_AUTO_TEST_CASE(vector_additive_test)
{
  Linx::Position<4> in {0, 1, 2, 3};
  Linx::Position<4> delta {1, 1, 1, 1};
  auto plus = in + delta;
  auto minus = in - delta;
  for (std::size_t i = 0; i < in.size(); ++i) {
    BOOST_TEST(plus[i] == i + 1);
    BOOST_TEST(minus[i] == i - 1);
  }
}

BOOST_AUTO_TEST_SUITE_END()
