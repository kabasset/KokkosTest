// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE BoxArithmeticTest

#include "Linx/Data/Image.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

using Linx::ProgramContext;
BOOST_TEST_GLOBAL_FIXTURE(ProgramContext);
BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE);

BOOST_AUTO_TEST_CASE(scalar_additive_test)
{
  Linx::Box<int, 2> in({0, 1}, {2, 3});
  auto plus = in + 1;
  auto minus = in - 1;
  BOOST_TEST((plus.start() == in.start() + 1));
  BOOST_TEST((plus.stop() == in.stop() + 1));
  BOOST_TEST((minus.start() == in.start() - 1));
  BOOST_TEST((minus.stop() == in.stop() - 1));
}

BOOST_AUTO_TEST_CASE(vector_additive_test)
{
  auto in = Linx::Box<int, 2>({0, 1}, {2, 3});
  auto delta = Linx::Box<int, 2>::value_type {-1, 1};
  auto plus = in + delta;
  auto minus = in - delta;
  BOOST_TEST((plus.start() == in.start() + delta));
  BOOST_TEST((plus.stop() == in.stop() + delta));
  BOOST_TEST((minus.start() == in.start() - delta));
  BOOST_TEST((minus.stop() == in.stop() - delta));
}

BOOST_AUTO_TEST_CASE(box_additive_test)
{
  auto in = Linx::Box<int, 2>({-10, -1}, {2, 3});
  auto margin = Linx::Box<int, 2>({-1, -2}, {2, 1});
  auto plus = in + margin;
  auto minus = in - margin;
  BOOST_TEST((plus.start() == in.start() + margin.start()));
  BOOST_TEST((plus.stop() == in.stop() + margin.stop()));
  BOOST_TEST((plus.shape() == in.shape() + margin.shape()));
  BOOST_TEST((minus.start() == in.start() - margin.start()));
  BOOST_TEST((minus.stop() == in.stop() - margin.stop()));
  BOOST_TEST((minus.shape() == in.shape() - margin.shape()));
}

BOOST_AUTO_TEST_SUITE_END();
