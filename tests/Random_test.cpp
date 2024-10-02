// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE RandomTest

#include "Linx/Data/Random.h"
#include "Linx/Data/Sequence.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

LINX_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(uniform_test)
{
  Linx::Sequence<int, 100> zero;
  auto a = Linx::generate<100>("a", Linx::UniformNoise(Linx::Slice(0, 1000), 42));
  auto b = Linx::generate("b", Linx::UniformNoise(0, 1000, 42), 100);
  auto c = Linx::generate("c", Linx::UniformNoise(0, 1000, 43), 100);
  BOOST_TEST((a != zero));
  BOOST_TEST((b == a));
  BOOST_TEST((c != zero));
  BOOST_TEST((c != a));
  // FIXME test range
}

BOOST_AUTO_TEST_SUITE_END()
