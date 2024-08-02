// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE SequenceArithmeticTest

#include "Linx/Data/Sequence.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

using Linx::ProgramContext;
BOOST_TEST_GLOBAL_FIXTURE(ProgramContext);
BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE);

BOOST_AUTO_TEST_CASE(nullary_test)
{
  auto a = Linx::Sequence<int, 3>("a").fill_with_offsets();
  auto b = Linx::exp(a);
  BOOST_TEST(b.label() == "exp(a)");
  BOOST_TEST(b.size() == a.size());
  a.exp();
  for (std::size_t i = 0; i < a.size(); ++i) {
    BOOST_TEST(b[i] == a[i]);
  }
}

BOOST_AUTO_TEST_CASE(unary_test)
{
  auto a = Linx::Sequence<int, 3>("a").fill_with_offsets();
  auto b = Linx::Sequence<int, 3>("b").fill_with_offsets();
  auto c = Linx::max(a, b);
  BOOST_TEST(c.label() == "max(a, b)");
  BOOST_TEST(c.size() == a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    BOOST_TEST(c[i] == std::max(a[i], b[i]));
  }
  a.max(b);
  for (std::size_t i = 0; i < a.size(); ++i) {
    BOOST_TEST(c[i] == a[i]);
  }
}

BOOST_AUTO_TEST_CASE(unary_scalar_test)
{
  auto a = Linx::Sequence<int, 3>("a").fill_with_offsets();
  auto b = Linx::pow(a, 2);
  BOOST_TEST(b.label() == "pow(a, 2)");
  BOOST_TEST(b.size() == a.size());
  a.pow(2);
  for (std::size_t i = 0; i < a.size(); ++i) {
    BOOST_TEST(b[i] == a[i]);
  }
}

BOOST_AUTO_TEST_SUITE_END();
