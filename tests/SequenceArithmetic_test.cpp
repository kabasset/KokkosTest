// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE SequenceArithmeticTest

#include "Linx/Data/Sequence.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

LINX_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(nullary_test)
{
  auto a = Linx::Sequence<int, 3>("a").fill_with_offsets();
  auto b = Linx::exp(a);
  BOOST_TEST(b.label() == "exp(a)");
  BOOST_TEST(b.size() == a.size());
  a.exp();
  auto a_on_host = Linx::on_host(a);
  auto b_on_host = Linx::on_host(b);
  for (std::size_t i = 0; i < a.size(); ++i) {
    BOOST_TEST(b_on_host[i] == a_on_host[i]);
  }
}

BOOST_AUTO_TEST_CASE(unary_test)
{
  auto a = Linx::Sequence<int, 3>("a").fill_with_offsets();
  auto b = Linx::Sequence<int, 3>("b").fill_with_offsets();
  auto c = Linx::max(a, b);
  BOOST_TEST(c.label() == "max(a, b)");
  BOOST_TEST(c.size() == a.size());
  auto a_on_host = Linx::on_host(a);
  auto b_on_host = Linx::on_host(b);
  auto c_on_host = Linx::on_host(c);
  for (std::size_t i = 0; i < a.size(); ++i) {
    BOOST_TEST(c_on_host[i] == std::max(a_on_host[i], b_on_host[i]));
  }
  a.max(b);
  auto a_max_on_host = Linx::on_host(a);
  for (std::size_t i = 0; i < a.size(); ++i) {
    BOOST_TEST(c_on_host[i] == a_max_on_host[i]);
  }
}

BOOST_AUTO_TEST_CASE(unary_scalar_test)
{
  auto a = Linx::Sequence<int, 3>("a").fill_with_offsets();
  auto b = Linx::pow(a, 2);
  BOOST_TEST(b.label() == "pow(a, 2)");
  BOOST_TEST(b.size() == a.size());
  a.pow(2);
  auto a_on_host = Linx::on_host(a);
  auto b_on_host = Linx::on_host(b);
  for (std::size_t i = 0; i < a.size(); ++i) {
    BOOST_TEST(b_on_host[i] == a_on_host[i]);
  }
}

BOOST_AUTO_TEST_SUITE_END()
