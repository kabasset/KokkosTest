// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE VectorTest

#include "Linx/Data/Vector.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

using Linx::ProgramContext;
BOOST_TEST_GLOBAL_FIXTURE(ProgramContext);
BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE);

BOOST_AUTO_TEST_CASE(static_empty_test)
{
  Linx::Vector<int, 0> vec;
  BOOST_TEST(vec.size() == 0);
  BOOST_TEST(vec.ssize() == 0);
  BOOST_TEST(vec.empty());
  BOOST_TEST(vec.end() == vec.begin());
}

BOOST_AUTO_TEST_CASE(dynamic_empty_test)
{
  Linx::Vector<int, -1> vec;
  BOOST_TEST(vec.size() == 0);
  BOOST_TEST(vec.ssize() == 0);
  BOOST_TEST(vec.empty());
  BOOST_TEST(vec.end() == vec.begin());
}

BOOST_AUTO_TEST_CASE(static_singleton_test)
{
  Linx::Vector<int, 1> vec;
  BOOST_TEST(vec.size() == 1);
  BOOST_TEST(vec.ssize() == 1);
  BOOST_TEST(not vec.empty());
  BOOST_TEST(vec.end() != vec.begin());
  vec[0] = 1;
  BOOST_TEST(vec[0] == 1);
  std::fill(vec.begin(), vec.end(), 2);
  BOOST_TEST(vec[0] == 2);
  vec = 3;
  BOOST_TEST(vec[0] == 3);
}

BOOST_AUTO_TEST_CASE(dynamic_singleton_test)
{
  Linx::Vector<int, -1> vec("", 1);
  BOOST_TEST(vec.size() == 1);
  BOOST_TEST(vec.ssize() == 1);
  BOOST_TEST(not vec.empty());
  BOOST_TEST(vec.end() != vec.begin());
  vec[0] = 1;
  BOOST_TEST(vec[0] == 1);
  std::fill(vec.begin(), vec.end(), 2);
  BOOST_TEST(vec[0] == 2);
  vec = 3;
  BOOST_TEST(vec[0] == 3);
}

BOOST_AUTO_TEST_CASE(static_multiple_test)
{
  constexpr int size = 3;
  Linx::Vector<int, size> vec;
  BOOST_TEST(vec.size() == size);
  BOOST_TEST(vec.ssize() == size);
  BOOST_TEST(not vec.empty());
  vec = 1;
  for (auto e : vec) {
    BOOST_TEST(e == 1);
  }
}

BOOST_AUTO_TEST_CASE(dynamic_multiple_test)
{
  constexpr int size = 3;
  Linx::Vector<int, -1> vec(size);
  BOOST_TEST(vec.size() == size);
  BOOST_TEST(vec.ssize() == size);
  BOOST_TEST(not vec.empty());
  vec = 1;
  for (auto e : vec) {
    BOOST_TEST(e == 1);
  }
}

BOOST_AUTO_TEST_CASE(list_test)
{
  Linx::Vector<int, 4> vec {0, 1, 2, 3};
  for (int i = 0; i < 4; ++i) {
    BOOST_TEST(vec[i] == i);
  }
}

BOOST_AUTO_TEST_SUITE_END();
