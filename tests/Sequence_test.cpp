// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE SequenceTest

#include "Linx/Data/Sequence.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

using Linx::ProgramContext;
BOOST_TEST_GLOBAL_FIXTURE(ProgramContext);
BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE);

BOOST_AUTO_TEST_CASE(static_empty_test)
{
  Linx::Sequence<int, 0> seq;
  BOOST_TEST(seq.size() == 0);
  BOOST_TEST(seq.ssize() == 0);
  BOOST_TEST(seq.empty());
  BOOST_TEST((seq.end() == seq.begin()));
}

BOOST_AUTO_TEST_CASE(dynamic_empty_test)
{
  Linx::Sequence<int, -1> seq;
  BOOST_TEST(seq.size() == 0);
  BOOST_TEST(seq.ssize() == 0);
  BOOST_TEST(seq.empty());
  BOOST_TEST((seq.end() == seq.begin()));
}

BOOST_AUTO_TEST_CASE(static_singleton_test)
{
  Linx::Sequence<int, 1> seq;
  BOOST_TEST(seq.size() == 1);
  BOOST_TEST(seq.ssize() == 1);
  BOOST_TEST(not seq.empty());
  BOOST_TEST((seq.end() != seq.begin()));
  seq[0] = 1;
  BOOST_TEST(seq[0] == 1);
  std::fill(seq.begin(), seq.end(), 2);
  BOOST_TEST(seq[0] == 2);
  seq.fill(3);
  BOOST_TEST(seq[0] == 3);
}

BOOST_AUTO_TEST_CASE(dynamic_singleton_test)
{
  Linx::Sequence<int, -1> seq("", 1);
  BOOST_TEST(seq.size() == 1);
  BOOST_TEST(seq.ssize() == 1);
  BOOST_TEST(not seq.empty());
  BOOST_TEST((seq.end() != seq.begin()));
  seq[0] = 1;
  BOOST_TEST(seq[0] == 1);
  std::fill(seq.begin(), seq.end(), 2);
  BOOST_TEST(seq[0] == 2);
  seq.fill(3);
  BOOST_TEST(seq[0] == 3);
}

BOOST_AUTO_TEST_CASE(static_multiple_test)
{
  constexpr int size = 3;
  Linx::Sequence<int, size> seq;
  BOOST_TEST(seq.size() == size);
  BOOST_TEST(seq.ssize() == size);
  BOOST_TEST(not seq.empty());
  seq.fill(1);
  for (auto e : seq) {
    BOOST_TEST(e == 1);
  }
}

BOOST_AUTO_TEST_CASE(dynamic_multiple_test)
{
  constexpr int size = 3;
  Linx::Sequence<int, -1> seq("seq", size);
  BOOST_TEST(seq.size() == size);
  BOOST_TEST(seq.ssize() == size);
  BOOST_TEST(not seq.empty());
  seq.fill(1);
  for (auto e : seq) {
    BOOST_TEST(e == 1);
  }
}

BOOST_AUTO_TEST_CASE(list_test)
{
  Linx::Sequence<int, 4> seq {0, 1, 2, 3};
  for (int i = 0; i < 4; ++i) {
    BOOST_TEST(seq[i] == i);
  }
}

BOOST_AUTO_TEST_SUITE_END();
