// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE SequenceCtorsTest

#include "Linx/Data/Box.h" // Position // FIXME move Position to Sequence?
#include "Linx/Data/Sequence.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

LINX_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

template <typename T, int N, typename TContainer>
void check_ctor(const Linx::Sequence<T, N, TContainer>& seq, const std::string& label, std::size_t size)
{
  BOOST_TEST(seq.label() == label);
  BOOST_TEST(seq.size() == size);
  BOOST_TEST(seq.ssize() == size);
  if (size == 0) {
    BOOST_TEST(seq.empty());
    BOOST_TEST((seq.end() == seq.begin()));
  } else {
    BOOST_TEST(not seq.empty());
    BOOST_TEST(seq.data());
    BOOST_TEST((seq.cdata() == seq.data()));
    BOOST_TEST((seq.end() != seq.begin()));
    for (auto e : Linx::on_host(seq)) {
      BOOST_TEST((e == T {1}));
    }
  }
}

LINX_TEST_CASE_TEMPLATE(static_empty_test)
{
  check_ctor(Linx::Sequence<T, 0>(), "", 0);
  check_ctor(Linx::Sequence<T, 0>("s"), "s", 0);
}

LINX_TEST_CASE_TEMPLATE(dynamic_empty_test)
{
  check_ctor(Linx::Sequence<T, -1>(), "", 0);
  check_ctor(Linx::Sequence<T, -1>("s"), "s", 0);
}

LINX_TEST_CASE_TEMPLATE(static_singleton_fill_test)
{
  check_ctor(Linx::Sequence<T, 1>().fill(1), "", 1);
  check_ctor(Linx::Sequence<T, 1>("s").fill(1), "s", 1);
}

LINX_QUICK_TEST_CASE_TEMPLATE(redundant_singleton_fill_test)
{
  check_ctor(Linx::Sequence<T, 1>(1).fill(1), "", 1);
  check_ctor(Linx::Sequence<T, 1>("s", 1).fill(1), "s", 1);
}

LINX_QUICK_TEST_CASE_TEMPLATE(static_singleton_list_test)
{
  check_ctor(Linx::Position({1}), "", 1);
  check_ctor(Linx::Sequence({1}), "", 1);
  check_ctor(Linx::Sequence<T, 1> {1}, "", 1);
  check_ctor(Linx::Sequence<T, 1>("s", {1}), "s", 1);
}

LINX_QUICK_TEST_CASE_TEMPLATE(static_singleton_one_test)
{
  check_ctor(Linx::Sequence<T, 1>(Linx::Constant(1)), "", 1);
  check_ctor(Linx::Sequence<T, 1>("s", Linx::Constant(1)), "s", 1);
}

LINX_TEST_CASE_TEMPLATE(dynamic_singleton_fill_test)
{
  check_ctor(Linx::Sequence<T, -1>(1).fill(1), "", 1);
  check_ctor(Linx::Sequence<T, -1>("s", 1).fill(1), "s", 1);
}

LINX_QUICK_TEST_CASE_TEMPLATE(dynamic_singleton_list_test)
{
  check_ctor(Linx::Sequence<T, -1> {1}, "", 1);
  check_ctor(Linx::Sequence<T, -1>("s", {1}), "s", 1);
}

LINX_QUICK_TEST_CASE_TEMPLATE(dynamic_singleton_one_test)
{
  check_ctor(Linx::Sequence<T, -1>(Linx::Constant(1)), "", 1);
  check_ctor(Linx::Sequence<T, -1>("s", Linx::Constant(1)), "s", 1);
}

LINX_TEST_CASE_TEMPLATE(static_multiple_fill_test)
{
  check_ctor(Linx::Sequence<T, 3>().fill(1), "", 3);
  check_ctor(Linx::Sequence<T, 3>("s").fill(1), "s", 3);
}

LINX_QUICK_TEST_CASE_TEMPLATE(redundant_multiple_fill_test)
{
  check_ctor(Linx::Sequence<T, 3>(3).fill(1), "", 3);
  check_ctor(Linx::Sequence<T, 3>("s", 3).fill(1), "s", 3);
}

LINX_QUICK_TEST_CASE_TEMPLATE(static_multiple_list_test)
{
  check_ctor(Linx::Position({1, 1, 1}), "", 3);
  check_ctor(Linx::Sequence({1, 1, 1}), "", 3);
  check_ctor(Linx::Sequence<T, 3> {1, 1, 1}, "", 3);
  check_ctor(Linx::Sequence<T, 3>("s", {1, 1, 1}), "s", 3);
}

LINX_TEST_CASE_TEMPLATE(dynamic_multiple_fill_test)
{
  check_ctor(Linx::Sequence<T, -1>(3).fill(1), "", 3);
  check_ctor(Linx::Sequence<T, -1>("s", 3).fill(1), "s", 3);
}

LINX_QUICK_TEST_CASE_TEMPLATE(dynamic_multiple_list_test)
{
  check_ctor(Linx::Sequence<T, -1> {1, 1, 1}, "", 3);
  check_ctor(Linx::Sequence<T, -1>("s", {1, 1, 1}), "s", 3);
}

BOOST_AUTO_TEST_SUITE_END()
