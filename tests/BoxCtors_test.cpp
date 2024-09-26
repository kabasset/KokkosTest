// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE BoxCtorsTest

#include "Linx/Data/Box.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

template <typename T, int N, typename TIn>
void check_ctor(const TIn& in, const Linx::Box<T, N>& expected)
{
  BOOST_TEST((std::is_same_v<typename TIn::size_type, T>));
  BOOST_TEST(TIn::Rank == N);
  BOOST_TEST((in == expected));
}

LINX_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(static_rank_test)
{
  check_ctor(Linx::Box(), Linx::Box<int, 0>());
  check_ctor(Linx::Box({-1}, {1}), Linx::Box<int, 1>({-1}, {1}));
  check_ctor(Linx::Box({-1, -2}, {1, 2}), Linx::Box<int, 2>({-1, -2}, {1, 2}));
  check_ctor(Linx::Box({-1, -2}, Linx::Shape({2, 4})), Linx::Box<int, 2>({-1, -2}, {1, 2}));
  check_ctor<int, 2>(Linx::Box({-1, -2}, {1, 2}), {{-1, -2}, {1, 2}});
}

BOOST_AUTO_TEST_SUITE_END()
