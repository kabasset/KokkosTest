// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE BoxCtorsTest

#include "Linx/Data/Box.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

template <typename T, Linx::Index N, typename TIn>
void check_ctor(const TIn& in, const Linx::GBox<T, N>& expected)
{
  BOOST_TEST((std::is_same_v<typename TIn::size_type, T>));
  BOOST_TEST(TIn::Rank == N);
  BOOST_TEST((in == expected));
}

LINX_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(static_rank_test)
{
  Linx::Position shape({2, 4});
  BOOST_TEST(Linx::Shape(shape).size() == 8);
  check_ctor(Linx::Box(), Linx::GBox<Linx::Index, 0>());
  check_ctor(Linx::Box({-1}, {1}), Linx::GBox<Linx::Index, 1>({-1}, {1}));
  check_ctor(Linx::Box({-1, -2}, {1, 2}), Linx::GBox<Linx::Index, 2>({-1, -2}, {1, 2}));
  check_ctor(Linx::Box({-1, -2}, Linx::Shape({2, 4})), Linx::GBox<Linx::Index, 2>({-1, -2}, {1, 2}));
  check_ctor(Linx::Box({-1, -2}, Linx::Shape(shape)), Linx::GBox<Linx::Index, 2>({-1, -2}, {1, 2}));
  check_ctor(Linx::Box({-1, -2}, {1, 2}), Linx::GBox({-1, -2}, {1, 2}));
}

BOOST_AUTO_TEST_SUITE_END()
