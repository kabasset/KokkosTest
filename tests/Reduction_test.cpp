// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE ReductionTest

#include "Linx/Base/Reduction.h"
#include "Linx/Data/Image.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

using Linx::ProgramContext;
BOOST_TEST_GLOBAL_FIXTURE(ProgramContext);
BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE);

BOOST_AUTO_TEST_CASE(sum_test)
{
  const int width = 4;
  const int height = 3;
  Linx::Image<int, 2> a("a", width, height);

  Linx::for_each(
      "range",
      a.domain(),
      KOKKOS_LAMBDA(int i, int j) { a(i, j) = i + j * width; });

  auto sum = Linx::sum(a);

  BOOST_TEST(sum == a.size() * (a.size() - 1) / 2);
}

void test_norm(const auto& in)
{
  in.fill_with_offsets();

  std::vector<int> expected(3); // FIXME map_reduce to view

  expected[0] = Linx::map_reduce(
      "norm0",
      std::plus {},
      0,
      KOKKOS_LAMBDA(auto e) { return e != 0; },
      in);

  expected[1] = Linx::map_reduce(
      "norm1",
      std::plus {},
      0,
      KOKKOS_LAMBDA(auto e) { return std::abs(e); },
      in);

  expected[2] = Linx::map_reduce(
      "norm2",
      std::plus {},
      0,
      KOKKOS_LAMBDA(auto e) { return e * e; },
      in);

  BOOST_TEST(Linx::distance<0>(in, in) == 0);
  BOOST_TEST(Linx::distance<1>(in, in) == 0);
  BOOST_TEST(Linx::distance<2>(in, in) == 0);
  BOOST_TEST(Linx::norm<0>(in) == expected[0]);
  BOOST_TEST(Linx::norm<1>(in) == expected[1]);
  BOOST_TEST(Linx::norm<2>(in) == expected[2]);
  BOOST_TEST(Linx::dot(in, in) == expected[2]);

  auto zero = (+in).fill(0);

  BOOST_TEST(Linx::distance<0>(zero, in) == expected[0]);
  BOOST_TEST(Linx::distance<1>(zero, in) == expected[1]);
  BOOST_TEST(Linx::distance<2>(zero, in) == expected[2]);
}

BOOST_AUTO_TEST_CASE(norm_1d_test)
{
  test_norm(Linx::Image<int, 1>("a", 4));
}

BOOST_AUTO_TEST_CASE(norm_2d_test)
{
  test_norm(Linx::Image<int, 2>("a", 3, 2));
}

BOOST_AUTO_TEST_CASE(norm_3d_test)
{
  test_norm(Linx::Image<int, 3>("a", 3, 2, 4));
}

// FIXME Requires Sequence.domain()
// BOOST_AUTO_TEST_CASE(norm_seq_test)
// {
//   test_norm(Linx::Sequence<int, 4>("a"));
// }

BOOST_AUTO_TEST_SUITE_END();
