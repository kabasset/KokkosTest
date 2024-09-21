// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE BoxApplyTest

#include "Linx/Base/Functional.h"
#include "Linx/Data/Box.h"
#include "Linx/Data/Image.h"
#include "Linx/Run/ProgramContext.h"

#include <Kokkos_Core.hpp>
#include <boost/test/unit_test.hpp>

LINX_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(count_test)
{
  std::vector<int> f {0, 0};
  std::vector<int> b {3, 4};
  Linx::Box<int, 2> box(f, b);

  int count = 1;
  kokkos_reduce("count", box, Linx::Constant(1), Kokkos::Sum<int>(count));

  BOOST_TEST(count == box.size());
}

struct NegateFirstIndex {
  KOKKOS_INLINE_FUNCTION auto operator()(auto i0, auto...) const
  {
    return -i0;
  }
};

BOOST_AUTO_TEST_CASE(reduce_test)
{
  std::vector<int> f {0, 0};
  std::vector<int> b {3, 4};
  Linx::Box<int, 2> box(f, b);

  int sum = 1;
  kokkos_reduce("sum", box, NegateFirstIndex(), Kokkos::Sum<int>(sum));

  BOOST_TEST(sum == -12);
}

BOOST_AUTO_TEST_CASE(dyn_rank_1_test)
{
  Linx::Box<int, -1> box({-1}, {1}); // FIXME negative index not supported (yet supported by NDRangePolicy)
  Linx::for_each<Kokkos::Serial>("test", box, [](int i) {
    BOOST_TEST(i >= -1);
    BOOST_TEST(i < 1);
  });
}

BOOST_AUTO_TEST_CASE(dyn_rank_2_test)
{
  Linx::Box<int, -1> box({-1, -2}, {1, 2});
  Linx::for_each<Kokkos::Serial>("test", box, [](int i, int j) {
    BOOST_TEST(i >= -1);
    BOOST_TEST(i < 1);
    BOOST_TEST(j >= -2);
    BOOST_TEST(j < 2);
  });
}

BOOST_AUTO_TEST_CASE(dyn_rank_6_test)
{
  Linx::Box<int, -1> box({-1, -2, -3, -4, -5, -6}, {1, 2, 3, 4, 5, 6});
  Linx::for_each<Kokkos::Serial>("test", box, [](int i, int j, int k, int l, int m, int n) {
    BOOST_TEST(i >= -1);
    BOOST_TEST(i < 1);
    BOOST_TEST(j >= -2);
    BOOST_TEST(j < 2);
    BOOST_TEST(k >= -3);
    BOOST_TEST(k < 3);
    BOOST_TEST(l >= -4);
    BOOST_TEST(l < 4);
    BOOST_TEST(m >= -5);
    BOOST_TEST(m < 5);
    BOOST_TEST(n >= -6);
    BOOST_TEST(n < 6);
  });
}

BOOST_AUTO_TEST_SUITE_END()
