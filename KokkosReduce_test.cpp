// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE "KokkosTest"

#include "KokkosContext.h"

#include <boost/test/unit_test.hpp>

BOOST_TEST_GLOBAL_FIXTURE(KokkosContext);

BOOST_AUTO_TEST_SUITE(KokkosTest);

int square_sum(int n)
{
  int out = 0;
  for (int i = 0; i < n; ++i) {
    out += i * i;
  }
  return out;
}

struct SquareSum {
  using value_type = int;

  KOKKOS_INLINE_FUNCTION
  void operator()(int i, int& out) const
  {
    out += i * i;
  }
};

BOOST_AUTO_TEST_CASE(lambda_reduce_test)
{
  const int n = 10;
  int sum = 0;

  Kokkos::parallel_reduce(
      n,
      KOKKOS_LAMBDA(int i, int& out) { out += i * i; },
      sum);

  BOOST_TEST(sum == square_sum(n));
}

BOOST_AUTO_TEST_CASE(functor_reduce_test)
{
  const int n = 10;
  int sum = 0;

  Kokkos::parallel_reduce(n, SquareSum(), sum);

  BOOST_TEST(sum == square_sum(n));
}

BOOST_AUTO_TEST_SUITE_END();
