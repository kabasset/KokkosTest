// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE BoxApplyTest

#include "Linx/Data/Image.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

using Linx::ProgramContext;
BOOST_TEST_GLOBAL_FIXTURE(ProgramContext);
BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE);

BOOST_AUTO_TEST_CASE(count_test)
{
  std::vector<int> f {0, 0};
  std::vector<int> b {3, 4};
  Linx::Box<int, 2> box(f, b);

  int count = 1;
  kokkos_reduce(
      "count",
      box,
      [](int, int) {
        return 1;
      },
      Kokkos::Sum<int>(count));

  BOOST_TEST(count == box.size());
}

BOOST_AUTO_TEST_CASE(reduce_test)
{
  std::vector<int> f {0, 0};
  std::vector<int> b {3, 4};
  Linx::Box<int, 2> box(f, b);

  int sum = 1;
  kokkos_reduce(
      "sum",
      box,
      [](int i, int) {
        return -i;
      },
      Kokkos::Sum<int>(sum));

  BOOST_TEST(sum == -12);
}

BOOST_AUTO_TEST_SUITE_END();
