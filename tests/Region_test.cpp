// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE RegionTest

#include "Linx/Data/Box.h"
#include "Linx/Data/concepts/Region.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

using Linx::ProgramContext;
BOOST_TEST_GLOBAL_FIXTURE(ProgramContext);
BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE);

template <Linx::Region T>
void check_region_size(const T& region, typename T::size_type expected)
{
  using Size = typename T::size_type;
  BOOST_TEST(std::size(region) == expected);
  Size count;
  kokkos_reduce(
      "count",
      region,
      KOKKOS_LAMBDA(auto...) { return Size(1); },
      Kokkos::Sum<Size>(count));
  BOOST_TEST(count == expected);
}

template <Linx::Window T>
void check_window_size(const T& region, typename T::size_type expected)
{
  check_region_size(region, expected);
}

BOOST_AUTO_TEST_CASE(box_test)
{
  Linx::Box<int, 2> region({1, 2}, {11, 22});
  check_window_size(region, 200);
}

BOOST_AUTO_TEST_SUITE_END();
