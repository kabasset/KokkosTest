// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE RegionTest

#include "Linx/Base/Reduction.h"
#include "Linx/Data/Box.h"
#include "Linx/Data/concepts/Region.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

LINX_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

template <typename T> // FIXME Region
void check_region_size(const T& region, typename T::size_type expected)
{
  using Size = typename T::size_type;
  BOOST_TEST(std::size(region) == expected);
  Size count;
  kokkos_reduce("count", region, Linx::Constant<Size>(1), Kokkos::Sum<Size>(count));
  BOOST_TEST(count == expected);

  Kokkos::Sum<Size> sum(count);
  using ProjectionReducer = Linx::Impl::ProjectionReducer<Size, Linx::Constant<Size>, Kokkos::Sum<Size>, 0, 1>;
  Kokkos::parallel_reduce(
      "count",
      Linx::kokkos_execution_policy<Kokkos::DefaultExecutionSpace>(region),
      ProjectionReducer(Linx::Constant<Size>(1), sum),
      sum);
  BOOST_TEST(count == expected);
}

template <typename T> // FIXME Window
void check_window_size(const T& region, typename T::size_type expected)
{
  check_region_size(region, expected);
}

BOOST_AUTO_TEST_CASE(box_test)
{
  check_window_size(Linx::Box({1, 2}, {11, 22}), 200);
}

BOOST_AUTO_TEST_SUITE_END()
