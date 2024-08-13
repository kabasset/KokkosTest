// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE RegionTest

#include "Linx/Base/Reduction.h"
#include "Linx/Data/Box.h"
#include "Linx/Data/concepts/Region.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

using Linx::ProgramContext;
BOOST_TEST_GLOBAL_FIXTURE(ProgramContext);
BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE);

template <typename T>
class Constant {
public:

  using value_type = T;
  
  KOKKOS_INLINE_FUNCTION Constant(T value) : m_value(value) {}

  KOKKOS_INLINE_FUNCTION const T& operator()(auto...) const
  {
    return m_value;
  }

private:

  T m_value;
};

template <typename T> // FIXME Region
void check_region_size(const T& region, typename T::size_type expected)
{
  using Size = typename T::size_type;
  BOOST_TEST(std::size(region) == expected);
  Size count;
  kokkos_reduce(
      "count",
      region,
      Constant<Size>(1),
      Kokkos::Sum<Size>(count));
  BOOST_TEST(count == expected);
  
  using ProjectionReducer = Linx::Internal::ProjectionReducer<Size, Constant<Size>, Kokkos::Sum<Size>, 0, 1>;
  Kokkos::parallel_reduce(
      "count",
      kokkos_execution_policy(region),
      ProjectionReducer(Constant<Size>(1), sum),
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
  Linx::Box<int, 2> region({1, 2}, {11, 22});
  check_window_size(region, 200);
}

BOOST_AUTO_TEST_SUITE_END();
