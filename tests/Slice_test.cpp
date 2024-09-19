// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE SliceTest

#include "Linx/Data/Box.h"
#include "Linx/Data/Slice.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>
#include <sstream>

LINX_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(unbounded_test)
{
  auto slice = Linx::Slice();
  BOOST_TEST((slice.kokkos_slice() == Kokkos::ALL));

  auto str = (std::stringstream() << slice).str();
  BOOST_TEST(str == ":");
}

BOOST_AUTO_TEST_CASE(singleton_test)
{
  int index = 10;
  auto slice = Linx::Slice(index);
  BOOST_TEST(slice.value() == index);
  BOOST_TEST(slice.kokkos_slice() == index);

  auto str = (std::stringstream() << slice).str();
  BOOST_TEST(str == std::to_string(index));
}

BOOST_AUTO_TEST_CASE(span_test)
{
  int start = 3;
  int stop = 14;
  int size = stop - start;
  auto slice = Linx::Slice(start, stop);
  BOOST_TEST(slice.start() == start);
  BOOST_TEST(slice.stop() == stop);
  BOOST_TEST(slice.size() == size);
  BOOST_TEST(slice.kokkos_slice().first == start);
  BOOST_TEST(slice.kokkos_slice().second == stop);

  auto str = (std::stringstream() << slice).str();
  BOOST_TEST(str == std::to_string(start) + ':' + std::to_string(stop));
}

BOOST_AUTO_TEST_CASE(span_from_size_test)
{
  int start = 3;
  int stop = 14;
  int size = stop - start;
  auto slice = Linx::Slice(start, Linx::Plus(size));
  BOOST_TEST(slice.start() == start);
  BOOST_TEST(slice.stop() == stop);
  BOOST_TEST(slice.size() == size);
}

BOOST_AUTO_TEST_CASE(unbounded_singleton_span_test)
{
  int index = 10;
  int start = 3;
  int stop = 14;
  auto slice = Linx::Slice()(index)(start, stop);
  BOOST_TEST(slice.Rank == 3);
  BOOST_TEST(int(slice.template get<0>().Type) == int(Linx::SliceType::Unbounded));
  BOOST_TEST(int(slice.template get<1>().Type) == int(Linx::SliceType::Singleton));
  BOOST_TEST(int(slice.template get<2>().Type) == int(Linx::SliceType::RightOpen));

  auto str = (std::stringstream() << slice).str();
  BOOST_TEST(str == ":, " + std::to_string(index) + ", " + std::to_string(start) + ':' + std::to_string(stop));
}

BOOST_AUTO_TEST_CASE(span_singleton_unbounded_test)
{
  int index = 10;
  int start = 3;
  int stop = 14;
  auto slice = Linx::Slice(start, stop)(index)();
  BOOST_TEST(slice.Rank == 3);
  BOOST_TEST(char(slice.template get<0>().Type) == char(Linx::SliceType::RightOpen));
  BOOST_TEST(char(slice.template get<2>().Type) == char(Linx::SliceType::Unbounded));
  BOOST_TEST(char(slice.template get<1>().Type) == char(Linx::SliceType::Singleton));

  auto str = (std::stringstream() << slice).str();
  BOOST_TEST(str == std::to_string(start) + ':' + std::to_string(stop) + ", " + std::to_string(index) + ", :");
}

BOOST_AUTO_TEST_CASE(box_test)
{
  int index = 10;
  int start = 3;
  int stop = 14;
  auto slice = Linx::Slice(index)(start, stop);
  auto box = Linx::box(slice);
  BOOST_TEST(box.start(0) == index);
  BOOST_TEST(box.start(1) == start);
  BOOST_TEST(box.stop(0) == index + 1);
  BOOST_TEST(box.stop(1) == stop);
}

BOOST_AUTO_TEST_CASE(clamp_test)
{
  auto slice = Linx::Slice(10)()(3, 14);
  Linx::Box<int, 4> box({1, 2, 3, 4}, {11, 12, 13, 14});
  auto clamped = Linx::box(slice & box);

  BOOST_TEST(clamped.Rank == 3);
  BOOST_TEST(clamped.start(0) == 10);
  BOOST_TEST(clamped.start(1) == 2);
  BOOST_TEST(clamped.start(2) == 3);
  BOOST_TEST(clamped.stop(0) == 11);
  BOOST_TEST(clamped.stop(1) == 12);
  BOOST_TEST(clamped.stop(2) == 13);
}

BOOST_AUTO_TEST_SUITE_END()
