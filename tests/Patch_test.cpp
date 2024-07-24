// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE PatchTest

#include "Linx/Data/Patch.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>
#include <sstream>

using Linx::ProgramContext;
BOOST_TEST_GLOBAL_FIXTURE(ProgramContext);
BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE);

BOOST_AUTO_TEST_CASE(span_test)
{
  int start = 3;
  int stop = 14;
  auto slice = Linx::Slice(start, stop);
  BOOST_TEST(slice.start() == start);
  BOOST_TEST(slice.stop() == stop);
  BOOST_TEST(slice.kokkos_slice().first == start);
  BOOST_TEST(slice.kokkos_slice().second == stop);

  auto str = (std::stringstream() << slice).str();
  BOOST_TEST(str == std::to_string(start) + ':' + std::to_string(stop));
}

BOOST_AUTO_TEST_CASE(span_unbounded_singleton_slice_test)
{
  auto image = Linx::Image<float, 3>("image", 16, 9, 4);
  image.domain().iterate(
      "init",
      KOKKOS_LAMBDA(auto i, auto j, auto k) { image(i, j, k) = i + j + k; });
  auto slice = Linx::slice(image, Linx::Slice(1, 5)()(3));
  BOOST_TEST(slice.Rank == 2);
  BOOST_TEST(slice.extent(0) == 4);
  BOOST_TEST(slice.extent(1) == 9);
  BOOST_TEST(slice.domain().start(0) == 0);
  BOOST_TEST(slice.domain().stop(0) == 4);
  BOOST_TEST(slice.domain().start(1) == 0);
  BOOST_TEST(slice.domain().stop(1) == 9);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 9; ++j) {
      BOOST_TEST(slice(i, j) == image(i + 1, j, 3));
    }
  }
}

BOOST_AUTO_TEST_CASE(patch_unbounded_singleton_patch_test)
{
  auto image = Linx::Image<float, 3>("image", 16, 9, 4);
  image.domain().iterate(
      "init",
      KOKKOS_LAMBDA(auto i, auto j, auto k) { image(i, j, k) = i + j + k; });
  auto patch = Linx::patch(image, Linx::Slice(1, 5)()(3));
  BOOST_TEST(patch.Rank == 3);
  BOOST_TEST(patch.extent(0) == 4);
  BOOST_TEST(patch.extent(1) == 9);
  BOOST_TEST(patch.extent(2) == 1);
  BOOST_TEST(patch.domain().start(0) == 1);
  BOOST_TEST(patch.domain().stop(0) == 5);
  BOOST_TEST(patch.domain().start(1) == 0);
  BOOST_TEST(patch.domain().stop(1) == 9);
  BOOST_TEST(patch.domain().start(2) == 3);
  BOOST_TEST(patch.domain().stop(2) == 4);
  for (int i = 1; i < 5; ++i) {
    for (int j = 0; j < 9; ++j) {
      BOOST_TEST(patch(i, j, 3) == image(i, j, 3));
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();
