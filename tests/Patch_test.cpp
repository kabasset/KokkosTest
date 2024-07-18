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

BOOST_AUTO_TEST_CASE(unbounded_singleton_span_test)
{
  auto image = Linx::Image<float, 3>("image", 16, 9, 4);
  image.domain().iterate(
      "init",
      KOKKOS_LAMBDA(auto i, auto j, auto k) { image(i, j, k) = i + j + k; });
  auto patch = Linx::patch("patch", image, Linx::Slice(1, 5)()(3));
  BOOST_TEST(patch.Rank == 2);
  BOOST_TEST(patch.extent(0) == 4);
  BOOST_TEST(patch.extent(1) == 9);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 9; ++j) {
      BOOST_TEST(patch(i, j) == image(i + 1, j, 3));
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();
