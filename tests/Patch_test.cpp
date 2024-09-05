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
  for_each(
      "init",
      image.domain(),
      KOKKOS_LAMBDA(int i, int j, int k) { image(i, j, k) = i + j + k; });
  auto slice = Linx::slice(image, Linx::Slice(1, 5)()(3));
  BOOST_TEST(slice.label() == image.label());
  BOOST_TEST(slice.Rank == 2);
  BOOST_TEST(slice.extent(0) == 4);
  BOOST_TEST(slice.extent(1) == 9);
  BOOST_TEST(slice.domain().start(0) == 0);
  BOOST_TEST(slice.domain().stop(0) == 4);
  BOOST_TEST(slice.domain().start(1) == 0);
  BOOST_TEST(slice.domain().stop(1) == 9);
  
  const Linx::Box<int, 2> box({0, 0}, {4, 9});
  Linx::Image<int, 2> diff("diff", box.stop());
  Linx::for_each(
      "test",
      Linx::Box<int, 2>({0, 0}, {4, 9}),
      KOKKOS_LAMBDA(int i, int j) {
        diff(i, j) = slice(i, j) - image(i + 1, j, 3);
      });
  BOOST_TEST(Linx::norm<0>(diff) == 0);
}

BOOST_AUTO_TEST_CASE(patch_unbounded_singleton_patch_test)
{
  auto image = Linx::Image<float, 3>("image", 16, 9, 4);
  for_each(
      "init",
      image.domain(),
      KOKKOS_LAMBDA(int i, int j, int k) { image(i, j, k) = i + j + k; });
  auto patch = Linx::patch(image, Linx::Slice(1, 5)()(3));
  BOOST_TEST((Linx::root(patch) == image));
  BOOST_TEST((Linx::root(patch).container() == image.container()));
  BOOST_TEST(patch.Rank == 3);
  const auto& domain = patch.domain();
  BOOST_TEST(domain.extent(0) == 4);
  BOOST_TEST(domain.extent(1) == 9);
  BOOST_TEST(domain.extent(2) == 1);
  BOOST_TEST(domain.start(0) == 1);
  BOOST_TEST(domain.stop(0) == 5);
  BOOST_TEST(domain.start(1) == 0);
  BOOST_TEST(domain.stop(1) == 9);
  BOOST_TEST(domain.start(2) == 3);
  BOOST_TEST(domain.stop(2) == 4);
  
  const Linx::Box<int, 2> box({1, 0}, {5, 9});
  Linx::Image<int, 2> diff("diff", box.shape());
  Linx::for_each(
      "test",
      box,
      KOKKOS_LAMBDA(int i, int j) {
        diff(i - box.start(0), j - box.start(1)) = patch(i, j, 3) - image(i, j, 3);
      });
  BOOST_TEST(Linx::norm<0>(diff) == 0);
}

BOOST_AUTO_TEST_CASE(patch_of_patch_test)
{
  auto image = Linx::Image<int, 2>("image", 10, 8);

  auto box_a = Linx::Box<int, 2>({1, -1}, {11, 7});
  auto patch_a = Linx::patch(image, box_a);
  const auto& domain_a = patch_a.domain();
  BOOST_TEST(domain_a.start(0) == 1);
  BOOST_TEST(domain_a.start(1) == 0);
  BOOST_TEST(domain_a.stop(0) == 10);
  BOOST_TEST(domain_a.stop(1) == 7);

  auto box_b = Linx::Box<int, 2>({-1, 1}, {9, 10});
  auto patch_b = Linx::patch(patch_a, box_b);
  const auto& domain_b = patch_b.domain();
  BOOST_TEST(domain_b.start(0) == 1);
  BOOST_TEST(domain_b.start(1) == 1);
  BOOST_TEST(domain_b.stop(0) == 9);
  BOOST_TEST(domain_b.stop(1) == 7);
}

BOOST_AUTO_TEST_SUITE_END();
