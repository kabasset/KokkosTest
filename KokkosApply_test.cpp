// Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE "KokkosTest"

#include "KokkosContext.h"

#include <boost/test/unit_test.hpp>

BOOST_TEST_GLOBAL_FIXTURE(KokkosContext);

BOOST_AUTO_TEST_SUITE(KokkosTest);

BOOST_AUTO_TEST_CASE(lambda_reduce_test) {
  const int width = 40;
  const int height = 30;
  const int n = width * height;
  using View = Kokkos::View<float**>;
  View a("a", width, height);
  View b("b", width, height);
  View c("c", width, height);

  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {width, height}),
      KOKKOS_LAMBDA(int i, int j) {
        a(i, j) = i + j;
        b(i, j) = 2 * i + 3 * j;
      });

  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      BOOST_TEST(a(i, j) == i + j);
      BOOST_TEST(b(i, j) == 2 * i + 3 * j);
    }
  }

  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {width, height}),
      KOKKOS_LAMBDA(int i, int j) {
        std::cout << i << ", " << j << std::endl; // FIXME rm
        const auto aij = a(i, j);
        const auto bij = b(i, j);
        c(i, j) = aij * aij + bij * bij;
      });

  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      BOOST_TEST(c(i, j) == 5 * i * i + 14 * i * j + 10 * j * j);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();
