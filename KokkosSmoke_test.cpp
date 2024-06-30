// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE "KokkosTest"

#include <Kokkos_Core.hpp>
#include <boost/test/unit_test.hpp>

using Kokkos::ScopeGuard;
BOOST_TEST_GLOBAL_FIXTURE(ScopeGuard);

BOOST_AUTO_TEST_SUITE(KokkosSmokeTest);

BOOST_AUTO_TEST_CASE(for_test)
{
  const int width = 4;
  const int height = 3;
  const int n = width * height;
  using View = Kokkos::View<int**>;
  View a("a", width, height);
  View b("b", width, height);
  View c("c", width, height);

  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {width, height}),
      KOKKOS_LAMBDA(int i, int j) {
        a(i, j) = i + j;
        b(i, j) = 2 * i + 3 * j;
      });
  Kokkos::fence();

  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      BOOST_TEST(a(i, j) == i + j);
      BOOST_TEST(b(i, j) == 2 * i + 3 * j);
    }
  }

  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {width, height}),
      KOKKOS_LAMBDA(int i, int j) {
        const auto aij = a(i, j);
        const auto bij = b(i, j);
        c(i, j) = aij * aij + bij * bij;
      });
  Kokkos::fence();

  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      BOOST_TEST(c(i, j) == 5 * i * i + 14 * i * j + 10 * j * j);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();
