// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE "KokkosTest"

#include "KokkosContext.h"
#include "Raster.h"

#include <boost/test/unit_test.hpp>

BOOST_TEST_GLOBAL_FIXTURE(KokkosContext);

BOOST_AUTO_TEST_SUITE(KokkosTest);

BOOST_AUTO_TEST_CASE(iterate_test)
{
  const int width = 4;
  const int height = 3;
  const int n = width * height;
  using Raster = Linx::Raster<float, 2>;
  Raster a("a", {width, height});
  Raster b("b", {width, height});
  Raster c("c", width, height);

  a.iterate(KOKKOS_LAMBDA(int i, int j) {
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

  c.iterate(KOKKOS_LAMBDA(int i, int j) {
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

BOOST_AUTO_TEST_CASE(apply_test)
{
  const int width = 4;
  const int height = 3;
  const int n = width * height;
  using Raster = Linx::Raster<float, 2>;
  Raster a("a", width, height);
  Raster b("b", width, height);

  a.iterate(KOKKOS_LAMBDA(int i, int j) {
    a(i, j) = i + 2 * j;
    b(i, j) = 3;
  });

  a.apply(
      [](auto ai, auto bi) {
        return ai * ai + bi;
      },
      b);
  Kokkos::fence();

  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      BOOST_TEST(a(i, j) == i * i + 4 * i * j + 4 * j * j + 3);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();
