// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE "KokkosTest"

#include "KokkosContext.h"

#include <boost/test/unit_test.hpp>

BOOST_TEST_GLOBAL_FIXTURE(KokkosContext);

BOOST_AUTO_TEST_SUITE(KokkosTest);

BOOST_AUTO_TEST_CASE(for_test)
{
  const int width = 4;
  const int height = 3;
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
        const auto aij = a(i, j);
        const auto bij = b(i, j);
        c(i, j) = aij * aij + bij * bij;
      });
  Kokkos::fence(); // Needed for benchmarking

  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      BOOST_TEST(c(i, j) == 5 * i * i + 14 * i * j + 10 * j * j);
    }
  }
}

namespace Linx {

template <typename TView>
auto range_policy(TView view)
{
  constexpr auto N = TView::rank();
  Kokkos::Array<std::int64_t, N> begin;
  Kokkos::Array<std::int64_t, N> end;
  for (std::size_t i = 0; i < N && i < 8; ++i) {
    begin[i] = 0;
    end[i] = view.extent(i);
  }
  return Kokkos::MDRangePolicy<Kokkos::Rank<N>>(begin, end);
}

} // namespace Linx

BOOST_AUTO_TEST_CASE(linx_for_test)
{
  const int width = 4;
  const int height = 3;
  using View = Kokkos::View<float**>;
  View a("a", width, height);
  View b("b", width, height);
  View c("c", width, height);

  Kokkos::parallel_for(
      Linx::range_policy(a),
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
      Linx::range_policy(c),
      KOKKOS_LAMBDA(int i, int j) {
        const auto aij = a(i, j);
        const auto bij = b(i, j);
        c(i, j) = aij * aij + bij * bij;
      });
  Kokkos::fence(); // Needed for benchmarking

  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      BOOST_TEST(c(i, j) == 5 * i * i + 14 * i * j + 10 * j * j);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();
