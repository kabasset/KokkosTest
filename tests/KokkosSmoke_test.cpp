// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE KokkosSmokeTest

#include <Kokkos_Core.hpp>
#include <Kokkos_DynRankView.hpp>
#include <boost/test/unit_test.hpp>

using Kokkos::ScopeGuard;
BOOST_TEST_GLOBAL_FIXTURE(ScopeGuard);
BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE);

namespace Linx {

auto on_host(const auto& in) // FIXME to Linx
{
  auto out = Kokkos::create_mirror_view(in);
  Kokkos::deep_copy(out, in);
  return out;
}

} // namespace Linx

BOOST_AUTO_TEST_CASE(view_for_test)
{
  const int width = 4;
  const int height = 3;
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

  auto a_on_host = Linx::on_host(a);
  auto b_on_host = Linx::on_host(b);
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      BOOST_TEST(a_on_host(i, j) == i + j);
      BOOST_TEST(b_on_host(i, j) == 2 * i + 3 * j);
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

  auto c_on_host = Linx::on_host(c);
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      BOOST_TEST(c_on_host(i, j) == 5 * i * i + 14 * i * j + 10 * j * j);
    }
  }
}

BOOST_AUTO_TEST_CASE(dynrankview_rank_test)
{
  const int width = 4;
  const int height = 3;
  using View = Kokkos::DynRankView<int>;
  View empty("0D", KOKKOS_INVALID_INDEX);
  BOOST_TEST(empty.rank() == 0);
  View sequence("1D", width, KOKKOS_INVALID_INDEX);
  BOOST_TEST(sequence.rank() == 1);
  BOOST_TEST(sequence.extent(0) == width);
  View image("2D", width, height, KOKKOS_INVALID_INDEX);
  BOOST_TEST(image.rank() == 2);
  BOOST_TEST(image.extent(0) == width);
  BOOST_TEST(image.extent(1) == height);
}

BOOST_AUTO_TEST_SUITE_END()
