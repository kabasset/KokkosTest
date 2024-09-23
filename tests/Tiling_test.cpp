// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE TilingTest

#include "Linx/Data/Image.h"
#include "Linx/Data/Tiling.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

namespace Linx {

template <int I, typename TIn>
class ProfileWise {
public:
  static constexpr int Axis = I;
  static constexpr int Rank = TIn::Rank;
  using Domain = Line<int, I, Rank>;
  
  ProfileWise(const TIn& in) :
      m_in(in),
      m_starts(m_in.domain()),
      m_stop(m_domain.stop(I))
  {
    m_starts.stop(I) = m_starts.start(I) + 1;
  }
  
  template <typename TFunc>
  void apply(TFunc func)
  {
    auto profile = patch(in, begin(+start, stop)); // FIXME Deep copy in regions
    Linx::for_each<typename TIn::execution_space>(
        "profiles",
        m_starts, // FIXME handle potential offset
        KOKKOS_LAMBDA(auto... is) {
          profile.shift(is...);
          func(profile);
          profile.ishift(is...);
        });
  }

private:
  TIn m_in;
  Box<int, Rank> m_starts;
  int m_stop;
};

}

LINX_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(rows_test)
{
  auto image = Linx::Image<int, 3>("image", 16, 9, 4);
  Linx::for_each(
      "fill",
      image.domain(),
      KOKKOS_LAMBDA(int i, int j, int k) { image(i, j, k) = i; });
  Linx::Sequence<int, 16> sum;
  Linx::rowwise(image).apply(
      "sum",
      KOKKOS_LAMBDA(auto row)
      {
        for (int i = 0; i < row.size(); ++i) {
          sum[i] += row.local(i);
        }
      });
  for (std::size_t i = 0; i < sum.size(); ++i) {
    BOOST_TEST(Linx::on_host(sum)[i] == 9 * 4 * i);
  }
}
/*
BOOST_AUTO_TEST_CASE(profiles_test)
{
  auto image = Linx::Image<int, 3>("image", 16, 9, 4);
  Linx::for_each<Kokkos::DefaultHostExecutionSpace>(
      "fill",
      image.domain(),
      KOKKOS_LAMBDA(auto i, auto j, auto k) { image(i, j, k) = j; });
  Linx::Position<int, 9> sum;
  for (const auto& column : Linx::profiles<1>(image)) {
    BOOST_TEST(column.size() == image.extent(1));
    BOOST_TEST(column.size() == sum.size());
    for (int i = 0; i < column.size(); ++i) {
      sum[i] += column.local(i);
    }
  }
  for (std::size_t i = 0; i < sum.size(); ++i) {
    BOOST_TEST(sum[i] == 16 * 4 * i);
  }
}
*/

BOOST_AUTO_TEST_SUITE_END()
