// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_TILING_H
#define _LINXDATA_TILING_H

#include "Linx/Data/Image.h"
#include "Linx/Data/Patch.h"
#include "Linx/Data/Raster.h"
#include "Linx/Data/Sequence.h"

#include <Kokkos_Core.hpp>

namespace Linx {

template <typename T, int I, int N>
class Line {
public:

  Line(Position<T, N> start, T stop) : m_start(LINX_MOVE(start)), m_step(1), m_stop(stop) {}

  KOKKOS_INLINE_FUNCTION std::size_t size() const
  {
    return m_stop - m_start[I];
  }

  KOKKOS_INLINE_FUNCTION Position<T, N> operator()(int i) const
  {
    auto out = +m_start;
    out[I] += i * m_step;
    return out;
  }

private:

  Position<T, N> m_start;
  T m_step;
  T m_stop;
};

namespace Impl {

template <int I, typename TIn, typename TOut>
class ProfileGenerator {
public:

  using Domain = Line<int, I, TIn::Rank>;
  using Row = Patch<TIn, Domain>;

  ProfileGenerator(const TIn& in, TOut& out) :
      m_in(in), m_out(out), m_start(in.domain().start()), m_stop(in.domain().stop(I))
  {}

  void operator()(auto... is) const
  {
    // m_out(is...) = Row();
    m_out.emplace_back(m_in, Domain(Position<int, TIn::Rank> {int(is)...} + m_start, m_stop));
  }

private:

  const TIn& m_in;
  TOut& m_out;
  const Position<int, TIn::Rank>& m_start;
  int m_stop;
};

} // namespace Impl
/**
 * @brief Get the collection of all the profiles of an image along a given axis.
 * @tparam I The profile axis.
 * 
 * A profile is a line-based patch.
 * 
 * \code
 * for (const auto& column : profiles<1>(image)) {
 *   // Do something with the column
 * }
 * \endcode
 * 
 * @see `rows()`
 */
template <int I, typename T, int N, typename TContainer>
auto profiles(const Image<T, N, TContainer>& in)
{
  using Domain = Line<int, I, N>;
  using Row = Patch<Image<T, N, TContainer>, Domain>;
  auto shape = in.shape();
  shape[I] = 1;
  // HostRaster<Row, N> out(compose_label("profiles", in), shape);
  std::vector<Row> out;
  out.reserve(product(shape));
  Linx::for_each<Kokkos::DefaultHostExecutionSpace>(
      "profiles()",
      Box<int, N> {Position<int, N> {}, shape}, // FIXME handle potential offset
      Impl::ProfileGenerator<I, Image<T, N, TContainer>, std::vector<Row>>(in, out));
  return out;
}

/**
 * @brief Get the collection of all the rows of an image.
 * 
 * \code
 * for (const auto& row : rows(image)) {
 *   // Do something with the row
 * }
 * \endcode
 * 
 * @see `profiles()`
 */
template <typename TIn>
auto rows(const TIn& in)
{
  return profiles<0>(in);
}

} // namespace Linx

#endif
