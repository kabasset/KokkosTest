// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#include <Kokkos_Core.hpp>
#include <array>
#include <utility>

namespace Linx {

/**
 * @brief Alias for indices and sizes.
 */
using Index = long;

/**
 * @brief Alias for positions and shapes.
 */
template <Index N>
using Position = std::array<Index, N>;

/**
 * @brief Make a range execution policy for an ND view.
 */
template <typename TView>
auto range_policy(TView view)
{
  constexpr auto N = TView::rank();
  Kokkos::Array<std::int64_t, N> begin;
  Kokkos::Array<std::int64_t, N> end;
  for (int i = 0; i < N && i < 8; ++i) {
    begin[i] = 0;
    end[i] = view.extent(i);
  }
  return Kokkos::MDRangePolicy<Kokkos::Rank<N>>(begin, end);
}

/// @cond
namespace Internal {

template <typename T, Index N>
struct KokkosDatatype {
  using value = typename KokkosDatatype<T, N - 1>::value*;
};

template <typename T>
struct KokkosDatatype<T, 0> {
  using value = T;
};

} // namespace Internal
/// @endcond

/**
 * @brief ND array container.
 */
template <typename T, Index N>
class Raster {
public:

  /**
   * @brief Length-based constructor.
   */
  template <typename... TInts>
  Raster(const std::string& name, TInts... lengths) : m_view(name, lengths...)
  {}

  /**
   * @brief Shape-based constructor.
   */
  Raster(const std::string& name, const Position<N>& shape) : Raster(name, shape, std::make_index_sequence<N>()) {}

  /**
   * @brief Access pixel at given position.
   */
  inline decltype(auto) operator[](const Position<N>& position) const
  {
    return at(position, std::make_index_sequence<N>());
  }

  /**
   * @brief Access pixel at given index-based position.
   */
  template <typename... TInts>
  inline decltype(auto) operator()(TInts... indices) const
  {
    return m_view(indices...);
  }

  /**
   * @brief Apply a transform to each element.
   */
  template <typename TFunc, typename... Ts>
  void apply(TFunc&& func, const Ts&... others) const
  {
    generate(std::forward<TFunc>(func), m_view, others...);
  }

  /**
   * @brief Apply a generator to each element.
   */
  template <typename TFunc, typename... Ts>
  void generate(TFunc&& func, const Ts&... others) const
  {
    iterate(KOKKOS_LAMBDA(auto... is) { m_view(is...) = func(others(is...)...); });
  }

  /**
   * @brief Iterate over all positions.
   */
  template <typename TFunc>
  void iterate(TFunc&& func) const
  {
    Kokkos::parallel_for(range_policy(m_view), std::forward<TFunc>(func));
  }

private:

  /**
   * @brief Helper constructor to unroll shape.
   */
  template <std::size_t... Is>
  Raster(const std::string& name, const Position<N>& shape, std::index_sequence<Is...>) : Raster(name, shape[Is]...)
  {}

  /**
   * @brief Helper pixel accessor to unroll position.
   */
  template <std::size_t... Is>
  inline T& at(const Position<N>& position, std::index_sequence<Is...>) const
  {
    return operator()(position[Is]...);
  }

  /**
   * @brief The underlying `Kokkos::View`.
   */
  Kokkos::View<typename Internal::KokkosDatatype<T, N>::value> m_view;
};

} // namespace Linx
