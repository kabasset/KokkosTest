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
using Index = std::int64_t;

/**
 * @brief Alias for positions and shapes.
 */
template <Index N>
using Position = Kokkos::Array<Index, N>;

/**
 * @brief Axis-aligned bounding box.
 */
template <Index N>
struct Box {
  Position<N> front;
  Position<N> back;
};

/**
 * @brief Make a range execution policy for an ND view.
 */
template <typename TView>
auto kokkos_exec_policy(const TView& view)
{
  constexpr auto N = TView::rank();
  Kokkos::Array<std::int64_t, N> begin;
  Kokkos::Array<std::int64_t, N> end;
  for (int i = 0; i < N && i < 8; ++i) {
    begin[i] = 0;
    end[i] = view.extent(i);
  }
  return Kokkos::MDRangePolicy<Kokkos::Rank<N>>(begin, end); // FIXME layout
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
class Array {
public:

  /**
   * @brief Length-based constructor.
   */
  template <typename... TInts>
  Array(const std::string& name, TInts... lengths) : m_view(name, lengths...)
  {}

  /**
   * @brief Shape-based constructor.
   */
  Array(const std::string& name, const Position<N>& shape) : Array(name, shape, std::make_index_sequence<N>()) {}

  Position<N> shape() const
  {
    Position<N> out;
    for (int i = 0; i < N; ++i) {
      out[i] = m_view.extent(i);
    }
    return out;
  }

  Box<N> domain() const
  {
    Position<N> begin;
    Position<N> end;
    for (int i = 0; i < N && i < 8; ++i) {
      begin[i] = 0;
      end[i] = m_view.extent(i);
    }
    return {std::move(begin), std::move(end)};
  }

  const auto& view() const
  {
    return m_view;
  }

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
   * @param func The transform
   * @param ins The optional inputs
   * 
   * The provided function takes as input the current element value of this array,
   * and the current element value of each input,
   * and returns a new value for the current element of this array.
   */
  template <typename TFunc, typename... Ts>
  void apply(TFunc&& func, const Ts&... ins) const
  {
    generate(std::forward<TFunc>(func), m_view, ins...);
  }

  /**
   * @brief Apply a generator to each element.
   * @param func The generator
   * @param ins The optional inputs
   * 
   * The provided function takes as input the current element value of each input (possibly none),
   * and returns a value for the current element of this array.
   */
  template <typename TFunc, typename... Ts>
  void generate(TFunc&& func, const Ts&... ins) const
  {
    iterate(KOKKOS_LAMBDA(auto... is) { m_view(is...) = func(ins(is...)...); });
  }

  /**
   * @brief Iterate over all positions.
   * 
   * The provided function takes as input the index-based positions and returns a value.
   */
  template <typename TFunc>
  void iterate(TFunc&& func) const
  {
    Kokkos::parallel_for(kokkos_exec_policy(m_view), std::forward<TFunc>(func));
  }

private:

  /**
   * @brief Helper constructor to unroll shape.
   */
  template <std::size_t... Is>
  Array(const std::string& name, const Position<N>& shape, std::index_sequence<Is...>) : Array(name, shape[Is]...)
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
  // FIXME fall back to Raster for N > 8
  // FIXME use DynRankView for N = -1 & dimension < 8
  // FIXME fall back to Raster for N = -1 & dimension > 7
};

} // namespace Linx
