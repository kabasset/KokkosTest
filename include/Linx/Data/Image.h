// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_IMAGE_H
#define _LINXDATA_IMAGE_H

#include "Linx/Data/Traits.h"
#include "Linx/Data/Vector.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <string>

namespace Linx {

template <typename T, int N>
class Box {
public:

  static constexpr int Rank = N;
  using Container = Vector<T, N>;

  using value_type = T;
  using element_type = std::decay_t<T>;
  using size_type = typename Container::size_type;
  using difference_type = std::ptrdiff_t;
  using reference = typename Container::reference;
  using pointer = typename Container::pointer;
  using iterator = pointer;

  const auto& front() const
  {
    return m_front;
  }
  const auto& back() const
  {
    return m_back;
  }

  template <typename TFunc>
  void iterate(const std::string& name, TFunc&& func) const
  {
    Kokkos::parallel_for(name, kokkos_range_policy(), std::forward<TFunc>(func));
  }

  // private:

  auto kokkos_range_policy() const
  {
    Kokkos::Array<std::int64_t, Rank> f;
    Kokkos::Array<std::int64_t, Rank> e;
    for (int i = 0; i < Rank; ++i) {
      f[i] = m_front[i];
      e[i] = m_back[i];
    }
    return Kokkos::MDRangePolicy<Kokkos::Rank<Rank>>(f, e); // FIXME layout?
  }

  Container m_front;
  Container m_back;
};

/**
 * @brief ND array container.
 */
template <typename T, int N>
class Image {
public:

  static constexpr int Rank = N;
  using Container = typename ContainerTraits<T, N>::Image;

  using value_type = typename Container::value_type;
  using element_type = std::decay_t<value_type>;
  using size_type = typename Container::size_type;
  using difference_type = std::ptrdiff_t;
  using reference = typename Container::reference_type;
  // using const_reference = typename Container::const_reference_type;
  using pointer = typename Container::pointer_type;
  // using const_pointer = typename Container::const_pointer_type;
  using iterator = pointer;
  // using const_iterator = typename Container::const_iterator;

  template <typename... TInts>
  Image(const std::string& name, TInts... lengths) : m_container(name, lengths...)
  {}

  template <typename TInt> // FIXME support N = -1
  Image(const std::string& name, const Vector<TInt, Rank>& shape) : Image(name, shape, std::make_index_sequence<Rank>())
  {}

  KOKKOS_INLINE_FUNCTION Vector<int, N> shape() const
  {
    Vector<int, N> out;
    for (int i = 0; i < N; ++i) {
      out[i] = m_container.extent(i);
    }
    return out;
  }

  KOKKOS_INLINE_FUNCTION Box<int, N> domain() const
  {
    Vector<int, N> f;
    Vector<int, N> e;
    for (int i = 0; i < N; ++i) {
      f[i] = 0;
      e[i] = m_container.extent_int(i);
    }
    return {std::move(f), std::move(e)};
  }

  const auto& container() const
  {
    return m_container;
  }

  template <typename TInt>
  KOKKOS_INLINE_FUNCTION decltype(auto) operator[](const Vector<TInt, N>& position) const
  {
    return at(position, std::make_index_sequence<N>());
  }

  template <typename... TInts>
  KOKKOS_INLINE_FUNCTION decltype(auto) operator()(TInts... indices) const
  {
    return m_container(indices...);
  }

  template <typename TFunc, typename... Ts>
  void apply(const std::string& name, TFunc&& func, const Ts&... ins) const
  {
    generate(name, std::forward<TFunc>(func), m_container, ins...);
  }

  template <typename TFunc, typename... Ts>
  void generate(const std::string& name, TFunc&& func, const Ts&... ins) const
  {
    domain().iterate(
        name,
        KOKKOS_LAMBDA(auto... is) { m_container(is...) = func(ins(is...)...); });
  }

private:

  /**
   * @brief Helper constructor to unroll shape.
   */
  template <typename TShape, std::size_t... Is>
  Image(const std::string& name, const TShape& shape, std::index_sequence<Is...>) : Image(name, shape[Is]...)
  {}

  /**
   * @brief Helper pixel accessor to unroll position.
   */
  template <typename TPosition, std::size_t... Is>
  inline T& at(const TPosition& position, std::index_sequence<Is...>) const
  {
    return operator()(position[Is]...);
  }

  /**
   * @brief The underlying container.
   */
  Container m_container;
};

} // namespace Linx

#endif
