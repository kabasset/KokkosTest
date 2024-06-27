// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_IMAGE_H
#define _LINXDATA_IMAGE_H

#include "Linx/Base/TypeUtils.h"
#include "Linx/Data/Box.h"
#include "Linx/Data/Traits.h"
#include "Linx/Data/Vector.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <string>

namespace Linx {

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

  size_type size() const
  {
    return m_container.size();
  }

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
    return {LINX_MOVE(f), LINX_MOVE(e)};
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
    generate(name, LINX_FORWARD(func), m_container, ins...);
  }

  template <typename TFunc, typename... Ts>
  void generate(const std::string& name, TFunc&& func, const Ts&... ins) const
  {
    domain().iterate(
        name,
        KOKKOS_LAMBDA(auto... is) { m_container(is...) = func(ins(is...)...); });
  }

  template <typename TFunc>
  auto reduce(const std::string& name, TFunc&& func) const
  {
    using Result = std::result_of_t<TFunc && (value_type, value_type)>;
    Result out;
    auto data = m_container.data();
    Kokkos::parallel_reduce(
        name,
        size(),
        KOKKOS_LAMBDA(auto i, Result& tmp) { tmp = func(data[i], tmp); },
        out);
    return out;
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
