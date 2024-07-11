// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_IMAGE_H
#define _LINXDATA_IMAGE_H

#include "Linx/Base/Containers.h"
#include "Linx/Base/Types.h"
#include "Linx/Base/mixins/Math.h"
#include "Linx/Base/mixins/Range.h"
#include "Linx/Data/Box.h"
#include "Linx/Data/Vector.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <string>

namespace Linx {

/**
 * @brief ND array container.
 */
template <typename T, int N>
class Image :
    public RangeMixin<T, Image<T, N>>,
    public ArithmeticMixin<EuclidArithmetic, T, Image<T, N>>,
    public MathFunctionsMixin<T, Image<T, N>> {
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

  Image(const std::string& name, std::integral auto... lengths) : m_container(name, lengths...) {}

  Image(const std::string& name, const Vector<std::integral auto, Rank>& shape) :
      Image(name, shape, std::make_index_sequence<Rank>())
  {} // FIXME support N = -1

  auto label() const
  {
    return m_container.label();
  }

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

  auto data() const
  {
    return m_container.data();
  }

  KOKKOS_INLINE_FUNCTION decltype(auto) operator[](const Vector<std::integral auto, N>& position) const
  {
    return at(position, std::make_index_sequence<N>());
  }

  KOKKOS_INLINE_FUNCTION decltype(auto) operator()(std::integral auto... indices) const
  {
    return m_container(indices...);
  }

  template <typename TFunc, typename... Ts>
  const Image& apply(const std::string& name, TFunc&& func, const Ts&... ins) const
  {
    return generate(name, LINX_FORWARD(func), m_container, ins...);
  }

  template <typename TFunc, typename... Ts>
  const Image& generate(const std::string& name, TFunc&& func, const Ts&... ins) const
  {
    domain().iterate(
        name,
        KOKKOS_LAMBDA(auto... is) { m_container(is...) = func(ins(is...)...); });
    return *this;
  }

  template <typename TRed>
  auto reduce(const std::string& name, TRed&& reducer) const // FIXME as free function
  {
    return domain().reduce(
        name,
        KOKKOS_LAMBDA(auto... is) { return m_container(is...); },
        LINX_FORWARD(reducer));
  }

  template <typename TRed, typename TProj, typename... Ts>
  auto reduce(const std::string& name, TRed&& reducer, TProj&& projection, const Ts&... ins) const
  {
    return domain().reduce(
        name,
        KOKKOS_LAMBDA(auto... is) { return projection(m_container(is...), ins(is...)...); },
        LINX_FORWARD(reducer));
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
