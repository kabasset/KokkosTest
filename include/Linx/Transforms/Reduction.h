// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXTRANSFORMS_REDUCTION_H
#define _LINXTRANSFORMS_REDUCTION_H

#include "Linx/Base/Containers.h"
#include "Linx/Base/Types.h"
#include "Linx/Base/mixins/Data.h"
#include "Linx/Data/Box.h"
#include "Linx/Data/Sequence.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <string>

namespace Linx {

/// @cond
namespace Internal {
template <typename T, typename TFunc, typename TSpace>
class Reducer {
public:

  using reducer = Reducer; // Required for concept
  using value_type = std::remove_cv_t<T>;
  typedef Kokkos::View<value_type, TSpace> result_view_type;

  KOKKOS_INLINE_FUNCTION Reducer(value_type& value, TFunc&& func, T&& neutral) :
      m_view(&value), m_func(LINX_FORWARD(func)), m_neutral(LINX_FORWARD(neutral))
  {}

  KOKKOS_INLINE_FUNCTION Reducer(const result_view_type& view, TFunc&& func, T&& neutral) :
      m_view(view), m_func(LINX_FORWARD(func)), m_neutral(LINX_FORWARD(neutral))
  {}

  KOKKOS_INLINE_FUNCTION void join(value_type& dst, const value_type& src) const
  {
    dst = m_func(dst, src);
  }

  KOKKOS_INLINE_FUNCTION void init(value_type& value) const
  {
    value = m_neutral;
  }

  KOKKOS_INLINE_FUNCTION value_type& reference() const
  {
    return *m_view.data();
  }

  KOKKOS_INLINE_FUNCTION result_view_type view() const
  {
    return m_view;
  }

private:

  result_view_type m_view;
  TFunc m_func;
  T m_neutral;
};
} // namespace Internal

/**
 * @brief Compute a reduction.
 * 
 * @param label A label for debugging
 * @param func The reduction function
 * @param neutral The reduction neutral element
 * @param in The input data container
 */
template <typename TFunc, typename T, typename TIn>
auto reduce(const std::string& label, TFunc&& func, T neutral, const TIn& in)
{
  using Reducer = Internal::Reducer<T, TFunc, typename TIn::Container::memory_space>;
  T value;
  in.domain().reduce(
      label,
      KOKKOS_LAMBDA(auto... is) { return in(is...); },
      Reducer(value, LINX_FORWARD(func), LINX_FORWARD(neutral)));
  return value;
}

/**
 * @brief Compute a reduction with mapping.
 * 
 * @param label A label for debugging
 * @param func The reduction function
 * @param neutral The reduction neutral element
 * @param projection The mapping function
 * @param in0 The input data container
 * @param others Optional input data containers
 */
template <typename TFunc, typename T, typename TProj, typename TIn0, typename... TIns>
auto map_reduce(
    const std::string& label,
    TFunc&& func,
    T neutral,
    TProj&& projection,
    const TIn0& in0,
    const TIns&... ins)
{
  using Reducer = Internal::Reducer<T, TFunc, typename TIn0::Container::memory_space>;
  T value;
  in0.domain().reduce(
      label,
      KOKKOS_LAMBDA(auto... is) { return projection(in0(is...), ins(is...)...); },
      Reducer(value, LINX_FORWARD(func), LINX_FORWARD(neutral)));
  return value;
}

/**
 * @brief Compute the sum of all elements of a data container.
 */
template <typename TIn>
auto sum(const std::string& label, const TIn& in) // FIXME limit to DataMixins
{
  using T = typename TIn::element_type; // FIXME to DataMixin
  return reduce(label, std::plus {}, T {}, in);
}

/**
 * @brief Compute the dot product of two data containers.
 */
template <typename TLhs, typename TRhs>
auto dot(const std::string& label, const TLhs& lhs, const TRhs& rhs) // FIXME limit to DataMixins
{
  using T = typename TLhs::element_type; // FIXME to DataMixin
  return map_reduce(label, std::plus {}, T {}, std::multiplies {}, lhs, rhs);
}

} // namespace Linx

#endif
