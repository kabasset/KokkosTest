// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXTRANSFORMS_REDUCTION_H
#define _LINXTRANSFORMS_REDUCTION_H

#include "Linx/Base/Containers.h"
#include "Linx/Base/Packs.h"
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
  const auto& readonly = as_readonly(in);
  readonly.domain().reduce(
      label,
      KOKKOS_LAMBDA(auto... is) { return readonly(is...); },
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
 * @param ins Input data containers
 */
template <typename TFunc, typename T, typename TProj, typename... TIns>
auto map_reduce(const std::string& label, TFunc&& func, T neutral, TProj&& projection, const TIns&... ins)
{
  return map_reduce_with_side_effects(
      label,
      LINX_FORWARD(func),
      neutral,
      LINX_FORWARD(projection),
      as_readonly(ins)...);
}

/**
 * @copydoc map_reduce()
 */
template <typename TFunc, typename T, typename TProj, typename... TIns>
auto map_reduce_with_side_effects(
    const std::string& label,
    TFunc&& func,
    T neutral,
    TProj&& projection,
    const TIns&... ins)
{
  using Reducer = Internal::Reducer<T, TFunc, typename PackTraits<TIns...>::First::Container::memory_space>;
  T value;
  (ins, ...).domain().reduce(
      label,
      KOKKOS_LAMBDA(auto... is) { return projection(ins(is...)...); },
      Reducer(value, LINX_FORWARD(func), LINX_FORWARD(neutral)));
  return value;
}

template <typename TIn>
auto min(const TIn& in)
{
  using T = typename TIn::element_type;
  T out;
  const auto& readonly = as_readonly(in);
  readonly.domain().reduce(
      compose_label("min", in),
      KOKKOS_LAMBDA(auto... is) { return readonly(is...); },
      Kokkos::Min<T>(out));
  Kokkos::fence();
  return out;
}

template <typename TIn>
auto max(const TIn& in)
{
  using T = typename TIn::element_type;
  T out;
  const auto& readonly = as_readonly(in);
  readonly.domain().reduce(
      compose_label("max", in),
      KOKKOS_LAMBDA(auto... is) { return readonly(is...); },
      Kokkos::Max<T>(out));
  Kokkos::fence();
  return out;
}

/**
 * @brief Compute the sum of all elements of a data container.
 */
template <typename TIn>
auto sum(const TIn& in) // FIXME limit to DataMixins
{
  using T = typename TIn::element_type; // FIXME to DataMixin
  return reduce("sum", std::plus {}, T {}, in);
}

/**
 * @brief Compute the dot product of two data containers.
 */
template <typename TLhs, typename TRhs>
auto dot(const TLhs& lhs, const TRhs& rhs)
{
  using T = typename TLhs::element_type; // FIXME to DataMixin
  return map_reduce("dot", std::plus {}, T {}, std::multiplies {}, lhs, rhs);
}

/**
 * @brief Compute the Lp-norm of a vector raised to the power p.
 * @tparam P The power
 */
template <int P, typename TIn>
auto norm(const TIn& in)
{
  using T = typename TIn::element_type;
  return map_reduce(
      "norm",
      std::plus {},
      T {},
      KOKKOS_LAMBDA(auto e) { return abspow<P>(e); },
      in);
}

/**
 * @brief Compute the absolute Lp-distance between two vectors raised to the power p.
 * @tparam P The power
 */
template <int P, typename TLhs, typename TRhs>
auto distance(const TLhs& lhs, const TRhs& rhs)
{
  using T = typename TLhs::element_type; // FIXME type of r - l
  return map_reduce(
      "distance",
      std::plus {},
      T {},
      KOKKOS_LAMBDA(auto l, auto r) { return abspow<P>(r - l); },
      lhs,
      rhs);
}

} // namespace Linx

#endif
