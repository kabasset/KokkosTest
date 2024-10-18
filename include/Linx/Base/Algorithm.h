// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXBASE_ALGORITHM_H
#define _LINXBASE_ALGORITHM_H

#include "Linx/Base/Types.h"

#include <algorithm> // min
#include <numeric> // midpoint

namespace Linx {

/**
 * @brief Sort the n first values of an array.
 * 
 * While `std::nth_element()` typically relies on introselect, this function implements insertion-sort,
 * which has higher complexity but should be faster for small arrays, which is typically the case for rank-filtering.
 */
template <typename TInOut>
const auto& sort_n(TInOut& in_out, Index n)
{
  typename TInOut::element_type current;
  std::size_t j;
  for (std::size_t i = 0; i < in_out.size(); ++i) {
    j = std::min<std::size_t>(i, n + 1);
    current = in_out[i];
    in_out[i] = in_out[j];
    for (; j > 0 && current < in_out[j - 1]; --j) {
      in_out[j] = in_out[j - 1];
    }
    in_out[j] = current;
  }
  return in_out[n];
}

/**
 * @brief Get the median of an array of odd length.
 * 
 * @warning Elements of `in_out` are shuffled (partially sorted).
 */
template <typename TInOut>
const auto& median_odd(TInOut& in_out)
{
  return sort_n(in_out, in_out.size() / 2);
}

/**
 * @brief Get the median of an array of even length.
 * 
 * The median is computed as the arithmetic mean of the `n`-th and `n + 1`-th elements of the array,
 * where `n` is half the size of the array.
 * 
 * @warning Elements of `in_out` are shuffled (partially sorted).
 */
template <typename TInOut>
auto median_even(TInOut& in_out)
{
  const auto& high = sort_n(in_out, in_out.size() / 2 + 1);
  const auto& low = *(&high - 1);
  return std::midpoint(low, high);
}

/**
 * @brief Get the median of an array.
 * 
 * This function simply picks `median_even()` or `median_odd()` depending on the array size.
 * 
 * @warning Elements of `in_out` are shuffled (partially sorted).
 */
template <typename TInOut>
auto median(TInOut& in_out)
{
  const auto size = in_out.size();
  if (size % 2 == 0) {
    return median_even(in_out);
  } else {
    return median_odd(in_out);
  }
}

} // namespace Linx

#endif
