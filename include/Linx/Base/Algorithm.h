// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXBASE_ALGORITHM_H
#define _LINXBASE_ALGORITHM_H

#include "Linx/Base/Functional.h"
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
  using T = std::remove_cvref_t<decltype(in_out[0])>;
  T current;
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
 * @brief Get the median of an array.
 * @tparam TParity The parity of the array, if known (`OddNumber`, `EvenNumber` or `Forward`)
 * 
 * This function simply picks `median_even()` or `median_odd()` depending on the array size.
 * 
 * If the array size is odd, the median is computed as the arithmetic mean of the `n`-th and `n + 1`-th elements of the array,
 * where `n` is half the size of the array.
 * 
 * @warning Elements of `in_out` are shuffled (partially sorted).
 */
template <typename TParity = Forward, typename TInOut>
auto median(TInOut& in_out)
{
  const auto size = in_out.size();

  if constexpr (std::is_same_v<TParity, OddNumber>) {
    return sort_n(in_out, in_out.size() / 2);
  } else if constexpr (std::is_same_v<TParity, EvenNumber>) {
    const auto& high = sort_n(in_out, in_out.size() / 2 + 1);
    const auto& low = *(&high - 1);
    return std::midpoint(low, high);
  } else {
    if (size % 2 == 0) {
      return median<EvenNumber>(in_out);
    } else {
      return median<OddNumber>(in_out);
    }
  }
}

} // namespace Linx

#endif
