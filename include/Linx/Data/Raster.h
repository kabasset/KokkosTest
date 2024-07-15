// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_RASTER_H
#define _LINXDATA_RASTER_H

#include "Linx/Data/Image.h"

namespace Linx {

/**
 * @brief Contiguous image with row-major ordering.
 */
template <typename T, int N>
class Raster : public Image<T, N, typename DefaultContainer<T, N, Kokkos::LayoutLeft>::Image> {
public:

  using Super = Image<T, N, typename DefaultContainer<T, N, Kokkos::LayoutLeft>::Image>;
  using iterator = Super::pointer;

  // Inherit constructors
  using Super::Image;

  // No need for a virtual destructor, because there are no member variables

  /**
   * @brief Iterator to the beginning.
   */
  iterator begin() const
  {
    return this->data();
  }

  /**
   * @brief Iterator to the end.
   */
  iterator end() const
  {
    return begin() + this->size();
  }

  /**
   * @brief Fill the container with evenly spaced value.
   * 
   * The difference between two adjacent values is _exactly_ `step`,
   * i.e. `container[i + 1] = container[i] + step`.
   * This means that rounding errors may sum up,
   * as opposed to `linspace()`.
   * 
   * @see `linspace()`
   */
  const Raster& range(const T& min = Limits<T>::zero(), const T& step = Limits<T>::one()) const
  {
    auto v = min;
    for (auto& e : *this) {
      e = v;
      v += step;
    }
    return *this;
  }

  /**
   * @brief Fill the container with evenly spaced value.
   * 
   * The first and last values of the container are _exactly_ `min` and `max`.
   * Intermediate values are computed as `container[i] = min + (max - min) / (size() - 1) * i`,
   * which means that the difference between two adjacent values
   * is not necessarily perfectly constant for floating point values,
   * as opposed to `range()`.
   * 
   * @see `range()`
   */
  const Raster& linspace(const T& min = Limits<T>::zero(), const T& max = Limits<T>::one()) const
  {
    const std::size_t size = this->size();
    const auto step = (max - min) / (size - 1);
    auto it = begin();
    for (std::size_t i = 0; i < size - 1; ++i, ++it) {
      *it = min + step * i;
    }
    *it = max;
    return *this;
  }
};

} // namespace Linx

#endif
