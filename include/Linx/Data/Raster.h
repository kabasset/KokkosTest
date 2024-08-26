// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_RASTER_H
#define _LINXDATA_RASTER_H

#include "Linx/Data/Image.h"

namespace Linx {

/**
 * @brief Contiguous image with row-major ordering.
 * 
 * As opposed to `Image`, this class is a standard range, and can therefore be iterated.
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
   * @see `linspace()`
   */
  const Raster& range(const T& min = Limits<T>::zero(), const T& step = Limits<T>::one()) const
  {
    range_impl(min, step);
    return *this;
  }

  /**
   * @brief Fill the container with evenly spaced value.
   * @see `range()`
   */
  const Raster& linspace(const T& min = Limits<T>::zero(), const T& max = Limits<T>::one()) const
  {
    const auto step = (max - min) / (this->size() - 1);
    return range(min, step);
  }

  /**
   * @brief Reverse the order of the elements.
   */
  const Raster& reverse() const // FIXME to DataMixin
  {
    std::reverse(begin(), end());
    return *this;
  }

  /**
   * @brief Helper method which returns void.
   */
  void range_impl(const T& min, const T& step) const
  // FIXME make private somehow?
  {
    const auto size = this->size();
    auto ptr = this->data();
    Kokkos::parallel_for("range()", size, KOKKOS_LAMBDA(int i) { ptr[i] = min + step * i; });
  }
    
};

} // namespace Linx

#endif
