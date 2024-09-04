// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_RASTER_H
#define _LINXDATA_RASTER_H

#include "Linx/Base/mixins/Range.h"
#include "Linx/Data/Image.h"

namespace Linx {

/**
 * @brief Contiguous image with row-major ordering.
 * 
 * As opposed to `Image`, this class is a standard range, and can therefore be iterated.
 */
template <typename T, int N> // FIXME foward TArgs to DefaultContainers?
class Raster :
    public Image<T, N, typename DefaultContainer<T, N, Kokkos::LayoutLeft>::Image>,
    public RangeMixin<T, Raster<T, N>> {
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
};

} // namespace Linx

#endif
