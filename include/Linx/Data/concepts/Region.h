// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_CONCEPTS_REGION_H
#define _LINXDATA_CONCEPTS_REGION_H

#include "Linx/Base/concepts/Arithmetic.h"

namespace Linx {

/**
 * @brief Concept for all regions.
 * 
 * A region is a collection of positions, which can be iterated.
 * If the region can be shifted, it is a window.
 * 
 * @see `Window`
 */
template <typename T>
concept Region = requires(const T region)
{
  typename T::size_type;
  typename T::Rank;
  typename T::value_type;
  std::size(region);
  region& Box<typename T::size_type, T::Rank>();
  for_each("", region, [](auto... is) {});
};

/**
 * @brief Concept for windows, i.e. additive regions.
 * 
 * Patches whose domain are windows can be translated.
 * 
 * @see `Region`
 */
template <typename T>
concept Window = Region<T> && Additive<T, typename T::value_type> && Additive<T, typename T::size_type>;

} // namespace Linx

#endif
