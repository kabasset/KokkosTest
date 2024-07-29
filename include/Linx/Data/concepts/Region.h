// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_CONCEPTS_REGION_H
#define _LINXDATA_CONCEPTS_REGION_H

namespace Linx {

/**
 * @brief Concept for all regions.
 */
template <typename T>
concept Region = requires(const T region, Box<typename T::value_type, T::Rank> box)
{
  typename T::value_type;
  typename T::Rank;
  typename T::Position;
  region.size();
  region &= box;
  region.iterate("", [](auto... is) {});
};

template <typename T, typename U>
concept Additive = requires(T lhs, const U rhs)
{
  ++lhs;
  --lhs;
  lhs++;
  rhs++;
  lhs += rhs;
  lhs -= rhs;
  lhs + rhs;
  lhs - rhs;
};

template <typename T>
concept Window = Region<T> && Additive<T, typename T::value_type> && Additive<T, typename T::Position>;

} // namespace Linx

#endif
