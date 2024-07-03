// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXTRANSFORMS_CORRELATION_H
#define _LINXTRANSFORMS_CORRELATION_H

#include "Linx/Data/Image.h"
#include "Linx/Data/Vector.h"

#include <Kokkos_StdAlgorithms.hpp>
#include <string>

namespace Linx {

template <typename TIn, typename TKernel, typename TOut>
void correlate_to(const std::string& name, const TIn& in, const TKernel& kernel, TOut& out)
{
  using T = typename TIn::value_type;
  out.domain().iterate(
      name,
      KOKKOS_LAMBDA(auto... is) {
        Image<T, TIn::Rank> patch("correlation patch", kernel.shape()); // FIXME use same layout as kernel
        auto container = patch.container();
        patch.domain().iterate( // FIXME sequential?
            name,
            KOKKOS_LAMBDA(auto... js) { container(js...) = in((is + js)...); });
        for (int i = 0; i < patch.shape()[0]; ++i) {
          for (int j = 0; j < patch.shape()[1]; ++j) {
          }
        }
        auto patch_begin = patch.container().data(); // FIXME only works for contiguous data;
        auto patch_end = patch_begin + patch.size();
        for (std::size_t i = 0; i < patch.size(); ++i) {
        }
        auto kernel_begin = kernel.container().data();
        namespace KE = Kokkos::Experimental;
        out(is...) = std::inner_product(patch_begin, patch_end, kernel_begin, T {});
      });
}

template <typename TIn, typename TKernel>
auto correlate(const std::string& name, const TIn& in, const TKernel& kernel)
{
  TKernel out("correlation out", in.shape() - kernel.shape());
  correlate_to(name, in, kernel, out);
  return out;
}

} // namespace Linx

#endif
