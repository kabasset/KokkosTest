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
  const auto kernel_domain = kernel.domain();
  //   out.domain().iterate(
  //       name,
  //       KOKKOS_LAMBDA(auto... is) {
  //         out(is...) = kernel_domain.template reduce<typename TIn::value_type>(
  //             "dot product",
  //             KOKKOS_LAMBDA(auto& tmp, auto... js) { tmp += kernel(js...) * in((is + js)...); });
  //       });
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
