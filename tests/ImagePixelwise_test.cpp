// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE "ImagePixelwiseTest"

#include "Linx/Data/Image.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

template <typename T, int N>
bool operator==(const Linx::Image<T, N>& lhs, const Linx::Image<T, N>& rhs) // FIXME to Arithmetic?
{
  for (int i = 0; i < lhs.shape()[0]; ++i) {
    for (int j = 0; j < lhs.shape()[1]; ++j) {
      if (lhs(i, j) != rhs(i, j)) {
        return false;
      }
    }
  }
  return true;
}

template <typename T, int N>
bool operator==(const Linx::Image<T, N>& lhs, T rhs) // FIXME to Arithmetic?
{
  for (int i = 0; i < lhs.shape()[0]; ++i) {
    for (int j = 0; j < lhs.shape()[1]; ++j) {
      if (lhs(i, j) != rhs) {
        std::cout << lhs(i, j) << " != " << rhs << std::endl;
        return false;
      }
    }
  }
  return true;
}

using Linx::ProgramContext;
BOOST_TEST_GLOBAL_FIXTURE(ProgramContext);

BOOST_AUTO_TEST_SUITE(ImagePixelwiseTest);

BOOST_AUTO_TEST_CASE(exp_test)
{
  const int width = 4;
  const int height = 3;
  using Image = Linx::Image<double, 2>;

  Image a("a", width, height);
  BOOST_TEST((a == 0.));

  auto b = exp(a); // FIXME add name as first argument
  BOOST_TEST((a == 0.));
  BOOST_TEST((b == 1.));

  a.exp();
  BOOST_TEST((a == 1.));
  BOOST_TEST((b == 1.));
}

BOOST_AUTO_TEST_SUITE_END();
