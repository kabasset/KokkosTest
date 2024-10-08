// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE ImageCtorsTest

#include "Linx/Data/Image.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

LINX_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

template <typename T, int N, typename TContainer>
void check_ctor(const Linx::Image<T, N, TContainer>& image, const std::string& label, const Linx::Position<N>& shape)
{
  const auto size = Linx::product(shape);
  BOOST_TEST(image.rank() == shape.size());
  BOOST_TEST(image.label() == label);
  BOOST_TEST(image.size() == size);
  BOOST_TEST(image.ssize() == size);
  BOOST_TEST((image.shape() == shape));
  if (size == 0) {
    BOOST_TEST(image.empty());
    // FIXME for Raster: BOOST_TEST((end(image) == begin(image)));
  } else {
    BOOST_TEST(not image.empty());
    BOOST_TEST(image.data());
    BOOST_TEST((image.cdata() == image.data()));
    // FIXME for Raster: BOOST_TEST((end(image) != begin(image)));
    BOOST_TEST(image.contains_only(T {1}));
  }
}

LINX_TEST_CASE_TEMPLATE(static_empty_test)
{
  Linx::Position<0> shape;
  check_ctor(Linx::Image<T, 0>(), "", {});
  check_ctor(Linx::Image<T, 0>("i"), "i", {});
  check_ctor(Linx::Image<T, 0>(shape), "", {});
  check_ctor(Linx::Image<T, 0>("i", shape), "i", {});
}

LINX_TEST_CASE_TEMPLATE(dynamic_empty_test)
{
  Linx::Position<-1> shape;
  check_ctor(Linx::Image<T, -1>(), "", {});
  check_ctor(Linx::Image<T, -1>("i"), "i", {});
  check_ctor(Linx::Image<T, -1>(shape), "", {});
  check_ctor(Linx::Image<T, -1>("i", shape), "i", {});
}

LINX_TEST_CASE_TEMPLATE(static_singleton_fill_test)
{
  Linx::Position<1> shape {1};
  check_ctor(Linx::Image<T, 1>(1).fill(1), "", {1});
  check_ctor(Linx::Image<T, 1>("i", 1).fill(1), "i", {1});
  check_ctor(Linx::Image<T, 1>(shape).fill(1), "", {1});
  check_ctor(Linx::Image<T, 1>("i", shape).fill(1), "i", {1});
}

LINX_TEST_CASE_TEMPLATE(dynamic_singleton_fill_test)
{
  Linx::Position<-1> shape {1};
  check_ctor(Linx::Image<T, -1>(1).fill(1), "", {1});
  check_ctor(Linx::Image<T, -1>("i", 1).fill(1), "i", {1});
  check_ctor(Linx::Image<T, -1>(shape).fill(1), "", {1});
  check_ctor(Linx::Image<T, -1>("i", shape).fill(1), "i", {1});
}

LINX_TEST_CASE_TEMPLATE(static_multiple_fill_test)
{
  Linx::Position<3> shape {1, 2, 3};
  check_ctor(Linx::Image<T, 3>(1, 2, 3).fill(1), "", {1, 2, 3});
  check_ctor(Linx::Image<T, 3>("i", 1, 2, 3).fill(1), "i", {1, 2, 3});
  check_ctor(Linx::Image<T, 3>(shape).fill(1), "", {1, 2, 3});
  check_ctor(Linx::Image<T, 3>("i", shape).fill(1), "i", {1, 2, 3});
}

LINX_TEST_CASE_TEMPLATE(dynamic_multiple_fill_test)
{
  Linx::Position<-1> shape {1, 2, 3};
  check_ctor(Linx::Image<T, -1>(1, 2, 3).fill(1), "", {1, 2, 3});
  check_ctor(Linx::Image<T, -1>("i", 1, 2, 3).fill(1), "i", {1, 2, 3});
  check_ctor(Linx::Image<T, -1>(shape).fill(1), "", {1, 2, 3});
  check_ctor(Linx::Image<T, -1>("i", shape).fill(1), "i", {1, 2, 3});
}

LINX_TEST_CASE_TEMPLATE(wrapper_test)
{
  T v[6] = {1, 1, 1, 1, 1, 1};
  Linx::Position<3> shape {1, 2, 3};
  check_ctor(Linx::Raster<T, 3>(Linx::Wrap(v), 1, 2, 3), "", {1, 2, 3}); // FIXME CTAD
  check_ctor(Linx::Raster<T, 3>(Linx::Wrap(v), shape), "", shape); // FIXME CTAD
}

BOOST_AUTO_TEST_SUITE_END()
