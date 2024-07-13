// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE RasterOrderingTest

#include "Linx/Data/Image.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

using Linx::ProgramContext;
BOOST_TEST_GLOBAL_FIXTURE(ProgramContext);
BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE);

BOOST_AUTO_TEST_CASE(single_row_test)
{
  const int width = 4;
  Linx::Raster<int, 1> raster("row", width);
  BOOST_TEST(raster.size() == width);
  BOOST_TEST(raster.container().span() == width);
  BOOST_TEST(raster.container().span_is_contiguous());
  BOOST_TEST(raster.container().stride(0) == 1);
}

BOOST_AUTO_TEST_CASE(single_column_test)
{
  const int height = 3;
  Linx::Raster<int, 2> raster("column", 1, height);
  BOOST_TEST(raster.size() == height);
  BOOST_TEST(raster.container().span() == height);
  BOOST_TEST(raster.container().span_is_contiguous());
  BOOST_TEST(raster.container().stride(1) == 1);
}

BOOST_AUTO_TEST_CASE(rectangle_test)
{
  const int width = 4;
  const int height = 4;
  Linx::Raster<int, 2> raster("rectangle", width, height);
  BOOST_TEST(raster.size() == width * height);
  BOOST_TEST(raster.container().span() == width * height);
  BOOST_TEST(raster.container().span_is_contiguous());
  BOOST_TEST(raster.container().stride(0) == 1);
  BOOST_TEST(raster.container().stride(1) == width);
}

BOOST_AUTO_TEST_SUITE_END();
