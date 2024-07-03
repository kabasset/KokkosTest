// @copyright 2022-2024, Antoine Basset (CNES)
// This file is part of Linx <github.com/kabasset/Linx>
// SPDX-License-Identifier: Apache-2.0

#include "Linx/Data/Image.h"
#include "Linx/Run/ProgramContext.h"
#include "Linx/Transforms/Correlation.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>

int main(int argc, char const* argv[])
{
  Linx::ProgramContext context("", argc, argv);
  context.named("image", "Raster length along each axis", 2048);
  context.named("kernel", "Kernel length along each axis", 5);
  context.parse();
  const auto image_diameter = context.as<int>("image");
  const auto kernel_diameter = context.as<int>("kernel");

  std::cout << "Generating raster and kernel..." << std::endl;
  const auto image = Linx::Image<float, 2>("image", image_diameter, image_diameter);
  const auto kernel = Linx::Image<float, 2>("kernel", kernel_diameter, kernel_diameter);
  image.generate(
      "init image",
      KOKKOS_LAMBDA() { return 1; });
  kernel.generate(
      "init kernel",
      KOKKOS_LAMBDA() { return 1; });
  Kokkos::fence();

  std::cout << "Filtering..." << std::endl;
  Kokkos::Timer timer;
  const auto output = Linx::correlate("correlate", image, kernel);
  Kokkos::fence();
  const auto elapsed = timer.seconds();

  std::cout << "  Done in " << elapsed << " s" << std::endl;

  return 0;
}
