// @copyright 2022-2024, Antoine Basset (CNES)
// This file is part of Linx <github.com/kabasset/Linx>
// SPDX-License-Identifier: Apache-2.0

#include "Linx/Data/Image.h"
#include "Linx/Run/ProgramContext.h"
#include "Linx/Transforms/Correlation.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>

void print_2d(const auto& image)
{
  auto name = image.label();
  auto width = image.shape()[0];
  auto height = image.shape()[1];
  std::cout << name << ":" << std::endl;
  std::cout << "  " << width << " x " << height << std::endl;
  std::cout << "  [" << image(0, 0) << ", ... , " << image(width - 1, height - 1) << "]" << std::endl;
}

int main(int argc, char const* argv[])
{
  Linx::ProgramContext context("", argc, argv);
  context.named("image", "Input length along each axis", 2048);
  context.named("kernel", "Kernel length along each axis", 5);
  context.parse();
  const auto image_diameter = context.as<int>("image");
  const auto kernel_diameter = context.as<int>("kernel");

  std::cout << "Generating input and kernel..." << std::endl;
  const auto image = Linx::Image<float, 2>("input", image_diameter, image_diameter);
  const auto kernel = Linx::Image<float, 2>("kernel", kernel_diameter, kernel_diameter);
  image.domain().iterate(
      "init image",
      KOKKOS_LAMBDA(int i, int j) { image(i, j) = i + j; });
  kernel.domain().iterate(
      "init kernel",
      KOKKOS_LAMBDA(int i, int j) { kernel(i, j) = i + j; });
  Kokkos::fence();
  print_2d(image);
  print_2d(kernel);

  std::cout << "Filtering..." << std::endl;
  Kokkos::Timer timer;
  const auto output = Linx::correlate("output", image, kernel);
  Kokkos::fence();
  const auto elapsed = timer.seconds();

  std::cout << "  Done in " << elapsed << " s" << std::endl;
  print_2d(output);

  return 0;
}
