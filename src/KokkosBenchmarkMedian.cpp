// @copyright 2022-2024, Antoine Basset (CNES)
// This file is part of Linx <github.com/kabasset/Linx>
// SPDX-License-Identifier: Apache-2.0

#include "Linx/Data/Image.h"
#include "Linx/Run/ProgramContext.h"
#include "Linx/Transforms/RankFiltering.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>

void print_2d(const auto& image)
{
  auto name = image.label();
  auto width = image.shape()[0];
  auto height = image.shape()[1];
  std::cout << name << ":" << std::endl;
  std::cout << "  " << width << " x " << height << std::endl;

  const auto& on_host = Linx::on_host(image);
  std::cout << "  [" << on_host(0, 0) << ", ... , " << on_host(width - 1, height - 1) << "]" << std::endl;
}

int main(int argc, char const* argv[])
{
  Linx::ProgramContext context("", argc, argv);
  context.named("image", "Input length along each axis", 2048);
  context.named("kernel", "Kernel length along each axis", 5);
  context.flag("parity", "Enable parity tag");
  context.parse();
  const auto image_diameter = context.as<int>("image");
  const auto kernel_diameter = context.as<int>("kernel");
  const auto kernel_parity = context.as<bool>("parity");
  const auto output_diameter = image_diameter - kernel_diameter + 1;

  std::cout << "Generating input and kernel..." << std::endl;
  const auto image = Linx::Image<float, 2>("input", image_diameter, image_diameter);
  const auto kernel = Linx::Box(Linx::Position<2>(), Linx::Position<2>(Linx::Constant(kernel_diameter)));
  for_each(
      "init image",
      image.domain(),
      KOKKOS_LAMBDA(int i, int j) { image(i, j) = i + j; });
  Kokkos::fence();
  print_2d(image);
  std::cout << "kernel:" << std::endl;
  std::cout << "  " << kernel.extent(0) << " x " << kernel.extent(1) << std::endl;

  std::cout << "Filtering..." << std::endl;
  Kokkos::Timer timer;
  auto output = Linx::Image<float, 2>("output", output_diameter, output_diameter);
  if (kernel_parity) {
    Linx::median_filter_to(kernel, image, output); // Parity tag is inferred from kernel
  } else {
    output.copy_from(Linx::MedianFilter(kernel, image)); // Unknown parity
  }
  Kokkos::fence();
  const auto elapsed = timer.seconds();

  std::cout << "  Done in " << elapsed << " s" << std::endl;
  print_2d(output);

  return 0;
}
