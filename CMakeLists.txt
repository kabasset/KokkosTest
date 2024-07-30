# Setup

cmake_minimum_required(VERSION 3.16)
project(KokkosTest)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Dependencies

find_package(Boost COMPONENTS unit_test_framework program_options REQUIRED)
find_package(Kokkos REQUIRED)

# Library

add_library(Linx INTERFACE)
target_include_directories(Linx INTERFACE include)
target_link_libraries(Linx INTERFACE Kokkos::kokkos ${Boost_LIBRARIES})

# Executables

add_executable(KokkosBenchmarkConvolution src/KokkosBenchmarkConvolution.cpp)
target_link_libraries(KokkosBenchmarkConvolution Linx)

add_executable(KokkosBenchmarkExp src/KokkosBenchmarkExp.cpp)
target_link_libraries(KokkosBenchmarkExp Linx)

add_executable(KokkosBenchmarkIteration src/KokkosBenchmarkIteration.cpp)
target_link_libraries(KokkosBenchmarkIteration Linx)

# Tests

enable_testing()

add_executable(KokkosSmoke_test tests/KokkosSmoke_test.cpp)
target_link_libraries(KokkosSmoke_test Kokkos::kokkos ${Boost_LIBRARIES})
add_test(KokkosSmoke_test KokkosSmoke_test)

add_executable(BoxApply_test tests/BoxApply_test.cpp)
target_link_libraries(BoxApply_test Linx ${Boost_LIBRARIES})
add_test(BoxApply_test BoxApply_test)

add_executable(Histogram_test tests/Histogram_test.cpp)
target_link_libraries(Histogram_test Linx ${Boost_LIBRARIES})
add_test(Histogram_test Histogram_test)

add_executable(ImageApply_test tests/ImageApply_test.cpp)
target_link_libraries(ImageApply_test Linx ${Boost_LIBRARIES})
add_test(ImageApply_test ImageApply_test)

add_executable(ImageCorrelate_test tests/ImageCorrelate_test.cpp)
target_link_libraries(ImageCorrelate_test Linx ${Boost_LIBRARIES})
add_test(ImageCorrelate_test ImageCorrelate_test)

add_executable(ImagePixelwise_test tests/ImagePixelwise_test.cpp)
target_link_libraries(ImagePixelwise_test Linx ${Boost_LIBRARIES})
add_test(ImagePixelwise_test ImagePixelwise_test)

add_executable(Packs_test tests/Packs_test.cpp)
target_link_libraries(Packs_test Linx ${Boost_LIBRARIES})
add_test(Packs_test Packs_test)

add_executable(Patch_test tests/Patch_test.cpp)
target_link_libraries(Patch_test Linx ${Boost_LIBRARIES})
add_test(Patch_test Patch_test)

add_executable(RasterOrdering_test tests/RasterOrdering_test.cpp)
target_link_libraries(RasterOrdering_test Linx ${Boost_LIBRARIES})
add_test(RasterOrdering_test RasterOrdering_test)

add_executable(Reduction_test tests/Reduction_test.cpp)
target_link_libraries(Reduction_test Linx ${Boost_LIBRARIES})
add_test(Reduction_test Reduction_test)

add_executable(Region_test tests/Region_test.cpp)
target_link_libraries(Region_test Linx ${Boost_LIBRARIES})
add_test(Region_test Region_test)

add_executable(Sequence_test tests/Sequence_test.cpp)
target_link_libraries(Sequence_test Linx ${Boost_LIBRARIES})
add_test(Sequence_test Sequence_test)

add_executable(Slice_test tests/Slice_test.cpp)
target_link_libraries(Slice_test Linx ${Boost_LIBRARIES})
add_test(Slice_test Slice_test)
