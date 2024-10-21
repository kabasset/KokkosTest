# Setup

cmake_minimum_required(VERSION 3.16)
project(
  KokkosTest
  VERSION 1.0
  LANGUAGES CXX
  DESCRIPTION "Cross-platform ND image library")

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

# Doc

find_package(Doxygen)

if(DOXYGEN_FOUND)
  set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/include)
  set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doc/doxygen)
  set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/xml/index.xml)
  set(DOXYFILE_IN ${PROJECT_SOURCE_DIR}/cmake/doc/Doxyfile.in)
  set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

  configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY) # Interpolate Doxyfile
  file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})

  add_custom_command(
    OUTPUT ${DOXYGEN_INDEX_FILE}
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
    MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
    COMMENT "Generating Doxygen documentation"
    VERBATIM)

  add_custom_target(doc DEPENDS ${DOXYGEN_INDEX_FILE})
endif(DOXYGEN_FOUND)

# Executables

add_executable(KokkosBenchmarkConvolution src/KokkosBenchmarkConvolution.cpp)
target_link_libraries(KokkosBenchmarkConvolution Linx)

add_executable(KokkosBenchmarkExp src/KokkosBenchmarkExp.cpp)
target_link_libraries(KokkosBenchmarkExp Linx)

add_executable(KokkosBenchmarkIteration src/KokkosBenchmarkIteration.cpp)
target_link_libraries(KokkosBenchmarkIteration Linx)

add_executable(KokkosBenchmarkMedian src/KokkosBenchmarkMedian.cpp)
target_link_libraries(KokkosBenchmarkMedian Linx)

# Tests

enable_testing()

add_executable(KokkosSmoke_test tests/KokkosSmoke_test.cpp)
target_link_libraries(KokkosSmoke_test Kokkos::kokkos ${Boost_LIBRARIES})
add_test(KokkosSmoke_test KokkosSmoke_test)

add_executable(AlgorithmSort_test tests/AlgorithmSort_test.cpp)
target_link_libraries(AlgorithmSort_test Linx ${Boost_LIBRARIES})
add_test(AlgorithmSort_test AlgorithmSort_test)

add_executable(BoxApply_test tests/BoxApply_test.cpp)
target_link_libraries(BoxApply_test Linx ${Boost_LIBRARIES})
add_test(BoxApply_test BoxApply_test)

add_executable(BoxArithmetic_test tests/BoxArithmetic_test.cpp)
target_link_libraries(BoxArithmetic_test Linx ${Boost_LIBRARIES})
add_test(BoxArithmetic_test BoxArithmetic_test)

add_executable(BoxCtors_test tests/BoxCtors_test.cpp)
target_link_libraries(BoxCtors_test Linx ${Boost_LIBRARIES})
add_test(BoxCtors_test BoxCtors_test)

add_executable(Histogram_test tests/Histogram_test.cpp)
target_link_libraries(Histogram_test Linx ${Boost_LIBRARIES})
add_test(Histogram_test Histogram_test)

add_executable(ImageApply_test tests/ImageApply_test.cpp)
target_link_libraries(ImageApply_test Linx ${Boost_LIBRARIES})
add_test(ImageApply_test ImageApply_test)

add_executable(ImageCorrelate_test tests/ImageCorrelate_test.cpp)
target_link_libraries(ImageCorrelate_test Linx ${Boost_LIBRARIES})
add_test(ImageCorrelate_test ImageCorrelate_test)

add_executable(ImageCtors_test tests/ImageCtors_test.cpp)
target_link_libraries(ImageCtors_test Linx ${Boost_LIBRARIES})
add_test(ImageCtors_test ImageCtors_test)

add_executable(ImageMorphology_test tests/ImageMorphology_test.cpp)
target_link_libraries(ImageMorphology_test Linx ${Boost_LIBRARIES})
add_test(ImageMorphology_test ImageMorphology_test)

add_executable(ImagePixelwise_test tests/ImagePixelwise_test.cpp)
target_link_libraries(ImagePixelwise_test Linx ${Boost_LIBRARIES})
add_test(ImagePixelwise_test ImagePixelwise_test)

add_executable(ImageRankFiltering_test tests/ImageRankFiltering_test.cpp)
target_link_libraries(ImageRankFiltering_test Linx ${Boost_LIBRARIES})
add_test(ImageRankFiltering_test ImageRankFiltering_test)

add_executable(Packs_test tests/Packs_test.cpp)
target_link_libraries(Packs_test Linx ${Boost_LIBRARIES})
add_test(Packs_test Packs_test)

add_executable(Patch_test tests/Patch_test.cpp)
target_link_libraries(Patch_test Linx ${Boost_LIBRARIES})
add_test(Patch_test Patch_test)

add_executable(PositionArithmetic_test tests/PositionArithmetic_test.cpp)
target_link_libraries(PositionArithmetic_test Linx ${Boost_LIBRARIES})
add_test(PositionArithmetic_test PositionArithmetic_test)

add_executable(Random_test tests/Random_test.cpp)
target_link_libraries(Random_test Linx ${Boost_LIBRARIES})
add_test(Random_test Random_test)

add_executable(RasterOrdering_test tests/RasterOrdering_test.cpp)
target_link_libraries(RasterOrdering_test Linx ${Boost_LIBRARIES})
add_test(RasterOrdering_test RasterOrdering_test)

add_executable(Reduction_test tests/Reduction_test.cpp)
target_link_libraries(Reduction_test Linx ${Boost_LIBRARIES})
add_test(Reduction_test Reduction_test)

add_executable(Region_test tests/Region_test.cpp)
target_link_libraries(Region_test Linx ${Boost_LIBRARIES})
add_test(Region_test Region_test)

add_executable(SequenceArithmetic_test tests/SequenceArithmetic_test.cpp)
target_link_libraries(SequenceArithmetic_test Linx ${Boost_LIBRARIES})
add_test(SequenceArithmetic_test SequenceArithmetic_test)

add_executable(SequenceCtors_test tests/SequenceCtors_test.cpp)
target_link_libraries(SequenceCtors_test Linx ${Boost_LIBRARIES})
add_test(SequenceCtors_test SequenceCtors_test)

add_executable(Slice_test tests/Slice_test.cpp)
target_link_libraries(Slice_test Linx ${Boost_LIBRARIES})
add_test(Slice_test Slice_test)

add_executable(Tiling_test tests/Tiling_test.cpp)
target_link_libraries(Tiling_test Linx ${Boost_LIBRARIES})
add_test(Tiling_test Tiling_test)

add_executable(Wrap_test tests/Wrap_test.cpp)
target_link_libraries(Wrap_test Linx ${Boost_LIBRARIES})
add_test(Wrap_test Wrap_test)
