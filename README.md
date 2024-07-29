# KokkosTest

Writing a few lines of code to learn on Kokkos...

## License

[Apache-2.0](LICENSE)

## Build

Build Kokkos:

```sh
cd <kokkos_clone_dir>
git clone https://github.com/kokkos/kokkos.git

mkdir <kokkos_build_dir>
cd <kokkos_build_dir>
cmake <kokkos_clone_dir>/kokkos -DCMAKE_CXX_COMPILER=g++ -DCMAKE_INSTALL_PREFIX=<kokkos_install_dir>
make install
```

Build the tests:

```sh
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=<kokkos_install_dir>
make
make test
```

## Design concepts

**Data classes**

There are two main data classes: `Sequence` for 1D data, and `Image` for ND data.
Underlying storage is handled by Kokkos by default, and adapts to the target infrastructure.
There is no ordering or contiguity guaratee.

In addition, for interfacing with libraries which require contiguity, `Raster` is a row-major ordered alternative to `Image`.
It is a standard range (providing `begin()` and `end()`) which eases interfacing with the standard library.
`Image` and `Raster` are also compatible with `std::mdrange`.

Many data classes and services are labeled for logging or debugging purposes, thanks to some `std::string` parameter.

**Element-wise transforms**

Data classes offer a variety of element-wise services which can either modify the data in-place or return new instances or values.
In-place services are methods, such as `Image::exp()`, while new-instance services are free functions, such as `exp(const Image&)`:

```cpp
auto image = Linx::Image(...);
image.exp(); // Modifies image in-place
auto exp = Linx::exp(image); // Creates new instance
```

Arbitrarily complex functions can also be applied element-wise with `apply()` or `generate()`:

```cpp
auto a = Linx::Image(...);
auto b = Linx::Image(...);
a.generate(Linx::GaussianNoise()); // Generate iid. Gaussian noise
a.apply([](auto a_i) { return 1. / (1. + std::exp(-a_i)); }); // Apply logistic curve
```

Both methods accept auxiliary data containers as function parameters:

```cpp
auto a = Linx::Image(...);
auto b = Linx::Image(...);
auto c = Linx::Image(...);
a.generate(KOKKOS_LAMBDA(auto b_i, auto c_i) { return std::sqrt(b_i * c_i); }, b, c); // Compute geometric mean
```

**Global transforms**

Global transforms such as Fourier transforms and convolutions are also supported and systematically return new instances.

**Regional transforms**

There are two ways to work on subsets of elements:
* by slicing some data classes with `slice()`, which return a view of type `Sequence` or `Image` depending on the input type;
* by associating a `Region` to a data class, which results in an object of type `Patch`.

Patches are extremely lightweight and can be moved around when the region is a `Window`, i.e. has translation capabilities.

Typical windows are `Box`, `Mask` or `Path` and can be used to apply filters.

Patches are data classes and can themselves be transformed element-wise:

```cpp
auto image = Linx::Image(...):
auto region = Linx::Box(...);
auto patch = Linx::patch(image, region);
patch.exp(); // Modifies image elements inside region
```
