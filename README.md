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
In return, execution is automatically parallelized by Kokkos, including on GPU.

In addition, for interfacing with libraries which require contiguity, `Raster` is a row-major ordered alternative to `Image`.
It is a standard range (providing `begin()` and `end()`) which eases interfacing with the standard library.
`Image` and `Raster` are also compatible with `std::mdspan`.

Data classes have shared pointer semantics, so that copy is shallow by default.
Deep copy has to be explicit:

```cpp
auto a = Linx::Image(...);
auto b = a;
b *= 2; // Modifies a and b
auto c = +a; // Copies
c *= 2; // Modifies c only
``` 

**Element-wise transforms**

Data classes offer a variety of element-wise services which can either modify the data in-place or return new instances or values.
In-place services are methods, such as `Image::exp()`, while new-instance services are free functions, such as `exp(const Image&)`:

```cpp
auto a = Linx::Image(...);
a.pow(2); // Modifies a in-place
auto a2 = Linx::pow(a, 2); // Creates new instance
auto norm2 = Linx::norm<2>(a); // Return value
```

Arbitrarily complex functions can also be applied element-wise with `apply()` or `generate()`:

```cpp
auto a = Linx::Image(...);
auto b = Linx::Image(...);
a.generate(
    "random noise",
    Linx::GaussianNoise());
a.apply(
    "logistic function",
    KOKKOS_LAMBDA(auto a_i) { return 1. / (1. + std::exp(-a_i)); });
```

Both methods accept auxiliary data containers as function parameters:

```cpp
auto a = Linx::Image(...);
auto b = Linx::Image(...);
auto c = Linx::Image(...);
a.generate(
    "geometric mean",
    KOKKOS_LAMBDA(auto b_i, auto c_i) { return std::sqrt(b_i * c_i); },
    b, c);
```

**Global transforms**

Global transforms such as Fourier transforms and convolutions are also supported.
They return new instances by default.
Function suffixed with `_to` fill an existing container instead:

```cpp
auto image = Linx::Image(...);
auto kernel = Linx::Image(...);
auto filtered = Linx::correlate("filter", image, kernel); // Creates a new instance
auto fourier = Linx::Image("DFT", image.shape());
Linx::dft_to(image, fourier); // Fills fourier
```

**Regional transforms**

There are two ways to work on subsets of elements:
* by slicing some data classes with `slice()`, which return a view of type `Sequence` or `Image` depending on the input type;
* by associating a `Region` to a data class with `patch()`, which results in an object of type `Patch`.

Patches are extremely lightweight and can be moved around when the region is a `Window`, i.e. has translation capabilities.

Typical windows are `Box`, `Mask` or `Path` and can be used to apply filters.

Patches are like data classes and can themselves be transformed element-wise:

```cpp
auto image = Linx::Image(...):
auto region = Linx::Box(...);
auto patch = Linx::patch(image, region);
patch.exp(); // Modifies image elements inside region
```

**Labels**

Most data classes and services are labeled for logging or debugging purposes, thanks to some `std::string` parameter.
As demonstrated in the snippets above, this also helps documenting the code,
which is why the parameter is purposedly mandatory most of the time.

Only when natural, labelling is automated, typically when calling simple functions:

```cpp
auto a = Linx::Image("a", ...);
auto b = Linx::sin(a);
assert(b.label() == "sin(a)");
```
