from argparse import ArgumentParser
import numpy as np
from scipy import signal, ndimage
import time


def make_2d(side):
    image = np.empty([side, side], dtype=int)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = i+j
    return image


def print_2d(name, image):
    print(f"{name}:")
    print(f"  {image.shape[1]} x {image.shape[0]}")
    print(f"  [{image[0,0]}, ... , {image[-1,-1]}]")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--image", type=int, default=2048)
    parser.add_argument("--kernel", type=int, default=5)
    parser.add_argument("--extrapolation", type=str, default=None)
    args = parser.parse_args()

    print("Generating input and kernel...")
    input = make_2d(args.image)
    print_2d("input", input)
    kernel = make_2d(args.kernel)
    print_2d("kernel", kernel)

    method = args.extrapolation
    print("Filtering...")
    if method == None:
        start = time.perf_counter()
        output = signal.correlate(input, kernel, mode="valid")
        stop = time.perf_counter()
    else:
        start = time.perf_counter()
        output = ndimage.correlate(input, kernel, mode=method)
        stop = time.perf_counter()
    print(f"  Done in {stop-start} s")
    print_2d("output", output)
