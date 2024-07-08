from argparse import ArgumentParser
import numpy as np
from scipy import signal, ndimage
import time

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--image", type=int, default=2048)
    parser.add_argument("--kernel", type=int, default=5)
    parser.add_argument("--extrapolation", type=str, default=None)
    args = parser.parse_args()

    print("Generating input and kernel...")
    input = np.ones([args.image, args.image], dtype=int)
    print(input)
    kernel = np.ones([args.kernel, args.kernel], dtype=int)
    print(kernel)

    method = args.extrapolation
    print("Filtering...")
    if method == None:
        start = time.perf_counter()
        output = signal.convolve(input, kernel, mode="valid")
        stop = time.perf_counter()
    else:
        start = time.perf_counter()
        output = ndimage.convolve(input, kernel, mode=method)
        stop = time.perf_counter()
    print(f"  Done in {stop-start} s")
    print(output)
