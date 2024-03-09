#!/usr/bin/bash python3
"""Compares 2 npy files to make sure they are (almost) equal."""

import sys
import argparse

import numpy


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Compares 2 npy files to make sure that they are (almost) equal')
    parser.add_argument('a.npy', type=argparse.FileType('rb'), help='first file to compare')
    parser.add_argument('b.npy', type=argparse.FileType('rb'), help='second file to compare')
    parser.add_argument('--exact', action='store_true', help='must be exactly equal instead of close')
    parser.add_argument('--abs-tol', type=float, help='absolute tolerance when measuring closeness (default=1e-8)', default=1e-8)
    parser.add_argument('--rel-tol', type=float, help='relative tolerance when measuring closeness (default=1e-5)', default=1e-5)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the matrices
    a = numpy.load(getattr(args, 'a.npy'))
    b = numpy.load(getattr(args, 'b.npy'))

    # Check matrices
    if a.shape != b.shape:
        print("unequal shapes:", a.shape, b.shape)
        sys.exit(2)
    if not args.exact and numpy.allclose(a, b, atol=args.abs_tol, rtol=args.rel_tol):
        if numpy.all(a == b):
            print("equal")
        else:
            print("all-close")
        sys.exit(0)
    elif args.exact and numpy.all(a == b):
        print("equal")
        sys.exit(0)
    else:
        print("not equal/allclose")
        a_isnan, b_isnan = numpy.isnan(a), numpy.isnan(b)
        if a_isnan.any(): print("a has", a_isnan.sum(), "NANs")
        if b_isnan.any(): print("b has", b_isnan.sum(), "NANs")
        num_close = numpy.isclose(a, b, atol=args.abs_tol, rtol=args.rel_tol).sum()
        print("there are %d (%.2f%%) close values" % (num_close, num_close*100/a.size))
        diff = abs(a - b)
        diff_max_loc = numpy.unravel_index(numpy.argmax(diff), a.shape)
        print("max absolute difference is", diff.max(), "at", diff_max_loc, "with", a[diff_max_loc], "and", b[diff_max_loc])
        maxes = numpy.maximum(abs(a), abs(b))
        maxes_zero = maxes == 0
        maxes[maxes_zero] = 1
        relative = diff / maxes
        relative[maxes_zero] = 0
        rel_max_loc = numpy.unravel_index(numpy.argmax(relative), a.shape)
        print("max relative difference is", relative.max(), "at", rel_max_loc, "with", a[rel_max_loc], "and", b[rel_max_loc])
        sys.exit(1)


if __name__ == "__main__":
    main()
