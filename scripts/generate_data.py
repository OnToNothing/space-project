#!/usr/bin/bash python3
"""Generates a set of data for the nbody program."""

import sys
import argparse

import numpy


def __die(msg):
    """Exit the program with an error"""
    print(msg, file=sys.stderr)
    sys.exit(1)


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Generates a set of data for the nbody program')
    parser.add_argument('n', type=int, help='number of bodies/particles to create')
    parser.add_argument('output.npy', type=argparse.FileType('wb'), help='file to write the n-by-5 matrix to')
    parser.add_argument('--min-mass', type=float, help='the minimum mass of a body in kg, default is 0.1 (must be >0)', default=0.1)
    parser.add_argument('--max-mass', type=float, help='the maximum mass of a body in kg, default is 1.0 (must be >=min-mass)', default=1.0)
    parser.add_argument('--mass', type=float, help='the fixed mass of all bodies in kg, overrides min-mass and max-mass (must be >0)')
    parser.add_argument('--min-position', type=float, help='the minimum position of a body in m/s, default is -1.0', default=-1.0)
    parser.add_argument('--max-position', type=float, help='the maximum position of a body in m/s, default is 1.0 (must be >min-position)', default=1.0)
    parser.add_argument('--min-velocity', type=float, help='the minimum velocity of a body in m/s, default is 0.0 (must be >=0)', default=0.0)
    parser.add_argument('--max-velocity', type=float, help='the maximum velocity of a body in m/s, default is 1.0 (must be >=min-velocity)', default=1.0)
    parser.add_argument('--velocity', type=float, help='the fixed velocity of all bodies in m/v, overrides --min-velocity and --max-velocity (must be >=0)')

    # Parse and check the command-line arguments
    args = parser.parse_args()
    if args.min_mass <= 0: __die("min-mass must be positive")
    if args.max_mass < args.min_mass: __die("max-mass must be greater than or equal to the min-mass")
    if args.mass is not None and args.mass <= 0: __die("mass must be positive")
    if args.max_position <= args.min_position: __die("max-position must be greater than the min-position")
    if args.min_velocity < 0: __die("min-velocity must be non-negative")
    if args.max_velocity < args.min_velocity: __die("max-velocity must be greater than or equal to the min-velocity")
    if args.velocity is not None and args.velocity < 0: __die("velocity must be non-negative")

    # The number of bodies and create the random number generator
    n = args.n
    rng = numpy.random.default_rng()

    # Generate random masses
    masses = (
        rng.uniform(args.min_mass, args.max_mass, (n,1)) if args.mass is None else
        numpy.ones((n,1)) * args.mass # constant masses
    )

    # Generate random positions
    positions = rng.uniform(args.min_position, args.max_position, (n,3))

    # Generate random velocties
    if args.velocity is not None or args.velocity != 0:
        # Even constant velocity has random orientation
        unit = rng.multivariate_normal([0,0,0], [[1,0,0],[0,1,0],[0,0,1]], (n,))
        unit /= numpy.linalg.norm(unit, axis=1, keepdims=True)
        velocities = (
            rng.uniform(args.min_velocity, args.max_velocity, (n,1)) if args.velocity is None else
            numpy.ones((n,1)) * args.velocity # constant velocity
        ) * unit
    else:
        # Constant-zero masses
        velocities = numpy.zeros((n, 3))

    # Combine all of the values into a single matrix and save it
    output = numpy.hstack((masses, positions, velocities))
    numpy.save(getattr(args, "output.npy"), output)


if __name__ == "__main__":
    main()
