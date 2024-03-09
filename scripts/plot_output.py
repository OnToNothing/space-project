#!/usr/bin/bash python3
"""Plots the output of the n-body program over time."""

import sys
import argparse

import numpy

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def __die(msg):
    """Exit the program with an error"""
    print(msg, file=sys.stderr)
    sys.exit(1)


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Plots the output of the n-body program over time')
    parser.add_argument('data.npy', type=argparse.FileType('rb'), help='file with the output data to plot')
    parser.add_argument('--max-points', type=int, default=100000, help='maximum number of points to plot to speed up rendering, default is 100000')
    parser.add_argument('--same-size', action='store_true', help='all bodies are plotted as the same size, regardless of mass')
    parser.add_argument('--solid-color', action='store_true', help='draw in solid colors instead of gradients')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the data file and reorganize the data
    data = numpy.load(getattr(args, 'data.npy'))
    if data.shape[1] % 3 != 0: __die("data file doesn't have 3 values per body")
    num_steps, n = data.shape[0], data.shape[1] // 3
    data = data.reshape(num_steps, n, 3)

    # Create the 3D plotting figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # If not using a solid color, darken the background
    if not args.solid_color:
        ax.xaxis.set_pane_color((0.15, 0.15, 0.15, 1.0))
        ax.yaxis.set_pane_color((0.15, 0.15, 0.15, 1.0))
        ax.zaxis.set_pane_color((0.15, 0.15, 0.15, 1.0))

    # Number of points to display per body
    points_per_body = (args.max_points + n - 1) // n
    # Number of data points to skip per body
    skip = max((num_steps + points_per_body - 1) // points_per_body, 1)

    # Setup the scatter line parameters depending on the command line arguments
    kwargs = {'s': 1}
    if not args.solid_color:
        kwargs['c'] = numpy.arange(0, num_steps, skip, dtype=float)
        kwargs['cmap'] = 'hot'
        kwargs['alpha'] = 0.5

    # Plot each body's trajectory
    for i in range(n):
        ax.scatter(data[::skip,i,0], data[::skip,i,1], data[::skip,i,2], **kwargs)
    
    # Set aspect ratio to 1
    #ax.set_aspect(1) # doesn't work with 3d plots
    limits = numpy.array((ax.get_xlim(), ax.get_ylim(), ax.get_zlim()))
    limits = limits[:,0].min(), limits[:,1].max()
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_zlim(*limits)

    # Set axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Show the plot
    if not args.solid_color:
        fig.colorbar(ax.collections[0], ax=ax, orientation='vertical')
    plt.show()


if __name__ == "__main__":
    main()
