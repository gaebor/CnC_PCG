# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import sys

import numpy

from lcw import MyFormatter
import generate


def mapwrite(templates, icons, tibtrees=[], trees=[], filename='map', width=62, height=62):
    assert templates.shape == icons.shape
    assert templates.shape == (64, 64)

    with open(filename + '.ini', 'w') as f:
        print("[Map]", file=f)
        print("X={}\nY={}\nWidth={}\nHeight={}".format(1, 1, width, height), file=f)
        print("Theater=temperate", file=f)

        print("", file=f)
        print("[Waypoints]", file=f)
        for w in range(6):
            print("{}={}".format(w, w + 65), file=f)

        print("", file=f)
        print("[Terrain]", file=f)
        for p in tibtrees:
            print("{}=split2,None".format(p - 64), file=f)
        for p in trees:
            treetype = [1, 2, 3, 5, 6, 7, 12, 13, 16, 17][generate.fixed_random(p) % 10]
            print("{}=T{:02d},None".format(p - 64, treetype), file=f)

    with open(filename + '.bin', 'wb') as f:
        data = numpy.zeros(64 * 64 * 2, dtype=numpy.uint8)
        data[0::2] = templates.flatten()
        data[1::2] = icons.flatten()
        f.write(bytes(data.data))


tile_patterns = generate.import_tiles_file('td_tiles.json')


def main(args):
    empty_templates = numpy.ones((64, 64), dtype=numpy.uint8) * 0xFF
    empty_icons = numpy.ones((64, 64), dtype=numpy.uint8) * 0xFF

    values = generate.main(args, empty_templates, empty_icons, tile_patterns)
    M = values[0]
    values = values[1:]

    if args.format == 'html':
        if args.output == '':
            outf = sys.stdout
        else:
            outf = open(args.output + '.html', "w")
        print(generate.html(M, width=args.width, hue=args.hue), file=outf)
    elif args.format == 'inibin':
        mapwrite(*values, filename=args.output, width=M.shape[1] - 1, height=M.shape[0] - 1)
    return 0


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=MyFormatter,
        description="""
Generate a random CnC (TD) map.
Author: Gábor Borbély (gaebor)

If a parameter is called 'random parameter' then it means that it can modify the density of said feature.
Such a parameter can have one or two values.
 * If one value ('a') is specified then it acts as a threshold of a uniform Poisson noise:
    f(x) = rand() < a
 * If two values are specified ('a' and 'b') then the noise can have a non-uniform distribution.
    f(x) = rand() < sigmoid(a*B(x)+b)
   where B is a Brownian noise.
""",
    )
    parser.add_argument(
        'output', type=str, default="", nargs='?', help="output map filename (without extension)"
    )

    parser.add_argument('-n', type=int, default=32, help="number of rows in the generated map")
    parser.add_argument('-H', type=float, default=0.7, help="Hurst parameter (alpha/2)")
    parser.add_argument(
        '-w',
        "--width",
        dest="width",
        type=int,
        default=15,
        help="Size of one map tile in pixel, only in html format.",
    )
    parser.add_argument(
        '-s',
        "--seed",
        dest="seed",
        type=int,
        default=-1,
        help="Random seed, if negative then random seed is random.",
    )
    parser.add_argument(
        "-dh",
        dest="dh",
        type=float,
        nargs='+',
        default=[3],
        help="'random parameter'\nminimum height difference between contour lines: dhbase*2^dh.\n"
        "If one parameter is given then it should be a non-negative integer.\n"
        "If two parameters are given then dh(x)=floor(32*sigmoid(a*B(x)+b))",
    )
    parser.add_argument(
        "--dhbase",
        dest="dhbase",
        type=float,
        default=0.05,
        help="minimum height difference to consider a 'step' in height.",
    )
    parser.add_argument(
        "-r",
        "--rock",
        dest="rockface",
        type=float,
        default=[0.2],
        metavar='param',
        nargs='+',
        help="'random parameter'\nDelete from an otherwise continuous cliff.",
    )

    parser.add_argument(
        "--tiberium",
        "--resource",
        dest="resource",
        type=float,
        default=[1, -5],
        nargs='+',
        help="'random parameter'\nWhen to place a tiberium tree.",
    )

    parser.add_argument(
        "-T",
        "--tree",
        "--terrain",
        dest="tree",
        type=float,
        default=[1, -4],
        nargs='+',
        help="'random parameter'\nWhen to place trees.",
    )

    parser.add_argument(
        "-t",
        "--type",
        dest="type",
        type=str,
        default="simple",
        choices=["brownian", "perlin", "simple"],
        help='Type of the noise to use.',
    )
    parser.add_argument("-hue", "--hue", dest="hue", type=int, default=4)
    parser.add_argument(
        "-o",
        "--offset",
        dest="offset",
        type=float,
        default=0.5,
        help="height offset of map (elevation)",
    )

    parser.add_argument(
        "-f",
        "--format",
        dest="format",
        type=str,
        default="inibin",
        choices=['html', 'inibin'],
        help='Output format.',
    )

    sys.exit(main(parser.parse_args()))
