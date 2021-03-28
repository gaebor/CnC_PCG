# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import sys
from functools import partial as bind

import numpy

from brownian_sheet import *
from lcw import base64write, LCWpack, MyFormatter
import generate


def mapwrite(templates, icons, mines=[], trees=[], f=sys.stdout, width=126, height=126):
    assert templates.shape == icons.shape
    assert templates.shape == (128, 128)
    print("[Basic]", file=f)
    print("NewINIFormat=3", file=f)
    print("", file=f)
    print("[Map]", file=f)
    print("X={}\nY={}\nWidth={}\nHeight={}".format(1, 1, width, height), file=f)
    print("Theater=temperate", file=f)

    print("", file=f)
    print("[Waypoints]", file=f)
    for w in range(8):
        print("{}={}".format(w, w + 129), file=f)
    print("", file=f)

    print("[TERRAIN]", file=f)
    for p in mines:
        print("{}=MINE".format(p), file=f)
    for p in trees:
        treetype = [1, 2, 3, 5, 6, 7, 12, 13, 16, 17][generate.fixed_random(p) % 10]
        print("{}=T{:02d}".format(p - 128, treetype), file=f)

    print("[MapPack]", file=f)
    base64write(
        LCWpack(bytes(templates[:32].data))
        + LCWpack(bytes(templates[32:64].data))
        + LCWpack(bytes(templates[64:96].data))
        + LCWpack(bytes(templates[96:].data))
        + LCWpack(bytes(icons[:64].data))
        + LCWpack(bytes(icons[64:].data)),
        f,
    )
    print("", file=f)
    print("[OverlayPack]\n1=BwAAIIH//v8f/4AHAAAggf/+/x//gA==", file=f)


tile_patterns = generate.import_tiles_file('ra1_tiles.json')


def to_tiles(M):
    """
    calculates RA1 template and icon enums from edge and cell data
    """
    templates = numpy.ones((128, 128), dtype=numpy.uint16) * 0xFFFF
    icons = numpy.ones((128, 128), dtype=numpy.uint8) * 0xFF

    for i in range(1, M.shape[0] - 1, 2):
        for j in range(1, M.shape[1] - 1, 2):
            target_slice = (slice(i, i + 2), slice(j, j + 2))
            match = bind(generate.match_pattern, map=M[i - 1 : i + 2, j - 1 : j + 2])
            pseudo_random_seed = generate.fixed_random(i, j)

            for patterns, templates_replace, icons_replace in tile_patterns:
                if patterns.shape[1:] == (3, 3) and templates_replace.shape[1:] == (2, 2):
                    if any(match(pattern) for pattern in patterns):
                        new_templates, new_icons = generate.get_pseudo_random_choice(
                            templates_replace, icons_replace, pseudo_random_seed
                        )
                        templates[target_slice] = new_templates
                        icons[target_slice] = new_icons
                        break

    for i in range(1, M.shape[0] - 3, 4):
        for j in range(1, M.shape[1] - 3, 4):
            target_slice = (slice(i, i + 4), slice(j, j + 4))
            match = bind(generate.match_pattern, map=M[i - 1 : i + 4, j - 1 : j + 4])
            pseudo_random_seed = generate.fixed_random(i, j)

            for patterns, templates_replace, icons_replace in tile_patterns:
                if patterns.shape[1:] == (5, 5) and templates_replace.shape[1:] == (4, 4):
                    if any(match(pattern) for pattern in patterns):
                        new_templates, new_icons = generate.get_pseudo_random_choice(
                            templates_replace, icons_replace, pseudo_random_seed
                        )
                        templates[target_slice] = new_templates
                        icons[target_slice] = new_icons
                        break

    for i in range(3, M.shape[0] - 5, 4):
        for j in range(3, M.shape[1] - 5, 4):
            target_slice = (slice(i, i + 4), slice(j, j + 4))
            if generate.match_pattern(
                numpy.array(
                    [
                        [0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, -1, 0, 0, 0],
                        [1, 0, 1, 0, 0, 0, -1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, -1, 0, 0, 0, -1, 0, 0],
                        [0, 0, 0, -1, 0, -1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ),
                M[i - 3 : i + 6, j - 3 : j + 6],
            ):
                if M[i - 1, j - 1] >= 0 and M[i - 1, j + 1] < 0:
                    # shore49 = 51
                    templates[target_slice] = numpy.array(
                        [[51, 51, 51, 1], [51, 51, 51, 1], [51, 51, 51, 1], [1, 1, 1, 1],]
                    )
                    icons[target_slice] = numpy.array(
                        [[0, 1, 2, 0], [3, 4, 5, 0], [6, 7, 8, 0], [0, 0, 0, 0],]
                    )
    return templates, icons


def main(args):
    values = generate.main(args, to_tiles)
    M = values[0]
    values = values[1:]
    if args.output == '':
        outf = sys.stdout
    else:
        outf = open(args.output + '.' + args.format, "w")

    if args.format == 'html':
        print(generate.html(M, width=args.width, hue=args.hue), file=outf)
    elif args.format == 'mpr':
        mapwrite(*values, width=M.shape[1] - 1, height=M.shape[0] - 1, f=outf)
    return 0


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=MyFormatter,
        description="""
Generate a random CnC (RA1) map.
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

    parser.add_argument('-n', type=int, default=64, help="number of rows in the generated map")
    parser.add_argument('-H', type=float, default=0.85, help="Hurst parameter (alpha/2)")
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
        "-m",
        "--mine",
        "--resource",
        dest="resource",
        type=float,
        default=[1, -6],
        nargs='+',
        help="'random parameter'\nwhen to place a resource field/mine",
    )

    parser.add_argument(
        "-T",
        "--tree",
        "--terrain",
        dest="tree",
        type=float,
        default=[1, -5],
        nargs='+',
        help="'random parameter'\nWhen to place trees.",
    )

    parser.add_argument(
        "-t",
        "--type",
        dest="type",
        type=str,
        default="brownian",
        choices=["brownian", "perlin", "simple"],
        help='Type of the noise to use.',
    )
    parser.add_argument("-hue", "--hue", dest="hue", type=int, default=4)
    parser.add_argument(
        "-o",
        "--offset",
        dest="offset",
        type=float,
        default=0,
        help="height offset of map (elevation)",
    )

    parser.add_argument(
        "-f",
        "--format",
        dest="format",
        type=str,
        default="mpr",
        choices=['html', 'mpr'],
        help='Output format.',
    )

    sys.exit(main(parser.parse_args()))
