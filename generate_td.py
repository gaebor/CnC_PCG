# -*- coding: utf-8 -*-
from brownian_sheet import *
import numpy
from argparse import ArgumentParser
import sys
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
            treetype = [1, 2, 3, 5, 6, 7, 12, 13, 16, 17][generate.fixed_random(p, n=10)]
            print("{}=T{:02d},None".format(p - 64, treetype), file=f)

    with open(filename + '.bin', 'wb') as f:
        data = numpy.zeros(64 * 64 * 2, dtype=numpy.uint8)
        data[0::2] = templates.flatten()
        data[1::2] = icons.flatten()
        f.write(bytes(data.data))


def to_tiles(M):
    """
    calculates TD template and icon enums from edge and cell data
    numbering is according to https://github.com/electronicarts/CnC_Remastered_Collection/blob/master/TIBERIANDAWN/DEFINES.H
    TemplateType enum
    """
    templates = numpy.ones((64, 64), dtype=numpy.uint8) * 0xFF
    icons = numpy.ones((64, 64), dtype=numpy.uint8) * 0xFF

    for i in range(1, M.shape[0], 2):
        for j in range(1, M.shape[1], 2):
            number_of_adjacent_rocks = 0
            if M[i - 1, j] != 0:
                number_of_adjacent_rocks += 1
            if M[i + 1, j] != 0:
                number_of_adjacent_rocks += 1
            if M[i, j - 1] != 0:
                number_of_adjacent_rocks += 1
            if M[i, j + 1] != 0:
                number_of_adjacent_rocks += 1

            template = 0xFF

            if number_of_adjacent_rocks == 1:
                if M[i + 1, j] != 0:
                    if M[i + 1, j - 1] > M[i + 1, j + 1]:
                        # 0 0
                        # +|-
                        # slope22, and a bit of padding
                        templates[i + 1, j] = 34
                        templates[i + 1, j + 1] = 34

                        icons[i + 1, j] = 0
                        icons[i + 1, j + 1] = 1
                    else:
                        # 0 0
                        # -|+
                        # slope08
                        template = 20
                elif M[i - 1, j] != 0:
                    if M[i - 1, j - 1] > M[i - 1, j + 1]:
                        # +|-
                        # 0 0
                        # slope28
                        templates[i, j] = 40
                        templates[i, j + 1] = 40
                        templates[i + 1, j + 1] = 40

                        icons[i, j] = 0
                        icons[i, j + 1] = 1
                        icons[i + 1, j + 1] = 3
                    else:
                        # -|+
                        # 0 0
                        # slope14
                        templates[i, j] = 26
                        templates[i, j + 1] = 26
                        templates[i + 1, j] = 26

                        icons[i, j] = 0
                        icons[i, j + 1] = 1
                        icons[i + 1, j] = 2
                elif M[i, j - 1] != 0:
                    if M[i - 1, j - 1] > M[i + 1, j - 1]:
                        # +_0
                        # - 0
                        # slope07
                        template = 19
                    else:
                        # -_0
                        # + 0
                        # slope21 bit with padding
                        templates[i, j] = 33
                        templates[i + 1, j] = 33

                        icons[i, j] = 0
                        icons[i + 1, j] = 1
                elif M[i, j + 1] != 0:
                    if M[i - 1, j + 1] > M[i + 1, j + 1]:
                        # 0_+
                        # 0 -
                        # slope01
                        template = 13
                    else:
                        # 0_-
                        # 0 +
                        # slope15
                        template = 27
            elif number_of_adjacent_rocks == 2:
                if M[i, j - 1] != 0 and M[i - 1, j] != 0:
                    # _|
                    if M[i - 1, j - 1] > M[i + 1, j - 1]:
                        # slope32
                        templates[i, j] = 44
                        templates[i, j + 1] = 44
                        templates[i + 1, j] = 44

                        icons[i, j] = 0
                        icons[i, j + 1] = 1
                        icons[i + 1, j] = 2
                    else:
                        # slope36
                        template = 48
                elif M[i - 1, j] != 0 and M[i, j + 1] != 0:
                    # |_
                    if M[i - 1, j + 1] > M[i - 1, j - 1]:
                        # slope29
                        template = 41
                    else:
                        # slope33
                        template = 45
                elif M[i, j + 1] != 0 and M[i + 1, j] != 0:
                    #  _
                    # |
                    if M[i + 1, j + 1] > M[i + 1, j - 1]:
                        # slope30
                        template = 42
                    else:
                        # slope34
                        template = 46
                elif M[i + 1, j] != 0 and M[i, j - 1] != 0:
                    # _
                    #  |
                    if M[i + 1, j - 1] > M[i - 1, j - 1]:
                        # slope31
                        template = 43
                    else:
                        # slope35
                        template = 47
                elif M[i + 1, j] != 0 and M[i - 1, j] != 0:
                    #  |
                    #  |
                    if M[i + 1, j - 1] > M[i + 1, j + 1]:
                        # slope24-slope26
                        template = [36, 37, 38][generate.fixed_random(i, j, 3)]
                    else:
                        # slope10-slope12
                        template = [22, 23, 24][generate.fixed_random(i, j, 3)]
                elif M[i, j - 1] != 0 and M[i, j + 1] != 0:
                    # _ _
                    if M[i + 1, j - 1] > M[i - 1, j - 1]:
                        # slope17-slope19
                        template = [29, 30, 31][generate.fixed_random(i, j, 3)]
                    else:
                        # slope03-slope05
                        template = [15, 16, 17][generate.fixed_random(i, j, 3)]
            elif number_of_adjacent_rocks == 4:
                if M[i - 1, j - 1] > M[i - 1, j + 1]:
                    # +0
                    # 0+
                    # slope37
                    template = 49
                else:
                    # 0+
                    # +0
                    # slope38
                    template = 50

            if template != 0 and template != 0xFF:
                templates[i : i + 2, j : j + 2] = template
                icons[i : i + 2, j : j + 2] = numpy.array([[0, 1], [2, 3]])

    # special cases
    # _
    #  |__

    for i in range(1, 4 * (M.shape[0] // 4) + 1, 4):
        for j in range(1, 4 * (M.shape[1] // 4) + 1, 4):
            if generate.match_mask(
                M[i - 1 : i + 4, j - 1 : j + 4],
                numpy.array(
                    [
                        [0, -1, 0, -1, 0],
                        [1, 0, -1, 0, -1],
                        [0, 1, 0, -1, 0],
                        [-1, 0, 1, 0, 1],
                        [0, -1, 0, -1, 0],
                    ]
                ),
            ) or generate.match_mask(
                M[i - 1 : i + 4, j - 1 : j + 4],
                numpy.array(
                    [
                        [0, -1, 0, -1, 0],
                        [1, 0, 1, 0, -1],
                        [0, -1, 0, 1, 0],
                        [-1, 0, -1, 0, 1],
                        [0, -1, 0, -1, 0],
                    ]
                ),
            ):
                # clear that part
                templates[i : i + 4, j : j + 4] = 0xFF
                icons[i : i + 4, j : j + 4] = 0xFF

                if M[i - 1, j - 1] > M[i + 1, j - 1]:
                    # slope02
                    templates[i : i + 3, j : j + 2] = 14
                    icons[i : i + 3, j : j + 2] = numpy.array([[0, 1], [2, 3], [4, 5]])
                    templates[i + 1 : i + 4, j + 2 : j + 4] = 14
                    icons[i + 1 : i + 4, j + 2 : j + 4] = numpy.array([[0, 1], [2, 3], [4, 5]])
                else:
                    # slope16
                    templates[i, j] = 28
                    templates[i + 1, j] = 28
                    templates[i + 1, j + 1] = 28
                    templates[i + 2, j + 1] = 28
                    icons[i, j] = 0
                    icons[i + 1, j] = 2
                    icons[i + 1, j + 1] = 3
                    icons[i + 2, j + 1] = 5

                    templates[i + 1, j + 2] = 28
                    templates[i + 2, j + 2] = 28
                    templates[i + 2, j + 3] = 28
                    templates[i + 3, j + 3] = 28
                    icons[i + 1, j + 2] = 0
                    icons[i + 2, j + 2] = 2
                    icons[i + 2, j + 3] = 3
                    icons[i + 3, j + 3] = 5
            elif generate.match_mask(
                M[i - 1 : i + 4, j - 1 : j + 4],
                numpy.array(
                    [
                        [0, -1, 0, -1, 0],
                        [-1, 0, 1, 0, 1],
                        [0, 1, 0, -1, 0],
                        [1, 0, -1, 0, -1],
                        [0, -1, 0, -1, 0],
                    ]
                ),
            ) or generate.match_mask(
                M[i - 1 : i + 4, j - 1 : j + 4],
                numpy.array(
                    [
                        [0, -1, 0, -1, 0],
                        [-1, 0, -1, 0, 1],
                        [0, -1, 0, 1, 0],
                        [1, 0, 1, 0, -1],
                        [0, -1, 0, -1, 0],
                    ]
                ),
            ):
                # clear that part
                templates[i : i + 4, j : j + 4] = 0xFF
                icons[i : i + 4, j : j + 4] = 0xFF
                if M[i + 1, j - 1] > M[i + 3, j - 1]:
                    # slope06
                    templates[i + 1, j] = 18
                    templates[i + 1, j + 1] = 18
                    templates[i + 2, j] = 18
                    templates[i + 2, j + 1] = 18
                    templates[i + 3, j] = 18
                    icons[i + 1, j] = 0
                    icons[i + 1, j + 1] = 1
                    icons[i + 2, j] = 2
                    icons[i + 2, j + 1] = 3
                    icons[i + 3, j] = 4

                    templates[i, j + 2] = 18
                    templates[i, j + 3] = 18
                    templates[i + 1, j + 2] = 18
                    templates[i + 1, j + 3] = 18
                    templates[i + 2, j + 2] = 18
                    icons[i, j + 2] = 0
                    icons[i, j + 3] = 1
                    icons[i + 1, j + 2] = 2
                    icons[i + 1, j + 3] = 3
                    icons[i + 2, j + 2] = 4
                else:
                    # slope20
                    templates[i + 1, j + 1] = 32
                    templates[i + 2, j] = 32
                    templates[i + 2, j + 1] = 32
                    templates[i + 3, j] = 32
                    templates[i + 3, j + 1] = 32
                    icons[i + 1, j + 1] = 1
                    icons[i + 2, j] = 2
                    icons[i + 2, j + 1] = 3
                    icons[i + 3, j] = 4
                    icons[i + 3, j + 1] = 5

                    templates[i, j + 3] = 32
                    templates[i + 1, j + 2] = 32
                    templates[i + 1, j + 3] = 32
                    templates[i + 2, j + 2] = 32
                    templates[i + 2, j + 3] = 32
                    icons[i, j + 3] = 1
                    icons[i + 1, j + 2] = 2
                    icons[i + 1, j + 3] = 3
                    icons[i + 2, j + 2] = 4
                    icons[i + 2, j + 3] = 5
            elif generate.match_mask(
                M[i - 1 : i + 4, j - 1 : j + 4],
                numpy.array(
                    [
                        [0, 1, 0, -1, 0],
                        [-1, 0, 1, 0, -1],
                        [0, -1, 0, 1, 0],
                        [-1, 0, -1, 0, -1],
                        [0, -1, 0, 1, 0],
                    ]
                ),
            ) or generate.match_mask(
                M[i - 1 : i + 4, j - 1 : j + 4],
                numpy.array(
                    [
                        [0, 1, 0, -1, 0],
                        [-1, 0, -1, 0, -1],
                        [0, 1, 0, -1, 0],
                        [-1, 0, 1, 0, -1],
                        [0, -1, 0, 1, 0],
                    ]
                ),
            ):
                # clear that part
                templates[i : i + 4, j : j + 4] = 0xFF
                icons[i : i + 4, j : j + 4] = 0xFF
                if M[i - 1, j - 1] > M[i - 1, j + 1]:
                    # slope27
                    templates[i : i + 2, j : j + 3] = 39
                    icons[i : i + 2, j : j + 3] = numpy.array([[0, 1, 2], [3, 4, 5]])
                    templates[i + 2 : i + 4, j + 1 : j + 4] = 39
                    icons[i + 2 : i + 4, j + 1 : j + 4] = numpy.array([[0, 1, 2], [3, 4, 5]])
                else:
                    # slope09
                    templates[i : i + 2, j : j + 3] = 21
                    icons[i : i + 2, j : j + 3] = numpy.array([[0, 1, 2], [3, 4, 5]])
                    templates[i + 2 : i + 4, j + 1 : j + 4] = 21
                    icons[i + 2 : i + 4, j + 1 : j + 4] = numpy.array([[0, 1, 2], [3, 4, 5]])
            elif generate.match_mask(
                M[i - 1 : i + 4, j - 1 : j + 4],
                numpy.array(
                    [
                        [0, -1, 0, 1, 0],
                        [-1, 0, 1, 0, -1],
                        [0, 1, 0, -1, 0],
                        [-1, 0, -1, 0, -1],
                        [0, 1, 0, -1, 0],
                    ]
                ),
            ) or generate.match_mask(
                M[i - 1 : i + 4, j - 1 : j + 4],
                numpy.array(
                    [
                        [0, -1, 0, 1, 0],
                        [-1, 0, -1, 0, -1],
                        [0, -1, 0, 1, 0],
                        [-1, 0, 1, 0, -1],
                        [0, 1, 0, -1, 0],
                    ]
                ),
            ):
                # clear that part
                templates[i : i + 4, j : j + 4] = 0xFF
                icons[i : i + 4, j : j + 4] = 0xFF
                if M[i - 1, j + 1] > M[i - 1, j + 3]:
                    # slope23
                    templates[i : i + 2, j + 1 : j + 4] = 35
                    icons[i : i + 2, j + 1 : j + 4] = numpy.array([[0, 1, 2], [3, 4, 5]])
                    templates[i + 2 : i + 4, j : j + 3] = 35
                    icons[i + 2 : i + 4, j : j + 3] = numpy.array([[0, 1, 2], [3, 4, 5]])
                else:
                    # slope13
                    templates[i : i + 2, j + 1 : j + 4] = 25
                    icons[i : i + 2, j + 1 : j + 4] = numpy.array([[0, 1, 2], [3, 4, 5]])
                    templates[i + 2 : i + 4, j : j + 3] = 25
                    icons[i + 2 : i + 4, j : j + 3] = numpy.array([[0, 1, 2], [3, 4, 5]])
    return templates, icons


def main(args):
    values = generate.main(args, to_tiles)
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
