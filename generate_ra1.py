# -*- coding: utf-8 -*-
from brownian_sheet import *
import numpy
from argparse import ArgumentParser
import sys
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
        treetype = [1, 2, 3, 5, 6, 7, 12, 13, 16, 17][generate.fixed_random(p, n=10)]
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


def to_tiles(M):
    """
    calculates RA1 template and icon enums from edge and cell data
    numbering is according to https://github.com/electronicarts/CnC_Remastered_Collection/blob/master/REDALERT/DEFINES.H
    TemplateType enum
    """
    templates = numpy.ones((128, 128), dtype=numpy.uint16) * 0xFFFF
    icons = numpy.ones((128, 128), dtype=numpy.uint8) * 0xFF

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

            template = 0xFFFF

            if number_of_adjacent_rocks == 0:
                if M[i - 1, j - 1] < 0:
                    # water2
                    template = 2
            elif number_of_adjacent_rocks == 1:
                # in this case there cannot be water
                if M[i + 1, j] != 0:
                    if M[i + 1, j - 1] > M[i + 1, j + 1]:
                        # 0 0
                        # +|-
                        # slope22, and a bit of padding
                        templates[i + 1, j] = 156
                        templates[i + 1, j + 1] = 156

                        icons[i + 1, j] = 0
                        icons[i + 1, j + 1] = 1
                    else:
                        # 0 0
                        # -|+
                        # slope08
                        template = 142
                elif M[i - 1, j] != 0:
                    if M[i - 1, j - 1] > M[i - 1, j + 1]:
                        # +|-
                        # 0 0
                        # slope150
                        templates[i, j] = 162
                        templates[i, j + 1] = 162
                        templates[i + 1, j + 1] = 162

                        icons[i, j] = 0
                        icons[i, j + 1] = 1
                        icons[i + 1, j + 1] = 3
                    else:
                        # -|+
                        # 0 0
                        # slope14
                        templates[i, j] = 148
                        templates[i, j + 1] = 148
                        templates[i + 1, j] = 148

                        icons[i, j] = 0
                        icons[i, j + 1] = 1
                        icons[i + 1, j] = 2
                elif M[i, j - 1] != 0:
                    if M[i - 1, j - 1] > M[i + 1, j - 1]:
                        # +_0
                        # - 0
                        # slope07
                        template = 141
                    else:
                        # -_0
                        # + 0
                        # slope21 bit with padding
                        templates[i, j] = 155
                        templates[i + 1, j] = 155

                        icons[i, j] = 0
                        icons[i + 1, j] = 1
                elif M[i, j + 1] != 0:
                    if M[i - 1, j + 1] > M[i + 1, j + 1]:
                        # 0_+
                        # 0 -
                        # slope01
                        template = 135
                    else:
                        # 0_-
                        # 0 +
                        # slope15
                        template = 149
            elif number_of_adjacent_rocks == 2:
                if M[i, j - 1] != 0 and M[i - 1, j] != 0:
                    # _|
                    if M[i - 1, j - 1] > M[i + 1, j - 1]:
                        if M[i + 1, j - 1] >= 0:
                            # slope32
                            templates[i, j] = 166
                            templates[i, j + 1] = 166
                            templates[i + 1, j] = 166

                            icons[i, j] = 0
                            icons[i, j + 1] = 1
                            icons[i + 1, j] = 2
                        else:
                            # watercliff32
                            template = 90
                    else:
                        if M[i - 1, j - 1] >= 0:
                            # slope36
                            template = 170
                        else:
                            # watercliff36
                            template = 94
                elif M[i - 1, j] != 0 and M[i, j + 1] != 0:
                    # |_
                    if M[i - 1, j + 1] > M[i - 1, j - 1]:
                        if M[i - 1, j - 1] >= 0:
                            # slope29
                            template = 163
                        else:
                            # watercliff29
                            template = 87
                    else:
                        if M[i - 1, j + 1] >= 0:
                            # slope33
                            template = 167
                        else:
                            # watercliff33
                            template = 91
                elif M[i, j + 1] != 0 and M[i + 1, j] != 0:
                    #  _
                    # |
                    if M[i + 1, j + 1] > M[i + 1, j - 1]:
                        if M[i + 1, j - 1] >= 0:
                            # slope30
                            template = 164
                        else:
                            # watercliff30
                            template = 88
                    else:
                        if M[i + 1, j + 1] >= 0:
                            # slope34
                            template = 168
                        else:
                            # watercliff34
                            template = 92
                elif M[i + 1, j] != 0 and M[i, j - 1] != 0:
                    # _
                    #  |
                    if M[i + 1, j - 1] > M[i - 1, j - 1]:
                        if M[i - 1, j - 1] >= 0:
                            # slope31
                            template = 165
                        else:
                            # watercliff31
                            template = 89
                    else:
                        if M[i + 1, j - 1] >= 0:
                            # slope35
                            template = 169
                        else:
                            # watercliff35
                            template = 93
                elif M[i + 1, j] != 0 and M[i - 1, j] != 0:
                    #  |
                    #  |
                    if M[i + 1, j - 1] > M[i + 1, j + 1]:
                        if M[i + 1, j + 1] >= 0:
                            # slope26
                            template = 160
                        else:
                            # watercliff26
                            template = 84
                    else:
                        if M[i + 1, j - 1] >= 0:
                            # slope12
                            template = 146
                        else:
                            # watercliff12
                            template = 70
                elif M[i, j - 1] != 0 and M[i, j + 1] != 0:
                    # _ _
                    if M[i + 1, j - 1] > M[i - 1, j - 1]:
                        if M[i - 1, j - 1] >= 0:
                            # slope17
                            template = 151
                        else:
                            # watercliff17
                            template = 75
                    else:
                        if M[i + 1, j - 1] >= 0:
                            # slope04
                            template = 138
                        else:
                            # watercliff04
                            template = 62
            elif number_of_adjacent_rocks == 4:
                if M[i - 1, j - 1] > M[i - 1, j + 1]:
                    if M[i - 1, j + 1] >= 0:
                        # +0
                        # 0+
                        # slope37
                        template = 171
                    else:
                        # +-
                        # -+
                        # watercliff37
                        template = 95
                else:
                    if M[i - 1, j - 1] >= 0:
                        # 0+
                        # +0
                        # slope38
                        template = 172
                    else:
                        # -+
                        # +-
                        # watercliff38
                        template = 96

            if template != 0 and template != 0xFFFF:
                templates[i : i + 2, j : j + 2] = template
                icons[i, j] = 0
                icons[i, j + 1] = 1
                icons[i + 1, j] = 2
                icons[i + 1, j + 1] = 3

    # special cases
    for i in range(1, 4 * (M.shape[0] // 4) + 1, 4):
        for j in range(1, 4 * (M.shape[1] // 4) + 1, 4):
            target_slice = (slice(i, i + 4), slice(j, j + 4))
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
                # _    or __
                #  |__      |_

                if M[i - 1, j - 1] > M[i + 1, j - 1]:
                    if M[i + 1, j - 1] >= 0:
                        # slope02
                        templates[target_slice] = numpy.array(
                            [
                                [136, 136, 0xFFFF, 0xFFFF],
                                [136, 136, 136, 136],
                                [136, 136, 136, 136],
                                [0xFFFF, 0xFFFF, 136, 136],
                            ]
                        )
                        icons[target_slice] = numpy.array(
                            [[0, 1, 0xFF, 0xFF], [2, 3, 0, 1], [4, 5, 2, 3], [0xFF, 0xFF, 4, 5],]
                        )
                    else:
                        # shorecliff02
                        templates[target_slice] = numpy.array(
                            [
                                [60, 60, 0xFFFF, 0xFFFF],
                                [60, 60, 60, 60],
                                [60, 60, 60, 60],
                                [1, 1, 60, 60],
                            ]
                        )
                        icons[target_slice] = numpy.array(
                            [[0, 1, 0xFF, 0xFF], [2, 3, 0, 1], [4, 5, 2, 3], [0, 0, 4, 5],]
                        )
                else:
                    if M[i - 1, j - 1] >= 0:
                        # slope16
                        templates[target_slice] = numpy.array(
                            [
                                [150, 0xFFFF, 0xFFFF, 0xFFFF],
                                [150, 150, 150, 0xFFFF],
                                [0xFFFF, 150, 150, 150],
                                [0xFFFF, 0xFFFF, 0xFFFF, 150],
                            ]
                        )
                        icons[target_slice] = numpy.array(
                            [
                                [0, 0xFF, 0xFF, 0xFF],
                                [2, 3, 0, 0xFF],
                                [0xFF, 5, 2, 3],
                                [0xFF, 0xFF, 0xFF, 5],
                            ]
                        )
                    else:
                        # shorecliff16
                        templates[target_slice] = numpy.array(
                            [
                                [74, 1, 1, 1],
                                [74, 74, 74, 1],
                                [0xFFFF, 74, 74, 74],
                                [0xFFFF, 0xFFFF, 0xFFFF, 74],
                            ]
                        )
                        icons[target_slice] = numpy.array(
                            [[0, 0, 0, 0], [2, 3, 0, 0], [0xFF, 5, 2, 3], [0xFF, 0xFF, 0xFF, 5],]
                        )
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
                #   __ or    _
                # _|      __|

                if M[i + 1, j - 1] > M[i + 3, j - 1]:
                    if M[i + 3, j - 1] >= 0:
                        # slope06 = 140
                        templates[target_slice] = numpy.array(
                            [
                                [0xFFFF, 0xFFFF, 140, 140],
                                [140, 140, 140, 140],
                                [140, 140, 140, 0xFFFF],
                                [140, 0xFFFF, 0xFFFF, 0xFFFF],
                            ]
                        )
                        icons[target_slice] = numpy.array(
                            [
                                [0xFF, 0xFF, 0, 1],
                                [0, 1, 2, 3],
                                [2, 3, 4, 0xFF],
                                [4, 0xFF, 0xFF, 0xFF],
                            ]
                        )
                    else:
                        # shorecliff06 = 64
                        templates[target_slice] = numpy.array(
                            [
                                [0xFFFF, 0xFFFF, 64, 64],
                                [64, 64, 64, 64],
                                [64, 64, 64, 1],
                                [64, 1, 1, 1],
                            ]
                        )
                        icons[target_slice] = numpy.array(
                            [[0xFF, 0xFF, 0, 1], [0, 1, 2, 3], [2, 3, 4, 0], [4, 0, 0, 0],]
                        )
                else:
                    if M[i + 1, j - 1] >= 0:
                        # slope20 = 154
                        templates[target_slice] = numpy.array(
                            [
                                [0xFFFF, 0xFFFF, 0xFFFF, 154],
                                [0xFFFF, 154, 154, 154],
                                [154, 154, 154, 154],
                                [154, 154, 0xFFFF, 0xFFFF],
                            ]
                        )
                        icons[target_slice] = numpy.array(
                            [
                                [0xFF, 0xFF, 0xFF, 1],
                                [0xFF, 1, 2, 3],
                                [2, 3, 4, 5],
                                [4, 5, 0xFF, 0xFF],
                            ]
                        )
                    else:
                        # shorecliff20 = 78
                        templates[target_slice] = numpy.array(
                            [
                                [1, 1, 1, 78],
                                [1, 78, 78, 78],
                                [78, 78, 78, 78],
                                [78, 78, 0xFFFF, 0xFFFF],
                            ]
                        )
                        icons[target_slice] = numpy.array(
                            [[0, 0, 0, 1], [0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 0xFF, 0xFF],]
                        )
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
                # |_  or |
                #   |    |_
                #   |      |

                if M[i - 1, j - 1] > M[i - 1, j + 1]:
                    if M[i - 1, j + 1] >= 0:
                        # slope27 = 161
                        templates[target_slice] = numpy.array(
                            [
                                [161, 161, 161, 0xFFFF],
                                [161, 161, 161, 0xFFFF],
                                [0xFFFF, 161, 161, 161],
                                [0xFFFF, 161, 161, 161],
                            ]
                        )
                        icons[target_slice] = numpy.array(
                            [[0, 1, 2, 0xFF], [3, 4, 5, 0xFF], [0xFF, 0, 1, 2], [0xFF, 3, 4, 5],]
                        )
                    else:
                        # shorecliff27 = 85
                        templates[target_slice] = numpy.array(
                            [
                                [85, 85, 85, 1],
                                [85, 85, 85, 1],
                                [0xFFFF, 85, 85, 85],
                                [0xFFFF, 85, 85, 85],
                            ]
                        )
                        icons[target_slice] = numpy.array(
                            [[0, 1, 2, 0], [3, 4, 5, 0], [0xFF, 0, 1, 2], [0xFF, 3, 4, 5],]
                        )
                else:
                    if M[i - 1, j - 1] >= 0:
                        # slope09 = 143
                        templates[target_slice] = numpy.array(
                            [
                                [143, 143, 143, 0xFFFF],
                                [143, 143, 143, 0xFFFF],
                                [0xFFFF, 143, 143, 143],
                                [0xFFFF, 143, 143, 143],
                            ]
                        )
                        icons[target_slice] = numpy.array(
                            [[0, 1, 2, 0xFF], [3, 4, 5, 0xFF], [0xFF, 0, 1, 2], [0xFF, 3, 4, 5],]
                        )
                    else:
                        # shorecliff09 = 67
                        templates[target_slice] = numpy.array(
                            [
                                [67, 67, 67, 0xFFFF],
                                [67, 67, 67, 0xFFFF],
                                [1, 67, 67, 67],
                                [1, 67, 67, 67],
                            ]
                        )
                        icons[target_slice] = numpy.array(
                            [[0, 1, 2, 0xFF], [3, 4, 5, 0xFF], [0, 0, 1, 2], [0, 3, 4, 5],]
                        )
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
                #  _| or |
                # |     _|
                # |    |
                if M[i - 1, j + 1] > M[i - 1, j + 3]:
                    if M[i - 1, j + 3] >= 0:
                        # slope23 = 157
                        templates[target_slice] = numpy.array(
                            [
                                [0xFFFF, 157, 157, 157],
                                [0xFFFF, 157, 157, 157],
                                [157, 157, 157, 0xFFFF],
                                [157, 157, 157, 0xFFFF],
                            ]
                        )
                        icons[target_slice] = numpy.array(
                            [[0xFF, 0, 1, 2], [0xFF, 3, 4, 5], [0, 1, 2, 0xFF], [3, 4, 5, 0xFF],]
                        )
                    else:
                        # shorecliff23 = 81
                        templates[target_slice] = numpy.array(
                            [
                                [0xFFFF, 81, 81, 81],
                                [0xFFFF, 81, 81, 81],
                                [81, 81, 81, 1],
                                [81, 81, 81, 1],
                            ]
                        )
                        icons[target_slice] = numpy.array(
                            [[0xFF, 0, 1, 2], [0xFF, 3, 4, 5], [0, 1, 2, 0], [3, 4, 5, 0],]
                        )
                else:
                    if M[i - 1, j + 1] >= 0:
                        # slope13 = 147
                        templates[target_slice] = numpy.array(
                            [
                                [0xFFFF, 147, 147, 147],
                                [0xFFFF, 147, 147, 147],
                                [147, 147, 147, 0xFFFF],
                                [147, 147, 147, 0xFFFF],
                            ]
                        )
                        icons[target_slice] = numpy.array(
                            [[0xFF, 0, 1, 2], [0xFF, 3, 4, 5], [0, 1, 2, 0xFF], [3, 4, 5, 0xFF],]
                        )
                    else:
                        # shorecliff13 = 71
                        templates[target_slice] = numpy.array(
                            [
                                [1, 71, 71, 71],
                                [1, 71, 71, 71],
                                [71, 71, 71, 0xFFFF],
                                [71, 71, 71, 0xFFFF],
                            ]
                        )
                        icons[target_slice] = numpy.array(
                            [[0, 0, 1, 2], [0, 3, 4, 5], [0, 1, 2, 0xFF], [3, 4, 5, 0xFF],]
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
        description='Generate a random CnC (RA1) map.\n' 'Author: Gábor Borbély (gaebor)\n',
    )
    parser.add_argument(
        'output', type=str, default="", nargs='?', help="output map filename (without extension)"
    )

    parser.add_argument('-n', type=int, default=64, help="number of rows in the generated map")
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
        help="minimum height difference between contour lines: dhbase*2^dh.\n"
        "If one parameter is given then it should be a non-negative integer.\n"
        "If two parameters are given then floor(32*sigmoid(a*X+b)) is used\n"
        "where X is a random noise.",
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
        default=[0.1],
        metavar='param',
        nargs='+',
        help="Sets when to break a rockface.\n"
        "If one argument is given, then rock is deleted with uniform probability 'r' (threshold a Poisson noise).\n"
        "If two arguments are given, then a rock is deleted with probability"
        " 'sigmoid(a*X+b)'\nwhere X is a random noise.",
    )

    parser.add_argument(
        "-m",
        "--mine",
        "--resource",
        dest="resource",
        type=float,
        default=[1, -6],
        nargs='+',
        help="Sets when to place a resource field/mine.\n"
        "If one parameter is given, then mines are placed with uniform probability 'm' (threshold a Poisson noise).\n"
        "If two arguments are given, then with probability"
        " 'sigmoid(a*X+b)'\nwhere X is a random noise.",
    )

    parser.add_argument(
        "-T",
        "--tree",
        "--terrain",
        dest="tree",
        type=float,
        default=[1, -6],
        nargs='+',
        help='When to place trees.',
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
