from brownian_sheet import *
import numpy
from argparse import ArgumentParser
import sys
from lcw import MyFormatter
import generate

def mapwrite(templates, icons, tibtrees=[], filename='map', width=62, height=62):
    assert(templates.shape == icons.shape)
    assert(templates.shape == (64, 64))
    
    with open(filename + '.ini', 'w') as f:
        print("[Map]", file=f)
        print("X={}\nY={}\nWidth={}\nHeight={}".format(1,1, width, height), file=f)
        print("Theater=temperate", file=f)
        
        print("", file=f)
        print("[Waypoints]", file=f)
        for w in range(6):
            print("{}={}".format(w, w+65), file=f)
        
        print("", file=f)
        print("[Terrain]", file=f)
        for p in tibtrees:
            print("{}=split2,None".format(p-64), file=f)

    with open(filename + '.bin', 'wb') as f:
        data = numpy.zeros(64*64*2, dtype=numpy.uint8)
        data[0::2] = templates.flatten()
        data[1::2] = icons.flatten()
        f.write(bytes(data.data))
        
def to_tiles(M):
    """
    calculates TD template and icon enums from edge and cell data
    numbering is according to https://github.com/electronicarts/CnC_Remastered_Collection/blob/master/TIBERIANDAWN/DEFINES.H
    TemplateType enum
    """
    templates = numpy.ones((64,64), dtype=numpy.uint8)*0xFF
    icons = numpy.ones((64,64), dtype=numpy.uint8)*0xFF
    
    # TODO remove water elements and renumber according to 
    for i in range(1, M.shape[0], 2):
        for j in range(1, M.shape[1], 2):
            number_of_adjacent_rocks = 0
            if M[i-1,j] != 0:
                number_of_adjacent_rocks += 1
            if M[i+1,j] != 0:
                number_of_adjacent_rocks += 1
            if M[i,j-1] != 0:
                number_of_adjacent_rocks += 1
            if M[i,j+1] != 0:
                number_of_adjacent_rocks += 1
            
            template = 0xFF
            
            if number_of_adjacent_rocks == 1:
                if M[i+1,j] != 0:
                    if M[i+1,j-1] > M[i+1,j+1]:
                        # 0 0
                        # +|-
                        # slope22, and a bit of padding
                        templates[i+1,j] = 34
                        templates[i+1,j+1] = 34
                        
                        icons[i+1,j] = 0
                        icons[i+1,j+1] = 1
                    else:
                        # 0 0
                        # -|+
                        # slope08
                        template = 20
                elif M[i-1,j] != 0:
                    if M[i-1,j-1] > M[i-1,j+1]:
                        # +|-
                        # 0 0
                        # slope28
                        templates[i,j] = 40
                        templates[i,j+1] = 40
                        templates[i+1,j+1] = 40
                        
                        icons[i,j] = 0
                        icons[i,j+1] = 1
                        icons[i+1,j+1] = 3
                    else:
                        # -|+
                        # 0 0
                        # slope14
                        templates[i,j] = 26
                        templates[i,j+1] = 26
                        templates[i+1,j] = 26
                        
                        icons[i,j] = 0
                        icons[i,j+1] = 1
                        icons[i+1,j] = 2
                elif M[i,j-1] != 0:
                    if M[i-1,j-1] > M[i+1,j-1]:
                        # +_0
                        # - 0
                        # slope07
                        template = 19
                    else:
                        # -_0
                        # + 0
                        # slope21 bit with padding
                        templates[i,j] = 33
                        templates[i+1,j] = 33
                        
                        icons[i,j] = 0
                        icons[i+1,j] = 1                        
                elif M[i,j+1] != 0:
                    if M[i-1,j+1] > M[i+1,j+1]:
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
                if M[i,j-1] != 0 and M[i-1,j] != 0:
                    # _|
                    if M[i-1,j-1] > M[i+1,j-1]:
                        # slope32
                        templates[i,j] = 44
                        templates[i,j+1] = 44
                        templates[i+1,j] = 44
                        
                        icons[i,j] = 0
                        icons[i,j+1] = 1
                        icons[i+1,j] = 2
                    else:
                        # slope36
                        template = 48
                elif M[i-1,j] != 0 and M[i,j+1] != 0:
                    # |_
                    if M[i-1,j+1] > M[i-1,j-1]:
                        # slope29
                        template = 41
                    else:
                        # slope33
                        template = 45
                elif M[i,j+1] != 0 and M[i+1,j] != 0:
                    #  _
                    # |
                    if M[i+1,j+1] > M[i+1,j-1]:
                        # slope30
                        template = 42
                    else:
                        # slope34
                        template = 46
                elif M[i+1,j] != 0 and M[i,j-1] != 0:
                    # _ 
                    #  |
                    if M[i+1,j-1] > M[i-1,j-1]:
                        # slope31
                        template = 43
                    else:
                        # slope35
                        template = 47
                elif M[i+1,j] != 0 and M[i-1,j] != 0:
                    #  |
                    #  |
                    if M[i+1,j-1] > M[i+1,j+1]:
                        # slope26
                        template = 38
                    else:
                        # slope12
                        template = 24
                elif M[i,j-1] != 0 and M[i,j+1] != 0:
                    # _ _
                    if M[i+1,j-1] > M[i-1,j-1]:
                        # slope17
                        template = 29
                    else:
                        # slope04
                        template = 16
            elif number_of_adjacent_rocks == 4:
                if M[i-1,j-1] > M[i-1,j+1]:
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
                templates[i:i+2,j:j+2] = template
                icons[i,j] = 0
                icons[i,j+1] = 1
                icons[i+1,j] = 2
                icons[i+1,j+1] = 3

    return templates, icons
    
def main(args):
    M, templates, icons, resource_positions = generate.main(args, to_tiles)
    
    if args.format == 'html':
        print(generate.html(M, width=args.width, hue=args.hue))
    elif args.format == 'inibin':
        mapwrite(templates, icons, resource_positions, 
                 filename=args.output, width=M.shape[1]-1, height=M.shape[0]-1)
    return 0

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=MyFormatter,
                        description='Generate a random CnC (TD) map.\n'
                        'Author: Gabor Borbely (gaebor)\n')
    parser.add_argument("output", type=str,
                        help="output filename, .ini and .bin files will be generated.")
    
    parser.add_argument('-n', type=int, default=32, 
                        help="number of rows in the generated map")
    parser.add_argument('-H', type=float, default=0.7, 
                        help="Hurst parameter (alpha/2)")
    parser.add_argument('-w', "--width", dest="width", type=int, default=15,
                        help="Size of one map tile in pixel, only in html format.")
    parser.add_argument('-s', "--seed", dest="seed", type=int, default=3,
                        help="Random seed, if negative then random seed is random.")
    parser.add_argument("-dh", dest="dh", type=int, default=3,
                        help="minimum height difference between contour lines: dhbase*2^dh.\n"
                            "If set to negative then random.")
    parser.add_argument("--dhbase", dest="dhbase", type=float, default=0.125,
                        help="minimum height difference to consider a 'step' in height.")
    parser.add_argument("-r", "--rock", dest="rockface", type=float, default=[0.2],
                        metavar='param',
                        nargs='+', help="Sets when to break a rockface.\n"
                        "If one argument is given, then rock is deleted with uniform probability 'r' (threshold a Poisson noise).\n"
                        "If two arguments are given, then a rock is deleted with probability"
                        " 'sigmoid(a*X+b)'\nwhere X is a Brownian noise.")
    
    parser.add_argument("-o", "--offset", dest="offset", type=float, default=1,
                        help="height offset of map (elevation)")
    
    parser.add_argument("--tiberium", "--resource", dest="resource", 
                        type=float, default=[0.005],
                        nargs='+', help="Sets when to place a tiberium tree.\n"
                        "If one parameter is given, then with uniform probability (threshold a Poisson noise).\n"
                        "If two arguments are given, then with probability"
                        " 'sigmoid(a*X+b)'\nwhere X is a Brownian noise.")

    parser.add_argument("-T", "--tree", "--terrain", dest="tree", 
                        type=float, default=[1, 0.1], nargs=2)
                       
    parser.add_argument("-t", "--type", dest="type", type=str, default="brownian",
                        choices=["brownian", "perlin"],
                        help='Type of the noise to use.')

    parser.add_argument("-hue", "--hue", dest="hue", type=int, default=4)
    
    parser.add_argument("-f", "--format", dest="format", type=str, default="html",
                        choices=['html', 'inibin'], help='Output format (to stdout).')
    
    sys.exit(main(parser.parse_args()))
