from brownian_sheet import *
import numpy
from argparse import ArgumentParser
import sys
from lcw import mapwrite, MyFormatter

def generate_by_color(F, G, H, width=20):
    result = ["<!DOCTYPE html>", "<html>", "<head>"]
    result.append("<style>")
    result.append("td {{ text-align: center; vertical-align: middle; width:{0}px; height:{0}px;}}".format(width))
    result.append("tr {{ height:{0}px;}}".format(width))
    result.append("</style>")
    result.append("</head>")
    result.append("<body>")
    result.append("<table border=0 cellpadding=0 cellspacing=0 style='border-collapse: collapse; table-layout:fixed;'>")
    
    result.append("  <col width={0} style='width={0}px;' span={1}>".format(width, F.shape[1]))
    for i in range(F.shape[0]):
        result.append("  <tr>")
        for j in range(F.shape[1]):
            color = "rgb({},{},{})".format(F[i,j], G[i,j], H[i,j])
            result.append(rendertile(i, j, color, width=width))
        result.append("  </tr>")
    result.append("</table>")
    result.append("</body>")
    result.append("</html>")
    return '\n'.join(result)

def generate_html(E, width=20, hue=4):
    result = ["<!DOCTYPE html>", "<html>", "<head>"]
    result.append("<style>")
    result.append("td {{ text-align: center; vertical-align: middle; width:{0}px; height:{0}px;}}".format(width))
    result.append("tr {{ height:{0}px;}}".format(width))
    result.append("</style>")
    result.append("</head>")
    result.append("<body>")
    result.append("<table border=0 cellpadding=0 cellspacing=0 style='border-collapse: collapse; table-layout:fixed;'>")
    
    h = (E.shape[0]+1)//2
    w = (E.shape[1]+1)//2
    
    result.append("  <col width={0} style='width={0}px;' span={1}>".format(width, w))
    for i in range(h):
        result.append("  <tr>")
        for j in range(w):
            sides = []
            color = "blue" if E[2*i,2*j]<0 else "hsl({},100%,50%)".format(140-hue*E[2*i,2*j])
            if i < h-1:
                if E[2*i+1,2*j] > 0:
                    sides.append(('bottom', "2px solid brown"))
            if j < w-1:
                if E[2*i,2*j+1] > 0:
                    sides.append(('right', "2px solid brown"))
            result.append(rendertile(i, j, color, sides, width))
        result.append("  </tr>")
    result.append("</table>")
    result.append("</body>")
    result.append("</html>")
    return '\n'.join(result)
 
def rendertile(i, j, color, l=[], width=20, thinstroke=1, thickstroke=2):
    result = "<td style='background-color:{}; ".format(color)
    for s in l:
        if type(s) == str:
            result += "border-{}: {};".format(s, "1px solid black")
        elif len(s) > 1:
            result += "border-{}: {};".format(*s)
    result += "'>"
    # result += "+"
    # cross=""
    # if j%2==0:
        # if i%4==0:
            # cross = "<path stroke-width={thick} stroke='white' d='M0 {1} l{0} 0 M{1} {1} l0 {1}' /> <path stroke-width={thin} stroke='white' d='M{1} 0 l0 {1}' />"
        # elif i%4==1:
            # cross = "<path stroke-width={thin} stroke='white' d='M0 {1} l{0} 0' /> <path stroke-width={thick} stroke='white' d='M{1} 0 l0 {0}' />"
        # elif i%4==2:
            # cross = "<path stroke-width={thin} stroke='white' d='M{1} {1} l0 {1}' /> <path stroke-width={thick} stroke='white' d='M0 {1} l{0} 0 M{1} 0 l0 {1}' />"
        # else:
            # cross = "<path stroke-width={thin} stroke='white' d='M0 {1} l{0} 0 M{1} 0 l0 {0}' />"
    # else:
        # if i%4==0:
            # cross = "<path stroke-width={thin} stroke='white' d='M{1} {1} l0 {1}' /> <path stroke-width={thick} stroke='white' d='M0 {1} l{0} 0 M{1} 0 l0 {1}' />"
        # elif i%4==1:
            # cross = "<path stroke-width={thin} stroke='white' d='M0 {1} l{0} 0 M{1} 0 l0 {0}' />"
        # elif i%4==2:
            # cross = "<path stroke-width={thick} stroke='white' d='M0 {1} l{0} 0 M{1} {1} l0 {1}' /> <path stroke-width={thin} stroke='white' d='M{1} 0 l0 {1}' />"
        # else:
            # cross = "<path stroke-width={thin} stroke='white' d='M0 {1} l{0} 0' /> <path stroke-width={thick} stroke='white' d='M{1} 0 l0 {0}' />"
    # result += ("<svg width='{0}' height='{0}'> " + cross + " </svg>").\
                # format(width-2, (width-2)/2, thin=thinstroke, thick=thickstroke)

    result += "</td>"
    return result

def to_tiles(M):
    """
    calculates RA1 template and icon enums from edge and cell data
    numbering is according to https://github.com/electronicarts/CnC_Remastered_Collection/blob/master/REDALERT/DEFINES.H
    TemplateType enum
    """
    templates = numpy.ones((128,128), dtype=numpy.uint16)*0xFFFF
    icons = numpy.ones((128,128), dtype=numpy.uint8)*0xFF
    
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
            
            template = 0xFFFF
            
            if number_of_adjacent_rocks == 0:
                if M[i-1,j-1] < 0:
                    # water2
                    template = 2
            elif number_of_adjacent_rocks == 1:
                # in this case there cannot be water
                if M[i+1,j] != 0:
                    if M[i+1,j-1] > M[i+1,j+1]:
                        # 0 0
                        # +|-
                        # slope22, and a bit of padding
                        templates[i+1,j] = 156
                        templates[i+1,j+1] = 156
                        
                        icons[i+1,j] = 0
                        icons[i+1,j+1] = 1
                    else:
                        # 0 0
                        # -|+
                        # slope08
                        template = 142
                elif M[i-1,j] != 0:
                    if M[i-1,j-1] > M[i-1,j+1]:
                        # +|-
                        # 0 0
                        # slope28
                        templates[i,j] = 162
                        templates[i,j+1] = 162
                        templates[i+1,j+1] = 162
                        
                        icons[i,j] = 0
                        icons[i,j+1] = 1
                        icons[i+1,j+1] = 3
                    else:
                        # -|+
                        # 0 0
                        # slope14
                        templates[i,j] = 148
                        templates[i,j+1] = 148
                        templates[i+1,j] = 148
                        
                        icons[i,j] = 0
                        icons[i,j+1] = 1
                        icons[i+1,j] = 2
                elif M[i,j-1] != 0:
                    if M[i-1,j-1] > M[i+1,j-1]:
                        # +_0
                        # - 0
                        # slope07
                        template = 141
                    else:
                        # -_0
                        # + 0
                        # slope21 bit with padding
                        templates[i,j] = 155
                        templates[i+1,j] = 155
                        
                        icons[i,j] = 0
                        icons[i+1,j] = 1                        
                elif M[i,j+1] != 0:
                    if M[i-1,j+1] > M[i+1,j+1]:
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
                if M[i,j-1] != 0 and M[i-1,j] != 0:
                    # _|
                    if M[i-1,j-1] > M[i+1,j-1]:
                        if M[i+1,j-1] >= 0:
                            # slope32
                            templates[i,j] = 166
                            templates[i,j+1] = 166
                            templates[i+1,j] = 166
                            
                            icons[i,j] = 0
                            icons[i,j+1] = 1
                            icons[i+1,j] = 2
                        else:
                            # watercliff32
                            template = 90
                    else:
                        if M[i-1,j-1] >= 0:
                            # slope36
                            template = 170
                        else:
                            # watercliff36
                            template = 94
                elif M[i-1,j] != 0 and M[i,j+1] != 0:
                    # |_
                    if M[i-1,j+1] > M[i-1,j-1]:
                        if M[i-1,j-1] >= 0:
                            # slope29
                            template = 163
                        else:
                            # watercliff29
                            template = 87
                    else:
                        if M[i-1,j+1] >= 0:
                            # slope33
                            template = 167
                        else:
                            # watercliff33
                            template = 91
                elif M[i,j+1] != 0 and M[i+1,j] != 0:
                    #  _
                    # |
                    if M[i+1,j+1] > M[i+1,j-1]:
                        if M[i+1,j-1] >= 0:
                            # slope30
                            template = 164
                        else:
                            # watercliff30
                            template = 88
                    else:
                        if M[i+1,j+1] >= 0:
                            # slope34
                            template = 168
                        else:
                            # watercliff34
                            template = 92
                elif M[i+1,j] != 0 and M[i,j-1] != 0:
                    # _ 
                    #  |
                    if M[i+1,j-1] > M[i-1,j-1]:
                        if M[i-1,j-1] >= 0:
                            # slope31
                            template = 165
                        else:
                            # watercliff31
                            template = 89
                    else:
                        if M[i+1,j-1] >= 0:
                            # slope35
                            template = 169
                        else:
                            # watercliff35
                            template = 93
                elif M[i+1,j] != 0 and M[i-1,j] != 0:
                    #  |
                    #  |
                    if M[i+1,j-1] > M[i+1,j+1]:
                        if M[i+1,j+1] >= 0:
                            # slope26
                            template = 160
                        else:
                            # watercliff26
                            template = 84
                    else:
                        if M[i+1,j-1] >= 0:
                            # slope12
                            template = 146
                        else:
                            # watercliff12
                            template = 70
                elif M[i,j-1] != 0 and M[i,j+1] != 0:
                    # _ _
                    if M[i+1,j-1] > M[i-1,j-1]:
                        if M[i-1,j-1] >= 0:
                            # slope17
                            template = 151
                        else:
                            # watercliff17
                            template =  75
                    else:
                        if M[i+1,j-1] >= 0:
                            # slope04
                            template = 138
                        else:
                            # watercliff04
                            template = 62
            elif number_of_adjacent_rocks == 4:
                if M[i-1,j-1] > M[i-1,j+1]:
                    if M[i-1,j+1] >= 0:
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
                    if M[i-1,j-1] >= 0:
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
                templates[i:i+2,j:j+2] = template
                icons[i,j] = 0
                icons[i,j+1] = 1
                icons[i+1,j] = 2
                icons[i+1,j+1] = 3

    return templates, icons
    
def main(args):
    if args.seed >= 0:
        numpy.random.seed(args.seed)
    
    if args.type == "brownian":
        B, X = generate(args.n, args.n, H=args.H)
    elif args.type == "perlin":
        B = generate_perlin((args.n, args.n), (1, 1), 1+int(numpy.log2(args.n)), H=args.H)
        X = generate_perlin((args.n, args.n), (1, 1), 1+int(numpy.log2(args.n)), H=args.H)
    B += args.offset
    
    if len(args.rockface) == 1:
        args.rockface = args.rockface[0]
    else:
        args.rockface = sigmoid(args.rockface[0]*X + args.rockface[1])
        
    if args.dh < 0:
        H, _ = generate(args.n, args.n, H=args.H)
        c = 4
        args.dh = (31 + (31*(-31 + c))/(31 - c + c*numpy.exp(H))).astype("int32")
        print(args.dh.max(), file=sys.stderr)

    print(B.min(), B.max(), file=sys.stderr)
    M = generate_map(B, dh=args.dh, dhbase=args.dhbase, dx=args.rockface)
    if args.format == 'html':
        print(generate_html(M, width=args.width, hue=args.hue))
    elif args.format == 'mpr':
        mapwrite(*to_tiles(M), width=M.shape[1]-1, height=M.shape[0]-1)
    return 0

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=MyFormatter,
                        description='Generate a random CnC (RA1) map.\n'
                        'Author: Gabor Borbely (gaebor)\n')
    parser.add_argument('-n', type=int, default=64, 
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
    parser.add_argument("-r", "--rock", dest="rockface", type=float, default=[0.1],
                        metavar='param',
                        nargs='+', help="Sets when to break a rockface.\n"
                        "If one argument is given, then rock is deleted with uniform probability 'r' (threshold a Poisson noise).\n"
                        "If two arguments are given, then a rock is deleted with probability"
                        " 'sigmoid(a*X+b)'\nwhere X is a Brownian noise.")
    
    parser.add_argument("-m", "--mine", "--resource", dest="resource", 
                        type=float, default=[1, 0.1], metavar="param",
                        nargs=2, help="Sets when to place a resource field/mine.\n"
                        "if m=sigmoid(a*X+b) then: gold is when 0.25 <= m < 0.5,\n"
                        "gem is when 0.5 <= m < 0.75 and mine is when m >= 0.75")

    parser.add_argument("-T", "--tree", "--terrain", dest="tree", 
                        type=float, default=[1, 0.1], nargs=2)
                       
    parser.add_argument("-t", "--type", dest="type", type=str, default="brownian",
                        choices=["brownian", "perlin"],
                        help='Type of the noise to use.')
    parser.add_argument("-hue", "--hue", dest="hue", type=int, default=4)
    parser.add_argument("-o", "--offset", dest="offset", type=float, default=0,
                        help="height offset of map (elevation)")
    
    parser.add_argument("-f", "--format", dest="format", type=str, default="html",
                        choices=['html', 'mpr'], help='Output format (to stdout).')
    
    sys.exit(main(parser.parse_args()))
