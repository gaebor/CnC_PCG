from brownian_sheet import *
import numpy

def by_color(F, G, H, width=20):
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

def html(E, width=20, hue=4):
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
    result += "</td>"
    return result
    
def main(args, render_f):
    if args.seed >= 0:
        numpy.random.seed(args.seed)
    
    final_size = args.n*2-2
    
    if args.type == "brownian":
        B, X = generate(args.n, args.n, H=args.H)
        R1, R2 = generate(final_size, final_size, H=args.H)
    elif args.type == "perlin":
        B = generate_perlin((args.n, args.n), (1, 1), 1+int(numpy.log2(args.n)), H=args.H)
        X = generate_perlin((args.n, args.n), (1, 1), 1+int(numpy.log2(args.n)), H=args.H)
        R1 = generate_perlin((final_size, final_size), (1, 1), 1+int(numpy.log2(final_size)), H=args.H)
        R2 = generate_perlin((final_size, final_size), (1, 1), 1+int(numpy.log2(final_size)), H=args.H)
    
    B += args.offset
    
    if len(args.rockface) == 1:
        args.rockface = args.rockface[0]
    else:
        args.rockface = sigmoid(args.rockface[0]*X + args.rockface[1])
    
    if len(args.resource) == 1:
        args.resource = args.resource[0]
    else:
        args.resource = sigmoid(args.resource[0]*R1 + args.resource[1])
    
    if args.dh < 0:
        H, _ = generate(args.n, args.n, H=args.H)
        c = 4
        args.dh = (31 + (31*(-31 + c))/(31 - c + c*numpy.exp(H))).astype("int32")

    M = generate_map(B, dh=args.dh, dhbase=args.dhbase, dx=args.rockface)
    templates, icons = render_f(M)
    
    Rmask = numpy.zeros(templates.shape, dtype=bool)
    Rmask[1:1+final_size,1:1+final_size] = numpy.logical_and(
            numpy.random.rand(final_size, final_size) < args.resource,
            templates[1:1+final_size,1:1+final_size] == numpy.iinfo(templates.dtype).max
            )
    resource_positions = numpy.where(Rmask.flatten())[0]
    
    return M, templates, icons, resource_positions
