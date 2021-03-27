import brownian_sheet
import numpy
from zlib import crc32
import struct


def match_mask2(map, pattern):
    return (
        numpy.array_equiv(map[pattern == 0], 0)  # 0: should be 0
        and numpy.all(map[pattern == 1] != 0)  # 1: must containg something
        and numpy.all(map[pattern >= 2] >= 0)  # 2 or above: land (i.e. 0 or positive)
        and numpy.all(map[pattern < 0] < 0)  # negative: water
        and numpy.all(map[pattern == 3] < map[pattern == 4])  # 3 should be lower than 4
    )


def prepend_dim(x):
    return x.reshape(1, *x.shape)


def prepare_formation(pattern, template, icon=None, template_map=None):
    pattern = numpy.array(pattern)
    template = numpy.array(template)
    if template_map is not None and template.dtype.type == numpy.dtype(str):
        template = numpy.vectorize(template_map.get)(template)

    if pattern.ndim == 2:
        pattern == prepend_dim(pattern)

    shape = pattern.shape[0] - 1, pattern.shape[1] - 1

    if template.ndim < 2:
        if template.ndim == 0:
            template = template.reshape(1)

        ones_of_shape = numpy.ones(shape, dtype=pattern.dtype)
        arange_of_shape = numpy.arange(shape[0] * shape[1]).reshape(shape)

        template = template[:, None, None] * ones_of_shape[None, :, :]
        icon = numpy.ones_like(template) * arange_of_shape
    else:
        icon = numpy.array(icon)
        if template.ndim == 2:
            template = prepend_dim(template)
            icon = prepend_dim(icon)
    return template, icon


# def check_formation(patterns, roi):
#     for pattern in patterns:
#         if match_mask(roi, pattern):
#             return True
#     return False


# def get_templates_and_icons(self, pseudo_random_choice=0):
#     i = pseudo_random_choice % templates.shape[0]
#     return (self.templates[i], self.icons[i])


def match_mask(A, M):
    '''
    Checks if A is zero where M is negative and A is non-zero where M is positive.
    It doesn't matter what A is where M == 0
    '''
    return numpy.array_equiv(A[M < 0], 0) and numpy.all(A[M > 0] != 0)


def fixed_random(x, y=None, n=2):
    if y is not None:
        return crc32(struct.pack('BB', x, y)) % n
    else:
        return crc32(struct.pack('H', x)) % n


def by_color(F, G, H, width=20):
    result = ["<!DOCTYPE html>", "<html>", "<head>"]
    result.append("<style>")
    result.append(
        "td {{ text-align: center; vertical-align: middle; width:{0}px; height:{0}px;}}".format(
            width
        )
    )
    result.append("tr {{ height:{0}px;}}".format(width))
    result.append("</style>")
    result.append("</head>")
    result.append("<body>")
    result.append(
        "<table border=0 cellpadding=0 cellspacing=0 style='border-collapse: collapse; table-layout:fixed;'>"
    )

    result.append("  <col width={0} style='width={0}px;' span={1}>".format(width, F.shape[1]))
    for i in range(F.shape[0]):
        result.append("  <tr>")
        for j in range(F.shape[1]):
            color = "rgb({},{},{})".format(F[i, j], G[i, j], H[i, j])
            result.append(rendertile(i, j, color, width=width))
        result.append("  </tr>")
    result.append("</table>")
    result.append("</body>")
    result.append("</html>")
    return '\n'.join(result)


def html(E, width=20, hue=4):
    result = ["<!DOCTYPE html>", "<html>", "<head>"]
    result.append("<style>")
    result.append(
        "td {{ text-align: center; vertical-align: middle; width:{0}px; height:{0}px;}}".format(
            width
        )
    )
    result.append("tr {{ height:{0}px;}}".format(width))
    result.append("</style>")
    result.append("</head>")
    result.append("<body>")
    result.append(
        "<table border=0 cellpadding=0 cellspacing=0 style='border-collapse: collapse; table-layout:fixed;'>"
    )

    h = (E.shape[0] + 1) // 2
    w = (E.shape[1] + 1) // 2

    result.append("  <col width={0} style='width={0}px;' span={1}>".format(width, w))
    for i in range(h):
        result.append("  <tr>")
        for j in range(w):
            sides = []
            color = (
                "blue"
                if E[2 * i, 2 * j] < 0
                else "hsl({},100%,50%)".format(140 - hue * E[2 * i, 2 * j])
            )
            if i < h - 1:
                if E[2 * i + 1, 2 * j] > 0:
                    sides.append(('bottom', "2px solid brown"))
            if j < w - 1:
                if E[2 * i, 2 * j + 1] > 0:
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
    result += "'><b>+</b></td>" if i % 2 == 0 and j % 2 == 0 else "'>+</td>"
    return result


def make_threshold_mask(th, X):
    if len(th) == 1:
        th = th[0]
    else:
        th = brownian_sheet.sigmoid(th[0] * X + th[1])
    return th


def main(args, render_f):
    if args.seed >= 0:
        numpy.random.seed(args.seed)

    final_size = args.n * 2 - 2

    generator = getattr(brownian_sheet, args.type)

    B, X = generator(args.n, args.n, H=args.H)
    R1, R2 = generator(final_size, final_size, H=args.H)

    B += args.offset

    args.rockface = make_threshold_mask(args.rockface, X)
    args.resource = make_threshold_mask(args.resource, R1)
    args.tree = make_threshold_mask(args.tree, R2)

    if len(args.dh) > 1:
        H, _ = generator(args.n, args.n, H=args.H)
        args.dh = (32 * make_threshold_mask(args.dh, H)).astype('int32')
    else:
        args.dh = int(args.dh[0])

    M = brownian_sheet.generate_map(B, dh=args.dh, dhbase=args.dhbase, dx=args.rockface)
    templates, icons = render_f(M)

    # free cells
    Fmask = numpy.zeros(templates.shape, dtype=bool)
    Fmask[1 : 1 + final_size, 1 : 1 + final_size] = (
        templates[1 : 1 + final_size, 1 : 1 + final_size] == numpy.iinfo(templates.dtype).max
    )

    Rmask = numpy.zeros(templates.shape, dtype=bool)
    Rmask[1 : 1 + final_size, 1 : 1 + final_size] = numpy.logical_and(
        numpy.random.rand(final_size, final_size) < args.resource,
        Fmask[1 : 1 + final_size, 1 : 1 + final_size],
    )
    resource_positions = numpy.where(Rmask.flatten())[0]

    Rmask[1 : 1 + final_size, 1 : 1 + final_size] = numpy.logical_and(
        numpy.random.rand(final_size, final_size) < args.tree,
        numpy.logical_and(
            Fmask[1 : 1 + final_size, 1 : 1 + final_size],
            ~Rmask[1 : 1 + final_size, 1 : 1 + final_size],
        ),
    )
    tree_positions = numpy.where(Rmask.flatten())[0]

    return M, templates, icons, resource_positions, tree_positions
