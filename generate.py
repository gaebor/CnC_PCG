from zlib import crc32
import struct
import json
import time

import numpy

import brownian_sheet


def match_pattern(pattern, roi):
    return (
        (roi[pattern == 0] == 0).all()
        and (roi[pattern == 1] > 0).all()
        and (roi[(pattern >= 2) & (pattern <= 4)] >= 0).all()
        # 2, 3 or 4: land (i.e. 0 or positive)
        and (roi[pattern < 0] < 0).all()
        and (roi[pattern == 3] < roi[pattern == 4]).all()  # 3 should be lower than 4
    )


def prepend_dim(x):
    return x.reshape(1, *x.shape)


def import_tiles_file(filename):
    with open(filename, 'rt') as f:
        tiles_spec = json.load(f)
        if 'template_list' in tiles_spec:
            template_map = {name: i for i, name in enumerate(tiles_spec['template_list'])}
            template_map['NONE'] = tiles_spec['template_none']
        else:
            template_map = None

        return [
            [prepare_formation(template_map=template_map, **pattern) for pattern in level]
            for level in tiles_spec['patterns']
        ]


def prepare_formation(pattern, template, icon=None, template_map=None):
    pattern = numpy.array(pattern)
    template = numpy.array(template)
    if template_map is not None and template.dtype.type == numpy.dtype(str):
        template = numpy.vectorize(template_map.__getitem__)(template)

    if pattern.ndim == 2:
        pattern = prepend_dim(pattern)

    shape = pattern.shape[1] - 1, pattern.shape[2] - 1

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
    return pattern, template, icon


def get_pseudo_random_choice(templates, icons, pseudo_random_seed=0):
    i = pseudo_random_seed % templates.shape[0]
    return templates[i], icons[i]


def fixed_random(x=None, y=None):
    if x is None:
        return crc32(struct.pack('d', time.time()))
    if y is None:
        return crc32(struct.pack('H', x))

    return crc32(struct.pack('BB', x, y))


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
        "<table border=0 cellpadding=0 cellspacing=0 "
        "style='border-collapse: collapse; table-layout:fixed;'>"
    )

    result.append("  <col width={0} style='width={0}px;' span={1}>".format(width, F.shape[1]))
    for i in range(F.shape[0]):
        result.append("  <tr>")
        for j in range(F.shape[1]):
            color = "rgb({},{},{})".format(F[i, j], G[i, j], H[i, j])
            result.append(rendertile(i, j, color))
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
        "<table border=0 cellpadding=0 cellspacing=0 "
        "style='border-collapse: collapse; table-layout:fixed;'>"
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
                elif E[2 * i + 1, 2 * j] < 0:
                    sides.append(('bottom', "1px dotted brown"))
            if j < w - 1:
                if E[2 * i, 2 * j + 1] > 0:
                    sides.append(('right', "2px solid brown"))
                elif E[2 * i, 2 * j + 1] < 0:
                    sides.append(('right', "1px dotted brown"))
            result.append(rendertile(i, j, color, sides))
        result.append("  </tr>")
    result.append("</table>")
    result.append("</body>")
    result.append("</html>")
    return '\n'.join(result)


def rendertile(i, j, color, l=()):
    result = "<td style='background-color:{}; ".format(color)
    for s in l:
        if isinstance(s, str):
            result += "border-{}: {};".format(s, "1px solid black")
        elif len(s) > 1:
            result += "border-{}: {};".format(*s)
    result += "'><b>+</b></td>" if i % 2 == 0 and j % 2 == 0 else "'>+</td>"
    return result


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def make_threshold_mask(th, X):
    if len(th) == 1:
        th = th[0]
    else:
        th = sigmoid(th[0] * X + th[1])
    return th


def render_tiles(M, templates, icons, tile_patterns):
    pattern_size = numpy.array(tile_patterns[0][0].shape[1:])
    tile_size = numpy.array(tile_patterns[0][1].shape[1:])
    offset = (pattern_size - tile_size - 1) // 2

    for i in range(1 + offset[0], M.shape[0] - (tile_size[0] - 1) - offset[0], tile_size[0]):
        for j in range(1 + offset[1], M.shape[1] - (tile_size[1] - 1) - offset[1], tile_size[1]):
            target_slice = (slice(i, i + tile_size[0]), slice(j, j + tile_size[1]))

            if (icons[target_slice] < 255).any():
                continue

            roi = M[
                i - 1 - offset[0] : i + tile_size[0] + offset[0],
                j - 1 - offset[1] : j + tile_size[1] + offset[1],
            ]
            pseudo_random_seed = fixed_random(i, j)

            for patterns, templates_replace, icons_replace in tile_patterns:
                if any(match_pattern(pattern, roi) for pattern in patterns):
                    new_templates, new_icons = get_pseudo_random_choice(
                        templates_replace, icons_replace, pseudo_random_seed
                    )
                    templates[target_slice] = new_templates
                    icons[target_slice] = new_icons
                    break

    return templates, icons


def random_height_map(n, noise_type='brownian', H=0.5, offset=0):
    generator = getattr(brownian_sheet, noise_type)
    return generator(n, n, H=H)[0] + offset


def rock_formations(height_map, rockface, dhbase, dh, noise_type='brownian', H=0.5):
    generator = getattr(brownian_sheet, noise_type)
    rock_seed, dh_seed = generator(*height_map.shape, H=H)
    rockface_threshold = make_threshold_mask(rockface, rock_seed)
    dh_threshold = make_threshold_mask(dh, dh_seed)

    return brownian_sheet.generate_map(
        height_map, dh=dh_threshold, dhbase=dhbase, dx=rockface_threshold
    )


def random_map(n, rockface, dhbase, dh, noise_type='brownian', H=0.5, offset=0):
    height_map = random_height_map(n, noise_type=noise_type, H=H, offset=offset)
    return rock_formations(height_map, rockface, dhbase, dh, noise_type=noise_type, H=H)


def scatter_overlays(
    templates, final_size, resource_params, tree_params, noise_type='brownian', H=0.5
):
    R1, R2 = getattr(brownian_sheet, noise_type)(final_size, final_size, H=H)
    resource_threshold = make_threshold_mask(resource_params, R1)
    tree_threshold = make_threshold_mask(tree_params, R2)

    # free cells
    Fmask = numpy.zeros(templates.shape, dtype=bool)
    Fmask[1 : 1 + final_size, 1 : 1 + final_size] = (
        templates[1 : 1 + final_size, 1 : 1 + final_size] == numpy.iinfo(templates.dtype).max
    )

    Rmask = numpy.zeros(templates.shape, dtype=bool)
    Rmask[1 : 1 + final_size, 1 : 1 + final_size] = numpy.logical_and(
        numpy.random.rand(final_size, final_size) < resource_threshold,
        Fmask[1 : 1 + final_size, 1 : 1 + final_size],
    )
    resource_positions = numpy.where(Rmask.flatten())[0]

    Rmask[1 : 1 + final_size, 1 : 1 + final_size] = numpy.logical_and(
        numpy.random.rand(final_size, final_size) < tree_threshold,
        numpy.logical_and(
            Fmask[1 : 1 + final_size, 1 : 1 + final_size],
            ~Rmask[1 : 1 + final_size, 1 : 1 + final_size],
        ),
    )
    tree_positions = numpy.where(Rmask.flatten())[0]

    return resource_positions, tree_positions
