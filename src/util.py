import numpy as np
import svgwrite
from matplotlib import pyplot as plt
from rdp import rdp


def xywh_to_p0p1(x, y, w, h):
    x0 = x - (w / 2)
    y0 = y - (h / 2)
    x1 = x + (w / 2)
    y1 = y + (h / 2)
    return x0, y0, x1, y1


def p0p1_to_xywh(x0, y0, x1, y1):
    w = x1 - x0
    h = y1 - y0
    x = (x0 + x1) / 2
    y = (y0 + y1) / 2
    return x, y, w, h


def raw_to_lines(raw):
    """Convert raw QuickDraw format to polyline format."""
    result = []
    N = len(raw)
    for i in range(N):
        line = []
        rawline = raw[i]
        M = len(rawline[0])
        for j in range(M):
            line.append([rawline[0][j], rawline[1][j]])
        result.append(line)
    return result


def visualize(sketch):
    plt.clf()
    X = []
    Y = []

    tmp_x, tmp_y = [], []
    sx = sy = 0
    for p in sketch:
        sx += p[0]
        sy += p[1]
        tmp_x.append(sx)
        tmp_y.append(-sy)
        if p[2] == 1:
            X.append(tmp_x)
            Y.append(tmp_y)
            tmp_x, tmp_y = [], []

    X.append(tmp_x)
    Y.append(tmp_y)

    for x, y in zip(X, Y):
        plt.plot(x, y)

    # save the image.
    plt.axes().set_aspect("equal")
    plt.savefig("sample.png")

    # show the plot
    # plt.show()


def get_bounds(data, factor):
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0])/factor
        y = float(data[i, 1])/factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)


def draw_strokes(data, factor=0.2, svg_filename="sample.svg"):
    data = np.array(data)
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)
    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill="white"))
    lift_pen = 1
    abs_x = 25 - min_x
    abs_y = 25 - min_y
    p = "M%s,%s " % (abs_x, abs_y)
    command = "m"
    for i in range(len(data)):
        if (lift_pen == 1):
            command = "m"
        elif (command != "l"):
            command = "l"
        else:
            command = ""
        x = float(data[i, 0])/factor
        y = float(data[i, 1])/factor
        lift_pen = data[i, 2]
        p += command+str(x)+","+str(y)+" "
    the_color = "black"
    stroke_width = 1
    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
    dwg.save()
    # display(SVG(dwg.tostring()))


def lines_to_sketch(lines):
    """Convert polyline format to stroke-3 format."""
    eos = 0
    strokes = [[0, 0, 0]]
    for line in lines:
        line = rdp(line, epsilon=2.0)
        linelen = len(line)
        for i in range(linelen):
            eos = 0 if i < linelen - 1 else 1
            strokes.append([line[i][0], line[i][1], eos])
    strokes = np.array(strokes)
    x0, y0, x1, y1 = strokes[1:, 0].min(), strokes[1:, 1].min(
    ), strokes[1:, 0].max(), strokes[1:, 1].max()
    x, y, w, h = p0p1_to_xywh(x0, y0, x1, y1)
    strokes[1:, 0] -= x0
    strokes[1:, 1] -= y0
    strokes[1:, 0:2] -= strokes[:-1, 0:2]

    return {
        "strokes": strokes[1:, :].tolist(),
        "position": list(map(int, [x, y, w, h]))
    }


def strokes_to_lines(strokes):
    """Convert stroke-3 format to polyline format."""
    strokes = np.array(strokes)
    x = 0
    y = 0
    lines = []
    line = []
    for i in range(len(strokes)):
        if strokes[i, 2] == 1:
            x += float(strokes[i, 0])
            y += float(strokes[i, 1])
            line.append([x, y])
            lines.append(line)
            line = []
        else:
            x += float(strokes[i, 0])
            y += float(strokes[i, 1])
            line.append([x, y])
    return lines


def adjust_lines(lines, position):
    x, y, w, h = position
    x0, y0, x1, y1 = xywh_to_p0p1(x, y, w, h)
    wp, hp = x1 - x0, y1 - y0

    xmin = ymin = float("inf")
    xmax = ymax = 0
    for line in lines:
        for x, y in line:
            xmin = min(x, xmin)
            ymin = min(y, ymin)
            xmax = max(x, xmax)
            ymax = max(y, ymax)
    ws, hs = xmax - xmin, ymax - ymin

    x_scale_factor = wp / ws
    y_scale_factor = hp / hs
    scale_factor = min(x_scale_factor, y_scale_factor)

    adjusted_line = []
    adjusted_lines = []
    for line in lines:
        for x, y in line:
            x -= xmin
            y -= ymin
            x *= scale_factor
            y *= scale_factor
            x += x0
            y += y0
            if x_scale_factor < y_scale_factor:
                y += (hp - hs * scale_factor) / 2
            else:
                x += (wp - ws * scale_factor) / 2
            adjusted_line.append([int(x), int(y)])
        adjusted_lines.append(adjusted_line)
        adjusted_line = []

    return adjusted_lines


def to_normal_strokes(big_stroke):
    """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
    l = 0
    for i in range(len(big_stroke)):
        if big_stroke[i, 4] > 0:
            l = i
            break
    if l == 0:
        l = len(big_stroke)
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3]
    result[-1, 2] = 1
    return result
