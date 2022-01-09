
import sys

import numpy as np
import svgwrite
from flask import Flask, jsonify, request
from IPython.display import SVG, display
from matplotlib import pyplot as plt
from rdp import rdp

sys.path.append("../sketchformer")
from basic_usage.sketchformer import continuous_embeddings

sketchformer = continuous_embeddings.get_pretrained_model()

app = Flask(__name__)


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
    plt.axes().set_aspect('equal')
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


def draw_strokes(data, factor=0.2, svg_filename='sample.svg'):
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)
    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
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


def lines_to_strokes(lines):
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
    strokes[1:, 0] -= strokes[1:, 0].min()
    strokes[1:, 1] -= strokes[1:, 1].min()
    strokes[1:, 0:2] -= strokes[:-1, 0:2]
    return strokes[1:, :]


@app.route('/api/sketch', methods=["POST"])
def draw_next_sketch():
    user_sketch = request.get_json()['sketch']

    # convert to stroke-3 format
    converted_user_sketch = lines_to_strokes(user_sketch)

    # visualize
    visualize(converted_user_sketch)
    draw_strokes(converted_user_sketch)

    # predict
    pred_class = sketchformer.classify([converted_user_sketch])
    print(pred_class)

    return jsonify({
        "class": pred_class
    })


if __name__ == "__main__":
    app.run(debug=True)
