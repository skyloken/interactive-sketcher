from flask import Flask, jsonify, request

from service import Agent, Sketchformer, SketchRNN
from util import adjust_lines, lines_to_sketch, strokes_to_lines

app = Flask(__name__)

agent = Agent()
sketchformer = Sketchformer()
sketchrnn = SketchRNN()


@app.route("/api/sketch", methods=["POST"])
def draw_next_sketch():
    previous_sketches = request.get_json()["previousSketches"]
    user_lines = request.get_json()["userLines"]

    # convert to stroke-3 format
    user_sketch = lines_to_sketch(user_lines)

    # next sketch
    inp = sketchformer.preprocess(previous_sketches + [user_sketch])
    next_sketch = agent.get_next_sketch(inp)
    print(next_sketch)

    strokes = sketchrnn.get_random_strokes(next_sketch["name"])
    agent_sketch = {
        "strokes": strokes,
        "position": next_sketch["position"]
    }

    lines = strokes_to_lines(strokes)
    adjusted_lines = adjust_lines(lines, next_sketch["position"])

    # predict
    pred_class = sketchformer.classify([user_sketch["strokes"]])
    print(pred_class)

    return jsonify({
        "nextSketch": next_sketch,
        "nextLines": adjusted_lines,
        "previousSketches": previous_sketches + [user_sketch] + [agent_sketch]
    })


if __name__ == "__main__":
    app.run(debug=True)
