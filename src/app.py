import datetime
import json
import os

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory
from waitress import serve

from service import Agent, Sketchformer, SketchRNN
from util import adjust_lines, lines_to_sketch, strokes_to_lines

app = Flask(__name__, static_folder="../app/build/static",
            template_folder="../app/build")

agent = Agent()
sketchformer = Sketchformer()
sketchrnn = SketchRNN()


def save_to_log(seq_id, username, new_sketches, mode):

    log_file = "../data/isketcher/exp_log.csv"
    log_df = pd.read_csv(log_file, index_col=0)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if seq_id not in log_df.index:
        # insert
        row = pd.Series([username, str(mode), timestamp, json.dumps(new_sketches)],
                        index=log_df.columns, name=seq_id)
        log_df = log_df.append(row)
    else:
        # update
        sketches = json.loads(log_df.loc[seq_id]["sketches"])
        row = pd.Series([username, str(mode), timestamp, json.dumps(sketches + new_sketches)],
                        index=log_df.columns, name=seq_id)
        log_df.loc[seq_id] = row

    log_df.to_csv(log_file)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/sketch", methods=["POST"])
def draw_next_sketch():
    seq_id = request.get_json()["seqId"]
    username = request.get_json()["username"]
    previous_sketches = request.get_json()["previousSketches"]
    user_lines = request.get_json()["userLines"]
    mode = request.get_json()["mode"]

    is_rand = False
    if mode == 1:
        is_rand = False
    elif mode == 2:
        is_rand = True

    # convert to stroke-3 format
    user_sketch = lines_to_sketch(user_lines)

    # next sketch
    inp = sketchformer.preprocess(previous_sketches + [user_sketch])
    next_sketch = agent.get_next_sketch(
        inp) if not is_rand else agent.get_rand_sketch()
    print("Next sketch:", next_sketch)

    strokes = sketchrnn.get_random_strokes(next_sketch["name"])
    agent_sketch = {
        "strokes": strokes,
        "position": next_sketch["position"]
    }

    lines = strokes_to_lines(strokes)
    adjusted_lines = adjust_lines(lines, next_sketch["position"])

    # save log
    save_to_log(seq_id, username, [user_lines] + [adjusted_lines], mode)

    return jsonify({
        "nextSketch": next_sketch,
        "nextLines": adjusted_lines,
        "previousSketches": previous_sketches + [user_sketch] + [agent_sketch]
    })


if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=3000)
