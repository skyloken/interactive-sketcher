import datetime
import json

import pandas as pd
from flask import Flask, jsonify, render_template, request
from waitress import serve

from service import Agent, Sketchformer, SketchRNN
from util import adjust_lines, lines_to_sketch, strokes_to_lines

app = Flask(__name__, static_folder="../app/build/static",
            template_folder="../app/build")

agent = Agent()
sketchformer = Sketchformer()
sketchrnn = SketchRNN()


def save_to_log(seq_id, username, new_sketches, mode, next_sketch):

    log_file = "../data/isketcher/exp_log.csv"
    log_df = pd.read_csv(log_file, index_col=0)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if seq_id not in log_df.index:
        # insert
        row = pd.Series([username, str(mode), timestamp, json.dumps(new_sketches), json.dumps([next_sketch])],
                        index=log_df.columns, name=seq_id)
        log_df = log_df.append(row)
    else:
        # update
        sketches = json.loads(log_df.loc[seq_id]["sketches"])
        outputs = json.loads(log_df.loc[seq_id]["outputs"])
        row = pd.Series([username, str(mode), timestamp, json.dumps(sketches + new_sketches), json.dumps(outputs + [next_sketch])],
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
    print("mode:", mode)

    is_rand = False
    sketch_from_dataset = False
    if mode == 1: # 構図生成モデル、スケッチ生成
        is_rand = False
        sketch_from_dataset = False
    elif mode == 2: # ランダムスケッチ、構図生成モデル
        is_rand = True
        sketch_from_dataset = False
    elif mode == 3: # 構図生成モデル、データセットからスケッチ取得
        is_rand = False
        sketch_from_dataset = True
    elif mode == 4: # ランダムスケッチ、データセットからスケッチ取得
        is_rand = True
        sketch_from_dataset = True

    # convert to stroke-3 format
    user_sketch = lines_to_sketch(user_lines)

    # next sketch
    inp = sketchformer.preprocess(previous_sketches + [user_sketch])
    next_sketch = agent.get_next_sketch(
        inp) if not is_rand else agent.get_rand_sketch()
    print("Next sketch:", next_sketch)

    strokes = sketchrnn.get_random_strokes(next_sketch["name"], from_dataset=sketch_from_dataset)
    agent_sketch = {
        "strokes": strokes,
        "position": next_sketch["position"]
    }

    lines = strokes_to_lines(strokes)
    adjusted_lines = adjust_lines(lines, next_sketch["position"])

    # save log
    save_to_log(seq_id, username, [user_lines] +
                [adjusted_lines], mode, next_sketch)

    return jsonify({
        "nextSketch": next_sketch,
        "nextLines": adjusted_lines,
        "previousSketches": previous_sketches + [user_sketch] + [agent_sketch]
    })


if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=3000)
