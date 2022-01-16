import { Button } from "@mui/material";
import React from "react";
import Sketch from "react-p5";

class Canvas extends React.Component {

    constructor(props) {
        super(props);
        this.userLine = [];
        this.userLines = [];
        this.previousSketches = [];
        this.agentLines = [];
        this.isUserTurn = true;
        this.isShowComposition = false;
        this.state = {
            agentMessage: "Agent: Draw your first sketch!",
        };
    }

    setup = (p5, canvasParentRef) => {
        // use parent to render the canvas in this ref
        // (without that p5 will render the canvas outside of your component)
        this.cnv = p5.createCanvas(750, 750).parent(canvasParentRef);
        p5.background(230);

        this.cnv.mouseReleased((event) => {
            this.userLines.push(this.userLine.slice(0, this.userLine.length));
            this.userLine.splice(0);
        })
        
        this.drawNextSketch = (label, position) => {
            const x = position[0];
            const y = position[1];
            const w = position[2];
            const h = position[3];
            const xmin = x - (w / 2)
            const ymin = y - (h / 2)
            
            p5.strokeWeight(1);
            p5.fill(0);
            p5.textSize(20);
            p5.text(label, xmin, ymin - 5);

            p5.noFill();
            p5.rect(xmin, ymin, w, h);
        }
    };

    draw = (p5) => {
        // NOTE: Do not use setState in the draw function or in functions that are executed
        // in the draw function...
        // please use normal variables or class properties for these purposes
        p5.stroke(0);
        p5.strokeWeight(4);
        
        if (this.isUserTurn) {
            // User
            if (p5.mouseIsPressed
                && p5.mouseX <= p5.width && p5.mouseX >= 0
                && p5.mouseY <= p5.height && p5.mouseY >= 0) {
                this.userLine.push([p5.mouseX, p5.mouseY]);
                p5.line(p5.mouseX, p5.mouseY, p5.pmouseX, p5.pmouseY);
            }
        } else {
            // Agent
            if (this.agentLines.length) {
                const p0 = this.agentLines[0].shift();
                if (!this.agentLines[0].length) {
                    this.agentLines.shift();
                    if (!this.agentLines.length) {
                        this.isUserTurn = true;
                    }
                } else {
                    const p1 = this.agentLines[0][0];
                    p5.line(p0[0], p0[1], p1[0], p1[1]);
                }
            }
        }

    };

    fetchSketch = () => {
        const data = {
            "previousSketches": this.previousSketches,
            "userLines": this.userLines,
        };

        fetch("/api/sketch", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(data)
        })
            .then(response => response.json())
            .then(data => {
                console.log(data);
                this.previousSketches = data.previousSketches;
                this.agentLines = data.nextLines;
                if (this.isShowComposition) {
                    this.drawNextSketch(data.nextSketch.name, data.nextSketch.position);
                }
                this.setState({
                    agentMessage: `Agent: I drew the ${data.nextSketch.name}, now it's your turn!`
                });
            });
    }

    handleEndSketchButtonClick = () => {
        if (!this.userLines.length) {
            return;
        }
        this.isUserTurn = false;
        this.fetchSketch();
        this.userLines.splice(0);
    }

    render() {
        return <>
            <Sketch setup={this.setup} draw={this.draw} />
            <p>{this.state.agentMessage}</p>
            <Button variant="contained" onClick={this.handleEndSketchButtonClick}>Next</Button>
        </>;
    }

}

export default Canvas;