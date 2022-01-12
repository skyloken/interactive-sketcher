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
    }

    setup = (p5, canvasParentRef) => {
        // use parent to render the canvas in this ref
        // (without that p5 will render the canvas outside of your component)
        this.cnv = p5.createCanvas(750, 750).parent(canvasParentRef);
        p5.background(200);

        this.cnv.mouseReleased((event) => {
            this.userLines.push(this.userLine.slice(0, this.userLine.length));
            this.userLine.splice(0);
        })
        
        this.drawNextSketch = (label, position) => {
            const x0 = position[0];
            const y0 = position[1];
            const x1 = position[2];
            const y1 = position[3];
            const w = x1 - x0;
            const h = y1 - y0;
            
            p5.strokeWeight(1);
            p5.fill(0);
            p5.textSize(20);
            p5.text(label, x0, y0);

            p5.noFill();
            p5.rect(x0, y0, w, h);
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
                this.drawNextSketch(data.nextSketch.name, data.nextSketch.position);
                this.agentLines = data.nextLines;
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
            <Button variant="contained" onClick={this.handleEndSketchButtonClick}>Next</Button>
        </>;
    }

}

export default Canvas;