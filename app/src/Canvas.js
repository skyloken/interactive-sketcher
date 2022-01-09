import { Button } from "@mui/material";
import Sketch from "react-p5";

function Canvas() {

    const userStrokes = [];

    const userSketch = [];

    let cnv;

    const setup = (p5, canvasParentRef) => {
        // use parent to render the canvas in this ref
        // (without that p5 will render the canvas outside of your component)
        cnv = p5.createCanvas(750, 750).parent(canvasParentRef);
        p5.background(200);

        cnv.mouseReleased((event) => {
            userSketch.push(userStrokes.slice(0, userStrokes.length));
            userStrokes.splice(0);
        })
    };

    const draw = (p5) => {
        // NOTE: Do not use setState in the draw function or in functions that are executed
        // in the draw function...
        // please use normal variables or class properties for these purposes
        p5.stroke(0);
        p5.strokeWeight(4);

        if (p5.mouseIsPressed
            && p5.mouseX <= p5.width && p5.mouseX >= 0
            && p5.mouseY <= p5.height && p5.mouseY >= 0) {
            userStrokes.push([p5.mouseX, p5.mouseY]);
            p5.line(p5.mouseX, p5.mouseY, p5.pmouseX, p5.pmouseY);
        }

    };

    function fetchSketch() {
        const data = {
            "sketch": userSketch,
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
                console.log(data)
            });
    }

    const handleEndSketchButtonClick = () => {
        fetchSketch();
        userSketch.splice(0)
    }

    return (<>
        <Sketch setup={setup} draw={draw} />
        <Button variant="contained" onClick={handleEndSketchButtonClick}>End Sketch</Button>
    </>
    );
};

export default Canvas;