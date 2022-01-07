import Sketch from "react-p5";

function Canvas() {

    const setup = (p5, canvasParentRef) => {
        // use parent to render the canvas in this ref
        // (without that p5 will render the canvas outside of your component)
        p5.createCanvas(750, 750).parent(canvasParentRef);
        p5.background(200);
    };

    const draw = (p5) => {
        // NOTE: Do not use setState in the draw function or in functions that are executed
        // in the draw function...
        // please use normal variables or class properties for these purposes
        p5.stroke(0);
        if (p5.mouseIsPressed === true) {
            p5.line(p5.mouseX, p5.mouseY, p5.pmouseX, p5.pmouseY);
        }
    };

    return <Sketch setup={setup} draw={draw} />;
};

export default Canvas;