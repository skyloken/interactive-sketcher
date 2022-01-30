import { Button, Checkbox, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle, Stack, TextField } from "@mui/material";
import React from "react";
import Sketch from "react-p5";
import { v4 as uuidv4 } from 'uuid';


class Canvas extends React.Component {

    constructor(props) {
        super(props);

        const params = new URLSearchParams(document.location.search.substring(1));
        let mode = 3;
        let paramMode = params.get("mode");
        if (paramMode) {
            paramMode = parseInt(paramMode)
            switch (paramMode) {
                case 1:
                case 2:
                case 3:
                case 4:
                    mode = paramMode;
                    break;
                default:
                    break;
            }
        }

        this.userLine = [];
        this.userLines = [];
        this.previousSketches = [];
        this.agentLines = [];
        this.isUserTurn = true;
        this.seqId = uuidv4();

        this.showMessage = true;
        this.showCompositionCheckbox = false;
        this.mode = mode;

        switch(mode) {
            case 1:
            case 2:
                this.showCategory = false
                break
            case 3:
            case 4:
                this.showCategory = true
                break
            default:
                this.showCategory = true
                break
        }

        this.state = {
            agentMessage: "Agent: Draw your first sketch!",
            isShowComposition: false,
            turn_num: 1,
            dialogOpen: false,
            formOpen: true,
            username: "",
            inputError: false,
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

        this.clearCanvas = () => {
            p5.clear();
            p5.background(230);
        }
    };

    draw = (p5) => {
        // NOTE: Do not use setState in the draw function or in functions that are executed
        // in the draw function...
        // please use normal variables or class properties for these purposes
        p5.stroke(0);
        p5.strokeWeight(4);

        if (!this.state.dialogOpen && !this.state.formOpen && this.isUserTurn) {
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
            "seqId": this.seqId,
            "username": this.state.username,
            "previousSketches": this.previousSketches,
            "userLines": this.userLines,
            "mode": this.mode,
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
                this.previousSketches = data.previousSketches;
                if (data.nextLines.length) {
                    this.agentLines = data.nextLines;
                } else {
                    this.isUserTurn = true
                }
                if (this.state.isShowComposition) {
                    this.drawNextSketch(data.nextSketch.name, data.nextSketch.position);
                }
                let message = ""
                if (this.showCategory) {
                    message = `Agent: I drew <b>${data.nextSketch.name}</b>, now it's your turn!`
                } else {
                    message = `Agent: I drew something, now it's your turn!`
                }
                this.setState({
                    turn_num: this.state.turn_num + 1,
                    agentMessage: message
                });
            });
    }

    handleEndSketchButtonClick = () => {
        if (!this.userLines.length) {
            return;
        }
        this.isUserTurn = false;
        this.setState({
            turn_num: this.state.turn_num + 1,
            agentMessage: `Agent: I'm thinking, so give me a minute...`
        });
        this.fetchSketch();
        this.userLines.splice(0);
    }

    handleCheckboxChange = () => {
        this.setState({
            isShowComposition: !this.state.isShowComposition,
        })
    }

    handleResetButtonClick = () => {
        this.clearCanvas()

        this.userLine = [];
        this.userLines = [];
        this.previousSketches = [];
        this.agentLines = [];
        this.isUserTurn = true;
        this.seqId = uuidv4();

        this.setState({
            agentMessage: "Agent: Draw your first sketch!",
            turn_num: 1,
            dialogOpen: false,
        });
    }

    handleClickOpen = () => {
        this.setState({
            dialogOpen: true,
        });
    }

    handleClose = () => {
        this.setState({
            dialogOpen: false,
        });
    }

    handleFormClose = () => {
        if (this.state.username !== "") {
            this.setState({
                formOpen: false,
                inputError: false,
            });
        } else {
            this.setState({
                inputError: true,
            })
        }
    };

    handleNameChange = (e) => {
        this.setState({
            username: e.target.value
        });
    }

    render() {
        return <>
            <p>Current turn: {this.state.turn_num} ({this.state.turn_num % 2 !== 0 ? "Your turn" : "Agent's turn"})</p>

            <Sketch setup={this.setup} draw={this.draw} />

            {this.showMessage && <p><span dangerouslySetInnerHTML={{ __html: this.state.agentMessage }} /></p>}

            <Stack spacing={2} direction="row" justifyContent="center">
                <Button variant="contained" onClick={this.handleEndSketchButtonClick}>Next</Button>
                <Button variant="contained" color="error" onClick={this.handleClickOpen}>Reset</Button>
                {this.showCompositionCheckbox && <Checkbox checked={this.state.isShowComposition} onChange={this.handleCheckboxChange} />}
            </Stack>

            <Dialog
                open={this.state.dialogOpen}
                onClose={this.handleClose}
                aria-labelledby="alert-dialog-title"
                aria-describedby="alert-dialog-description"
            >
                <DialogContent>
                    <DialogContentText id="alert-dialog-description">
                        Do you want to reset this collaborative sketch?
                    </DialogContentText>
                </DialogContent>
                <DialogActions>
                    <Button variant="outlined" onClick={this.handleClose}>Cancel</Button>
                    <Button variant="contained" onClick={this.handleResetButtonClick} color="error" autoFocus>Reset</Button>
                </DialogActions>
            </Dialog>

            <Dialog open={this.state.formOpen}>
                <DialogTitle id="alert-dialog-title">
                    Welcome to CoSketcher
                </DialogTitle>
                <DialogContent>
                    <DialogContentText>
                        Please enter your name.
                    </DialogContentText>
                    <TextField
                        required
                        autoFocus
                        error={this.state.inputError}
                        margin="dense"
                        id="name"
                        label="Name"
                        variant="standard"
                        value={this.state.username}
                        onChange={this.handleNameChange}
                    />
                </DialogContent>
                <DialogActions>
                    <Button variant="contained" onClick={this.handleFormClose}>OK</Button>
                </DialogActions>
            </Dialog>
        </>;
    }

}

export default Canvas;