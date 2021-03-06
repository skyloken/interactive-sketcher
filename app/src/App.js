import { CssBaseline } from '@mui/material';
import { Box } from '@mui/system';
import './App.css';
import Canvas from './Canvas';


function App() {

  return (
    <div className="App">
      <CssBaseline />
        <Box m="auto">
          <Canvas />
        </Box>
    </div>
  );
}

export default App;
