import { useState } from 'react';
import { ThemeProvider, CssBaseline, AppBar, Toolbar, Typography, Container } from '@mui/material';
import theme from './theme';
import './styles.css';
import ImageProcessor from './components/ImageProcessor';

function App() {
  const [count, setCount] = useState(0);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AppBar position="sticky">
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            Stampit WebUI
          </Typography>
        </Toolbar>
      </AppBar>
      <Container maxWidth="lg" sx={{ py: 2 }}>
        <ImageProcessor />
      </Container>
    </ThemeProvider>
  );
}

export default App;
