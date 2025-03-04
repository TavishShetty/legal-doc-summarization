import React, { useState } from 'react';
import { Routes, Route, Link } from 'react-router-dom';
import { AppBar, Toolbar, Typography, Button, Container } from '@mui/material';
import Upload from './components/Upload';
import History from './components/History';
import Library from './components/Library';
import './styles/App.css';

function App() {
  const [mockUser, setMockUser] = useState({ name: 'Test User' });  // Mock logged-in state

  return (
    <div className="App">
      <AppBar position="static" sx={{ background: '#1976d2' }}>
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            LegalDocs AI
          </Typography>
          <Button color="inherit" component={Link} to="/upload">Upload</Button>
          <Button color="inherit" component={Link} to="/history">History</Button>
          <Button color="inherit" component={Link} to="/library">Library</Button>
          <Button color="inherit" onClick={() => setMockUser(null)}>Logout</Button>
        </Toolbar>
      </AppBar>
      <Container sx={{ mt: 4 }}>
        <Routes>
          <Route path="/" element={<Upload user={mockUser} />} />
          <Route path="/upload" element={<Upload user={mockUser} />} />
          <Route path="/history" element={<History user={mockUser} />} />
          <Route path="/library" element={<Library user={mockUser} />} />
        </Routes>
      </Container>
    </div>
  );
}

export default App;