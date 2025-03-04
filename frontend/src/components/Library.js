import React from 'react';
import { Typography, Box, Paper } from '@mui/material';

function Library({ user }) {
  return (
    <Paper elevation={3} sx={{ p: 4, borderRadius: 8 }}>
      <Typography variant="h4" gutterBottom sx={{ fontWeight: 'bold', color: '#1976d2' }}>
        Library
      </Typography>
      <Typography sx={{ color: '#555' }}>
        Your saved anonymized documents will appear here once integrated with the backend.
      </Typography>
    </Paper>
  );
}

export default Library;