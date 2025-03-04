import React from 'react';
import { List, ListItem, ListItemText, Typography, Box, Paper } from '@mui/material';

function History({ user }) {
  // Mock history data
  const mockHistory = [
    { id: 1, summary: "Summary: Legal agreement processed.", timestamp: new Date().toLocaleString() },
    { id: 2, summary: "Summary: Invoice anonymized.", timestamp: new Date(Date.now() - 3600000).toLocaleString() },
  ];

  return (
    <Paper elevation={3} sx={{ p: 4, borderRadius: 8 }}>
      <Typography variant="h4" gutterBottom sx={{ fontWeight: 'bold', color: '#1976d2' }}>
        History
      </Typography>
      <List>
        {mockHistory.map((item) => (
          <ListItem key={item.id} sx={{ borderBottom: '1px solid #eee', py: 2 }}>
            <ListItemText
              primary={item.summary}
              secondary={item.timestamp}
              primaryTypographyProps={{ fontWeight: 'medium', color: '#333' }}
              secondaryTypographyProps={{ color: '#666' }}
            />
          </ListItem>
        ))}
      </List>
    </Paper>
  );
}

export default History;