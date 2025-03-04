import React, { useState } from 'react';
import { Button, TextField, Checkbox, FormControlLabel, Typography, Box, Paper } from '@mui/material';

function Upload({ user }) {
  const [file, setFile] = useState(null);
  const [anonymize, setAnonymize] = useState(true);
  const [summarize, setSummarize] = useState(true);
  const [summary, setSummary] = useState('');
  const [downloadUrl, setDownloadUrl] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!file) {
      alert('Please select a file.');
      return;
    }

    // Mock processing
    const mockText = "This is a sample legal document with sensitive data like John Doe, 123 Main Street, Mumbai.";
    setSummary(summarize ? "Summary: This document contains legal information and sensitive data." : '');
    if (anonymize) {
      const anonText = mockText.replace(/John Doe/g, 'xxxx').replace(/123 Main Street, Mumbai/g, 'xxxx');
      const blob = new Blob([anonText], { type: file.type });
      const url = window.URL.createObjectURL(blob);
      setDownloadUrl(url);
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 4, borderRadius: 8 }}>
      <Typography variant="h4" gutterBottom sx={{ fontWeight: 'bold', color: '#1976d2' }}>
        Upload Document
      </Typography>
      <form onSubmit={handleSubmit}>
        <TextField
          type="file"
          inputProps={{ accept: '.pdf,.docx,.txt' }}
          onChange={(e) => setFile(e.target.files[0])}
          fullWidth
          sx={{ mb: 3, bgcolor: '#f5f5f5', borderRadius: 4 }}
        />
        <Box sx={{ display: 'flex', justifyContent: 'center', gap: 4, mb: 3 }}>
          <FormControlLabel
            control={<Checkbox checked={anonymize} onChange={(e) => setAnonymize(e.target.checked)} />}
            label="Anonymize"
            sx={{ color: '#555' }}
          />
          <FormControlLabel
            control={<Checkbox checked={summarize} onChange={(e) => setSummarize(e.target.checked)} />}
            label="Summarize"
            sx={{ color: '#555' }}
          />
        </Box>
        <Button
          type="submit"
          variant="contained"
          color="primary"
          fullWidth
          sx={{ py: 1.5, fontSize: '1.1rem', borderRadius: 6 }}
        >
          Process
        </Button>
      </form>
      {summary && (
        <Box mt={4} sx={{ bgcolor: '#e8f0fe', p: 3, borderRadius: 6 }}>
          <Typography variant="h6" sx={{ fontWeight: 'medium', color: '#1976d2' }}>
            Summary
          </Typography>
          <Typography sx={{ mt: 1, color: '#333' }}>{summary}</Typography>
        </Box>
      )}
      {downloadUrl && (
        <Button
          variant="contained"
          color="secondary"
          href={downloadUrl}
          download={file ? `anon_${file.name}` : 'anon_document.txt'}
          sx={{ mt: 3, py: 1.5, fontSize: '1.1rem', borderRadius: 6 }}
        >
          Download Anonymized Document
        </Button>
      )}
    </Paper>
  );
}

export default Upload;