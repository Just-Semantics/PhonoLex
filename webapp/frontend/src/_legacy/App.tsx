/**
 * PhonoLex Web App
 *
 * Main application component for phonological word list generation
 */

import React from 'react';
import { ThemeProvider, CssBaseline, Container, Box, Typography, Paper } from '@mui/material';
import { theme } from '@theme/theme';
import PatternBuilder from '@components/PatternBuilder';
import WordResults from '@components/WordResults';

const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="xl" sx={{ py: 4 }}>
        {/* Header */}
        <Paper
          elevation={0}
          sx={{
            p: 3,
            mb: 3,
            background: 'linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)',
            color: 'white',
          }}
        >
          <Typography variant="h3" fontWeight={700}>
            PhonoLex
          </Typography>
          <Typography variant="h6" sx={{ opacity: 0.9, mt: 1 }}>
            Phonological Word List Generator for SLPs
          </Typography>
        </Paper>

        {/* Main Content */}
        <Box sx={{ display: 'grid', gap: 3 }}>
          <PatternBuilder />
          <WordResults />
        </Box>

        {/* Footer */}
        <Box sx={{ mt: 4, textAlign: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            Built with hierarchical phonological embeddings • 125,764 English words • Universal
            Phoible features
          </Typography>
        </Box>
      </Container>
    </ThemeProvider>
  );
};

export default App;
