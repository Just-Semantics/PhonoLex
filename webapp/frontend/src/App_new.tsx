/**
 * PhonoLex - Modern Phonological Analysis Tool
 *
 * Flat card-based interface with progressive disclosure:
 * - Custom Word Lists
 * - Contrastive Sets (Minimal Pairs, Maximal Opposition, Multiple Opposition)
 * - Phonological Similarity
 * - Lookup (Words, Phonemes, Phoneme Comparison)
 */

import React from 'react';
import {
  ThemeProvider,
  CssBaseline,
  Container,
  Box,
  Typography,
  Link,
} from '@mui/material';
import {
  Build as BuildIcon,
  Search as SearchIcon,
  CompareArrows as ContrastiveIcon,
  Tune as SimilarityIcon,
  Description as TextAnalysisIcon,
} from '@mui/icons-material';
import { theme } from './theme/theme';
import api from './services/phonolexApi';

// Import components
import AppHeader from './components/AppHeader';
import ExpandableToolCard from './components/ExpandableToolCard';
import ContrastiveInterventionTool from './components/tools/ContrastiveInterventionTool';
import PhonologicalSimilarityTool from './components/tools/PhonologicalSimilarityTool';
import SearchTool from './components/tools/SearchTool';
import TextAnalysisTool from './components/tools/TextAnalysisTool';
import Builder from './components/Builder';

const App: React.FC = () => {
  // Pre-load data at app startup to avoid lag when opening tools
  React.useEffect(() => {
    // Trigger data load in background
    api.getStats().catch(err => {
      console.error('Failed to pre-load data:', err);
    });
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />

      {/* App Header with Navigation */}
      <AppHeader onNavigate={(section) => console.log('Navigate to:', section)} />

      {/* Main Content */}
      <Container
        maxWidth="lg"
        sx={{
          mt: { xs: 1.5, sm: 2, md: 3 },
          mb: { xs: 3, sm: 4 },
          px: { xs: 1.5, sm: 2, md: 3 },
        }}
        role="main"
      >
        {/* Tool Cards - Flat, Progressive Disclosure */}
        <Box sx={{ maxWidth: 900, mx: 'auto' }}>

          {/* Custom Word Lists - THE POWER TOOL */}
          <ExpandableToolCard
            icon={<BuildIcon />}
            title="Custom Word Lists"
            description="Pattern matching with phonological, lexical, semantic, and affective property filters"
            color="primary.main"
          >
            <Builder />
          </ExpandableToolCard>

          {/* Contrastive Sets */}
          <ExpandableToolCard
            icon={<ContrastiveIcon />}
            title="Contrastive Sets"
            description="Research-based phonological interventions: minimal pairs, maximal opposition, and multiple opposition"
            color="secondary.main"
          >
            <ContrastiveInterventionTool />
          </ExpandableToolCard>

          {/* Phonological Similarity */}
          <ExpandableToolCard
            icon={<SimilarityIcon />}
            title="Phonological Similarity"
            description="Find similar words with adjustable onset, nucleus, and coda weights - perfect for rhymes, alliteration, and more"
            color="#D4A747"
          >
            <PhonologicalSimilarityTool />
          </ExpandableToolCard>

          {/* Text Analysis */}
          <ExpandableToolCard
            icon={<TextAnalysisIcon />}
            title="Text Analysis"
            description="Analyze passages for readability across phonological, lexical, semantic, and affective dimensions with interactive highlighting"
            color="#4CAF50"
          >
            <TextAnalysisTool />
          </ExpandableToolCard>

          {/* Lookup */}
          <ExpandableToolCard
            icon={<SearchIcon />}
            title="Lookup"
            description="Look up words and phonemes, compare phoneme features, or search by distinctive features"
            color="#7A7A78"
          >
            <SearchTool />
          </ExpandableToolCard>

        </Box>

        {/* Footer */}
        <Box
          component="footer"
          sx={{
            mt: 6,
            pt: 3,
            borderTop: 1,
            borderColor: 'divider',
            textAlign: 'center',
            px: { xs: 2, sm: 0 },
          }}
          role="contentinfo"
        >
          <Typography variant="body2" color="text.secondary" sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, mb: 1 }}>
            © {new Date().getFullYear()} Just Semantics. Provided as-is without warranty.
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>
            Licensed under CC BY-SA 3.0 •{' '}
            <Link href="/privacy" underline="hover">Privacy</Link>
            {' • '}
            <Link href="/terms" underline="hover">Terms</Link>
            {' • '}
            <Link href="https://github.com/Just-Semantics/PhonoLex" target="_blank" rel="noopener noreferrer" underline="hover">
              GitHub
            </Link>
          </Typography>
        </Box>
      </Container>
    </ThemeProvider>
  );
};

export default App;
