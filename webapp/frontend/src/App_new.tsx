/**
 * PhonoLex - Modern Phonological Analysis Tool
 *
 * Flat card-based interface with progressive disclosure:
 * - Minimal Pairs
 * - Rhyme Sets
 * - Word Search
 * - Similarity Search
 * - Custom Builder
 * - Phoneme Comparison
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
  Compare as CompareIcon,
  SwapHoriz as MinimalPairIcon,
  MusicNote as RhymeIcon,
  TuneOutlined as FilterIcon,
} from '@mui/icons-material';
import { theme } from './theme/theme';

// Import components
import AppHeader from './components/AppHeader';
import ExpandableToolCard from './components/ExpandableToolCard';
import MinimalPairsTool from './components/tools/MinimalPairsTool';
import RhymeSetsTool from './components/tools/RhymeSetsTool';
import NormFilteredListsTool from './components/tools/NormFilteredListsTool';
import SearchTool from './components/tools/SearchTool';
import Builder from './components/Builder';
import Compare from './components/Compare';

const App: React.FC = () => {

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />

      {/* App Header with Navigation */}
      <AppHeader onNavigate={(section) => console.log('Navigate to:', section)} />

      {/* Main Content */}
      <Container maxWidth="lg" sx={{ mt: { xs: 2, sm: 3 }, mb: 4, px: { xs: 2, sm: 3 } }} role="main">
        {/* Tool Cards - Flat, Progressive Disclosure */}
        <Box sx={{ maxWidth: 900, mx: 'auto' }}>

          {/* Custom Builder - THE POWER TOOL */}
          <ExpandableToolCard
            icon={<BuildIcon />}
            title="Custom Word List Builder"
            description="Pattern matching with phonological, lexical, semantic, and affective property filters"
            defaultExpanded={true}
            color="primary.main"
          >
            <Builder />
          </ExpandableToolCard>

          {/* Minimal Pairs */}
          <ExpandableToolCard
            icon={<MinimalPairIcon />}
            title="Minimal Pairs"
            description="Word pairs differing by single phoneme for contrastive analysis"
            color="secondary.main"
          >
            <MinimalPairsTool />
          </ExpandableToolCard>

          {/* Norm-Filtered Lists */}
          <ExpandableToolCard
            icon={<FilterIcon />}
            title="Norm-Filtered Lists"
            description="Filter by psycholinguistic norms: frequency, AoA, imageability, valence"
            color="success.main"
          >
            <NormFilteredListsTool />
          </ExpandableToolCard>

          {/* Rhyme Sets */}
          <ExpandableToolCard
            icon={<RhymeIcon />}
            title="Rhyme Sets"
            description="Generate rhyming word sets with configurable match criteria"
            color="#D4A747"
          >
            <RhymeSetsTool />
          </ExpandableToolCard>

          {/* Phoneme Comparison */}
          <ExpandableToolCard
            icon={<CompareIcon />}
            title="Phoneme Comparison"
            description="Compare distinctive features and compute phonological distance"
            color="info.main"
          >
            <Compare />
          </ExpandableToolCard>

          {/* Word Search */}
          <ExpandableToolCard
            icon={<SearchIcon />}
            title="Search"
            description="Lookup words, phonemes, or compute phonological similarity"
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
