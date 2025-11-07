/**
 * App Header Component
 *
 * Top navigation bar with:
 * - Branding/logo
 * - Navigation menu
 * - About/Info drawer
 * - Contact drawer
 */

import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Menu,
  MenuItem,
  Drawer,
  Box,
  Divider,
  List,
  ListItem,
  ListItemText,
  Button,
  Chip,
  ListItemButton,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Info as InfoIcon,
  Description as DocsIcon,
  GitHub as GitHubIcon,
  Close as CloseIcon,
  ChevronRight as ChevronRightIcon,
  Email as EmailIcon,
} from '@mui/icons-material';
import CitationDialog from './CitationDialog';
import ContactForm from './ContactForm';

interface AppHeaderProps {
  onNavigate?: (section: string) => void;
}

const AppHeader: React.FC<AppHeaderProps> = () => {
  const [mobileMenuAnchor, setMobileMenuAnchor] = useState<null | HTMLElement>(null);
  const [infoDrawerOpen, setInfoDrawerOpen] = useState(false);
  const [contactDrawerOpen, setContactDrawerOpen] = useState(false);
  const [citationDialogOpen, setCitationDialogOpen] = useState(false);
  const [activeCitationCategory, setActiveCitationCategory] = useState<'phonological' | 'lexical' | 'semantic' | 'affective' | 'embeddings' | 'data-sources' | 'clinical-interventions' | null>(null);

  const handleCitationClick = (category: typeof activeCitationCategory) => {
    setActiveCitationCategory(category);
    setCitationDialogOpen(true);
  };

  const handleMobileMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setMobileMenuAnchor(event.currentTarget);
  };

  const handleMobileMenuClose = () => {
    setMobileMenuAnchor(null);
  };

  return (
    <>
      <AppBar position="sticky" elevation={1} sx={{ bgcolor: 'primary.main' }}>
        <Toolbar sx={{ minHeight: { xs: 56, sm: 64 }, px: { xs: 1, sm: 2 } }}>
          {/* Logo / Branding */}
          <Typography
            variant="h6"
            component="div"
            sx={{
              flexGrow: 1,
              fontWeight: 700,
              display: 'flex',
              alignItems: 'center',
              gap: { xs: 0.75, sm: 1.5 },
              fontSize: { xs: '1rem', sm: '1.25rem' },
            }}
          >
            <Box
              component="img"
              src="/logo.png"
              alt="PhonoLex Logo"
              sx={{
                height: { xs: 28, sm: 32 },
                width: { xs: 28, sm: 32 },
                filter: 'brightness(0) invert(1)', // Make logo white on dark background
                flexShrink: 0,
              }}
            />
            PhonoLex
            <Chip
              label="v2.2.0-beta"
              size="small"
              sx={{
                display: { xs: 'none', sm: 'inline-flex' },
                height: { sm: 20 },
                fontSize: { sm: '0.7rem' },
                bgcolor: 'rgba(255,255,255,0.2)',
                color: 'white',
                '& .MuiChip-label': {
                  px: { sm: 1 },
                },
              }}
            />
          </Typography>

          {/* Desktop Navigation */}
          <Box sx={{ display: { xs: 'none', md: 'flex' }, gap: 1 }}>
            <Button
              color="inherit"
              startIcon={<InfoIcon />}
              onClick={() => setInfoDrawerOpen(true)}
            >
              Info
            </Button>
            <Button
              color="inherit"
              startIcon={<DocsIcon />}
              component="a"
              href="https://phonolex.readthedocs.io"
              target="_blank"
              rel="noopener noreferrer"
            >
              Docs
            </Button>
            <Button
              color="inherit"
              startIcon={<EmailIcon />}
              onClick={() => setContactDrawerOpen(true)}
            >
              Contact
            </Button>
          </Box>

          {/* Mobile Menu */}
          <IconButton
            color="inherit"
            aria-label="menu"
            onClick={handleMobileMenuOpen}
            sx={{
              display: { xs: 'flex', md: 'none' },
              minWidth: 44,
              minHeight: 44,
              p: 1,
            }}
          >
            <MenuIcon />
          </IconButton>
        </Toolbar>
      </AppBar>

      {/* Mobile Menu Dropdown */}
      <Menu
        anchorEl={mobileMenuAnchor}
        open={Boolean(mobileMenuAnchor)}
        onClose={handleMobileMenuClose}
      >
        <MenuItem onClick={() => { setInfoDrawerOpen(true); handleMobileMenuClose(); }}>
          <InfoIcon sx={{ mr: 1 }} /> Info
        </MenuItem>
        <MenuItem
          component="a"
          href="https://phonolex.readthedocs.io"
          target="_blank"
          rel="noopener noreferrer"
          onClick={handleMobileMenuClose}
        >
          <DocsIcon sx={{ mr: 1 }} /> Docs
        </MenuItem>
        <MenuItem onClick={() => { setContactDrawerOpen(true); handleMobileMenuClose(); }}>
          <EmailIcon sx={{ mr: 1 }} /> Contact
        </MenuItem>
      </Menu>

      {/* Info Drawer (combines About + Research) */}
      <Drawer
        anchor="right"
        open={infoDrawerOpen}
        onClose={() => setInfoDrawerOpen(false)}
        sx={{
          '& .MuiDrawer-paper': {
            width: { xs: '100%', sm: 500 },
            maxWidth: '100%',
          },
        }}
      >
        <Box sx={{ p: { xs: 2, sm: 3 }, height: '100%', overflow: 'auto' }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Typography variant="h5" fontWeight={700}>
              Info
            </Typography>
            <IconButton onClick={() => setInfoDrawerOpen(false)}>
              <CloseIcon />
            </IconButton>
          </Box>

          {/* About Section */}
          <Typography variant="h6" gutterBottom fontWeight={600}>
            About PhonoLex
          </Typography>

          <Typography variant="body1" paragraph>
            <strong>PhonoLex</strong> combines universal phonological features (Phoible), phoneme-sequence soft
            Levenshtein similarity, and psycholinguistic norms for word analysis, similarity computation, and list generation.
          </Typography>

          <Divider sx={{ my: 3 }} />

          <Typography variant="h6" gutterBottom fontWeight={600}>
            Tools
          </Typography>
          <List dense>
            <ListItem>
              <ListItemText
                primary="Custom Word Lists"
                secondary="Pattern matching with phonological, lexical, semantic, and affective property filters"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Contrastive Sets"
                secondary="Minimal pairs, maximal opposition, and multiple opposition for speech therapy"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Phonological Similarity"
                secondary="Adjustable onset/nucleus/coda weights for rhymes, alliteration, and consonance"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Lookup"
                secondary="Word lookup, phoneme features, phoneme comparison, and feature-based phoneme search"
              />
            </ListItem>
          </List>

          <Divider sx={{ my: 3 }} />

          <Typography variant="h6" gutterBottom fontWeight={600}>
            How It Works
          </Typography>

          <Typography variant="body2" color="text.secondary" paragraph>
            PhonoLex v2.3 uses <strong>phoneme-sequence soft Levenshtein distance</strong> to preserve the
            sequential structure of consonant clusters and diphthongs without averaging.
          </Typography>

          <Box component="ol" sx={{ pl: 3, mb: 2, '& li': { mb: 1 } }}>
            <Typography component="li" variant="body2" color="text.secondary">
              <strong>Phase 1 (Raw Features):</strong> Universal phonological features from PHOIBLE database
              (38 distinctive features covering 2,716 languages)
            </Typography>
            <Typography component="li" variant="body2" color="text.secondary">
              <strong>Phase 2 (Normalized Vectors):</strong> Continuous 76-dim vectors for consonants/monophthongs,
              152-dim trajectory vectors for diphthongs
            </Typography>
            <Typography component="li" variant="body2" color="text.secondary">
              <strong>Phase 3 (Syllable Structures):</strong> Onset/nucleus/coda represented as <strong>sequences of
              phoneme vectors</strong> (NO averaging!). Example: "crest" onset = [k_vec, ɹ_vec] (2 vectors), not averaged.
            </Typography>
          </Box>

          <Typography variant="body2" color="text.secondary" paragraph>
            <strong>Three-Level Similarity Hierarchy:</strong>
          </Typography>

          <Box component="ul" sx={{ pl: 3, mb: 2, '& li': { mb: 0.5 } }}>
            <Typography component="li" variant="body2" color="text.secondary">
              <strong>Phoneme level:</strong> Cosine similarity between phoneme vectors (76-dim or 152-dim)
            </Typography>
            <Typography component="li" variant="body2" color="text.secondary">
              <strong>Component level:</strong> Soft Levenshtein distance on phoneme sequences within onset/nucleus/coda
            </Typography>
            <Typography component="li" variant="body2" color="text.secondary">
              <strong>Syllable level:</strong> Weighted average of component similarities (user-adjustable weights)
            </Typography>
            <Typography component="li" variant="body2" color="text.secondary">
              <strong>Word level:</strong> Soft Levenshtein distance on syllable sequences
            </Typography>
          </Box>

          <Typography variant="body2" color="text.secondary" paragraph>
            <strong>Data Sources:</strong> CMU Pronouncing Dictionary (125K words), PHOIBLE (phonological features),
            SUBTLEX-US (frequency), Glasgow Norms, Brysbaert et al., Warriner et al., and additional psycholinguistic
            datasets. Filtered vocabulary: 24,744 English words with comprehensive phonological and psycholinguistic properties.
          </Typography>

          <Divider sx={{ my: 3 }} />

          <Typography variant="h6" gutterBottom fontWeight={600}>
            Disclaimer
          </Typography>
          <Box sx={{ bgcolor: 'warning.50', p: 2, borderRadius: 1, border: 1, borderColor: 'warning.light' }}>
            <Typography variant="body2" paragraph sx={{ mb: 0 }}>
              While reasonable effort has been made to ensure data accuracy and system reliability, PhonoLex may contain:
            </Typography>
            <Typography variant="body2" component="ul" sx={{ mt: 1, mb: 0, pl: 2 }}>
              <li>Implementation errors or bugs in data processing or algorithms</li>
              <li>Inaccuracies, biases, or limitations inherent in the source datasets</li>
              <li>Fundamental constraints of the computational approaches employed</li>
            </Typography>
            <Typography variant="body2" sx={{ mt: 1, mb: 0 }}>
              This resource is provided "as-is" for research and educational purposes. Users should independently verify results for any critical applications.
            </Typography>
          </Box>

          <Divider sx={{ my: 3 }} />

          <Typography variant="h6" gutterBottom fontWeight={600}>
            References
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Click any category below to view detailed citations
          </Typography>

          <List sx={{ py: 0 }}>
            {/* Data Sources & Methodology */}
            <Typography variant="caption" color="text.secondary" sx={{ px: 2, py: 1, display: 'block', fontWeight: 600 }}>
              Data Sources & Methodology
            </Typography>
            <ListItemButton onClick={() => handleCitationClick('data-sources')} sx={{ borderRadius: 1 }}>
              <ListItemText
                primary="Primary Data Sources"
                secondary="CMU Dictionary, PHOIBLE, SUBTLEX-US"
              />
              <ChevronRightIcon />
            </ListItemButton>
            <ListItemButton onClick={() => handleCitationClick('embeddings')} sx={{ borderRadius: 1 }}>
              <ListItemText
                primary="Phonological Similarity Architecture"
                secondary="Phoneme-sequence soft Levenshtein with Phoible features"
              />
              <ChevronRightIcon />
            </ListItemButton>

            <Divider sx={{ my: 1.5 }} />

            {/* Psycholinguistic Measurements */}
            <Typography variant="caption" color="text.secondary" sx={{ px: 2, py: 1, display: 'block', fontWeight: 600 }}>
              Psycholinguistic Measurements
            </Typography>
            <ListItemButton onClick={() => handleCitationClick('phonological')} sx={{ borderRadius: 1 }}>
              <ListItemText
                primary="Phonological Complexity"
                secondary="WCM, MSH, distinctive features"
              />
              <ChevronRightIcon />
            </ListItemButton>
            <ListItemButton onClick={() => handleCitationClick('lexical')} sx={{ borderRadius: 1 }}>
              <ListItemText
                primary="Lexical Properties"
                secondary="Word frequency, age of acquisition"
              />
              <ChevronRightIcon />
            </ListItemButton>
            <ListItemButton onClick={() => handleCitationClick('semantic')} sx={{ borderRadius: 1 }}>
              <ListItemText
                primary="Semantic Properties"
                secondary="Imageability, familiarity, concreteness"
              />
              <ChevronRightIcon />
            </ListItemButton>
            <ListItemButton onClick={() => handleCitationClick('affective')} sx={{ borderRadius: 1 }}>
              <ListItemText
                primary="Affective Properties"
                secondary="Valence, arousal, dominance"
              />
              <ChevronRightIcon />
            </ListItemButton>

            <Divider sx={{ my: 1.5 }} />

            {/* Clinical Applications */}
            <Typography variant="caption" color="text.secondary" sx={{ px: 2, py: 1, display: 'block', fontWeight: 600 }}>
              Clinical Applications
            </Typography>
            <ListItemButton onClick={() => handleCitationClick('clinical-interventions')} sx={{ borderRadius: 1 }}>
              <ListItemText
                primary="Intervention Approaches"
                secondary="Minimal pairs, maximal opposition, multiple opposition"
              />
              <ChevronRightIcon />
            </ListItemButton>
          </List>

          <Divider sx={{ my: 3 }} />

          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', mt: 3 }}>
            <Button
              variant="outlined"
              startIcon={<GitHubIcon />}
              href="https://github.com/Just-Semantics/PhonoLex"
              target="_blank"
            >
              View on GitHub
            </Button>
            <Button
              variant="outlined"
              startIcon={<DocsIcon />}
              href="https://phonolex.readthedocs.io"
              target="_blank"
            >
              Documentation
            </Button>
          </Box>

          <Box sx={{ mt: 3, p: 2, bgcolor: 'primary.50', borderRadius: 1 }}>
            <Typography variant="caption" color="text.secondary" align="center" display="block">
              PhonoLex v2.2.0-beta • Built with React + TypeScript (Client-Side)
              <br />
              Licensed under CC BY-SA 3.0 • Data resource for phonological research
            </Typography>
          </Box>
        </Box>
      </Drawer>

      {/* Contact Drawer */}
      <Drawer
        anchor="right"
        open={contactDrawerOpen}
        onClose={() => setContactDrawerOpen(false)}
        sx={{
          '& .MuiDrawer-paper': {
            width: { xs: '100%', sm: 500 },
            maxWidth: '100%',
          },
        }}
      >
        <Box sx={{ p: { xs: 2, sm: 3 } }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Typography variant="h5" fontWeight={700}>
              Contact
            </Typography>
            <IconButton onClick={() => setContactDrawerOpen(false)}>
              <CloseIcon />
            </IconButton>
          </Box>

          {/* Contact Form */}
          <ContactForm />

          <Divider sx={{ my: 3 }} />

          <Typography variant="h6" gutterBottom fontWeight={600}>
            Other Ways to Reach Us
          </Typography>

          <Typography variant="body2" color="text.secondary" paragraph>
            Prefer email? Reach us directly at:
          </Typography>
          <Button
            variant="outlined"
            startIcon={<EmailIcon />}
            href="mailto:contact@justsemantics.net"
            fullWidth
            sx={{ mb: 2 }}
          >
            contact@justsemantics.net
          </Button>

          <Typography variant="body2" color="text.secondary" paragraph>
            For bug reports and feature requests:
          </Typography>
          <Button
            variant="outlined"
            startIcon={<GitHubIcon />}
            href="https://github.com/Just-Semantics/PhonoLex/issues"
            target="_blank"
            fullWidth
            sx={{ mb: 3 }}
          >
            GitHub Issues
          </Button>

          <Box sx={{ mt: 4, p: 2, bgcolor: 'info.50', borderRadius: 1, border: 1, borderColor: 'info.light' }}>
            <Typography variant="caption" color="text.secondary" align="center" display="block">
              PhonoLex v2.2.0-beta • Licensed under CC BY-SA 3.0
            </Typography>
          </Box>
        </Box>
      </Drawer>

      {/* Citation Dialog */}
      <CitationDialog
        open={citationDialogOpen}
        onClose={() => setCitationDialogOpen(false)}
        category={activeCitationCategory}
      />
    </>
  );
};

export default AppHeader;
