/**
 * App Header Component
 *
 * Top navigation bar with:
 * - Branding/logo
 * - Navigation menu
 * - About/Info drawer
 * - Settings
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
  Settings as SettingsIcon,
  Description as DocsIcon,
  School as ScholarIcon,
  GitHub as GitHubIcon,
  Close as CloseIcon,
  ChevronRight as ChevronRightIcon,
  Email as EmailIcon,
} from '@mui/icons-material';
import CitationDialog from './CitationDialog';

interface AppHeaderProps {
  onNavigate?: (section: string) => void;
}

const AppHeader: React.FC<AppHeaderProps> = () => {
  const [mobileMenuAnchor, setMobileMenuAnchor] = useState<null | HTMLElement>(null);
  const [infoDrawerOpen, setInfoDrawerOpen] = useState(false);
  const [settingsDrawerOpen, setSettingsDrawerOpen] = useState(false);
  const [contactDrawerOpen, setContactDrawerOpen] = useState(false);
  const [citationDialogOpen, setCitationDialogOpen] = useState(false);
  const [activeCitationCategory, setActiveCitationCategory] = useState<'phonological' | 'lexical' | 'semantic' | 'affective' | 'embeddings' | 'data-sources' | null>(null);

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
        <Toolbar>
          {/* Logo / Branding */}
          <Typography
            variant="h6"
            component="div"
            sx={{
              flexGrow: 1,
              fontWeight: 700,
              display: 'flex',
              alignItems: 'center',
              gap: 1.5,
            }}
          >
            <Box
              component="img"
              src="/logo.png"
              alt="PhonoLex Logo"
              sx={{
                height: 32,
                width: 32,
                filter: 'brightness(0) invert(1)', // Make logo white on dark background
              }}
            />
            PhonoLex
            <Chip
              label="v2.0.0"
              size="small"
              sx={{
                height: 20,
                fontSize: '0.7rem',
                bgcolor: 'rgba(255,255,255,0.2)',
                color: 'white',
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
              startIcon={<SettingsIcon />}
              onClick={() => setSettingsDrawerOpen(true)}
            >
              Settings
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
            sx={{ display: { xs: 'flex', md: 'none' } }}
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
        <MenuItem onClick={() => { setSettingsDrawerOpen(true); handleMobileMenuClose(); }}>
          <SettingsIcon sx={{ mr: 1 }} /> Settings
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
            <strong>PhonoLex</strong> combines universal phonological features, learned phonological embeddings,
            and psycholinguistic norms for word analysis, similarity computation, and list generation.
          </Typography>

          <Divider sx={{ my: 3 }} />

          <Typography variant="h6" gutterBottom fontWeight={600}>
            Capabilities
          </Typography>
          <List dense>
            <ListItem>
              <ListItemText primary="Minimal pair generation with phonological constraints" />
            </ListItem>
            <ListItem>
              <ListItemText primary="Multi-property filtering: lexical, semantic, affective norms" />
            </ListItem>
            <ListItem>
              <ListItemText primary="Hierarchical phonological embeddings for similarity computation" />
            </ListItem>
            <ListItem>
              <ListItemText primary="Phonological complexity measures (WCM, MSH, syllable structure)" />
            </ListItem>
          </List>

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
            Research & Data Sources
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Click any category below to view detailed citations and references
          </Typography>

          <List sx={{ py: 0 }}>
            <ListItemButton onClick={() => handleCitationClick('phonological')} sx={{ borderRadius: 1 }}>
              <ListItemText
                primary="Phonological Complexity"
                secondary="WCM, distinctive features, phonotactic constraints"
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
                primary="Affective/Emotional Norms"
                secondary="Valence, arousal, dominance"
              />
              <ChevronRightIcon />
            </ListItemButton>
            <ListItemButton onClick={() => handleCitationClick('embeddings')} sx={{ borderRadius: 1 }}>
              <ListItemText
                primary="Phonological Embeddings"
                secondary="BERT-style architecture, hierarchical syllable structure"
              />
              <ChevronRightIcon />
            </ListItemButton>
            <ListItemButton onClick={() => handleCitationClick('data-sources')} sx={{ borderRadius: 1 }}>
              <ListItemText
                primary="Primary Data Sources"
                secondary="CMU Dictionary, PHOIBLE, SUBTLEX-US, ipa-dict"
              />
              <ChevronRightIcon />
            </ListItemButton>
          </List>

          <Divider sx={{ my: 3 }} />

          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', mt: 3 }}>
            <Button
              variant="outlined"
              startIcon={<GitHubIcon />}
              href="https://github.com/neumanns-workshop/PhonoLex"
              target="_blank"
            >
              View on GitHub
            </Button>
            <Button
              variant="outlined"
              startIcon={<DocsIcon />}
              href="https://docs.example.com"
              target="_blank"
            >
              Documentation
            </Button>
          </Box>

          <Box sx={{ mt: 3, p: 2, bgcolor: 'primary.50', borderRadius: 1 }}>
            <Typography variant="caption" color="text.secondary" align="center" display="block">
              PhonoLex v2.0.0 • Built with FastAPI + React + PostgreSQL + pgvector
              <br />
              Licensed under CC BY-SA 3.0 • Data resource for phonological research
            </Typography>
          </Box>
        </Box>
      </Drawer>

      {/* Settings Drawer */}
      <Drawer
        anchor="right"
        open={settingsDrawerOpen}
        onClose={() => setSettingsDrawerOpen(false)}
        sx={{
          '& .MuiDrawer-paper': {
            width: { xs: '100%', sm: 400 },
            maxWidth: '100%',
          },
        }}
      >
        <Box sx={{ p: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Typography variant="h5" fontWeight={700}>
              Settings
            </Typography>
            <IconButton onClick={() => setSettingsDrawerOpen(false)}>
              <CloseIcon />
            </IconButton>
          </Box>

          <Typography variant="body2" color="text.secondary">
            Settings panel coming soon! This will include:
          </Typography>
          <List dense>
            <ListItem>
              <ListItemText primary="• Display preferences" />
            </ListItem>
            <ListItem>
              <ListItemText primary="• Default filter values" />
            </ListItem>
            <ListItem>
              <ListItemText primary="• Column visibility presets" />
            </ListItem>
            <ListItem>
              <ListItemText primary="• IPA vs ARPAbet display" />
            </ListItem>
            <ListItem>
              <ListItemText primary="• Export format preferences" />
            </ListItem>
          </List>
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
              Contact & Support
            </Typography>
            <IconButton onClick={() => setContactDrawerOpen(false)}>
              <CloseIcon />
            </IconButton>
          </Box>

          <Typography variant="body2" paragraph color="text.secondary">
            Licensed under CC BY-SA 3.0. Contributions, bug reports, and feedback welcome.
          </Typography>

          <Divider sx={{ my: 3 }} />

          <Typography variant="h6" gutterBottom fontWeight={600}>
            Repository
          </Typography>
          <Typography variant="body2" paragraph color="text.secondary">
            Issues, features, contributions
          </Typography>
          <Button
            variant="outlined"
            startIcon={<GitHubIcon />}
            href="https://github.com/neumanns-workshop/PhonoLex"
            target="_blank"
            fullWidth
            sx={{ mb: 3 }}
          >
            View on GitHub
          </Button>

          <Divider sx={{ my: 3 }} />

          <Typography variant="h6" gutterBottom fontWeight={600}>
            Documentation
          </Typography>
          <Typography variant="body2" paragraph color="text.secondary">
            API reference, user guides, technical docs
          </Typography>
          <Button
            variant="outlined"
            startIcon={<DocsIcon />}
            href="https://docs.example.com"
            target="_blank"
            fullWidth
            sx={{ mb: 3 }}
          >
            Read the Docs
          </Button>

          <Divider sx={{ my: 3 }} />

          <Typography variant="h6" gutterBottom fontWeight={600}>
            Contact
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            General inquiries, collaboration, and support
          </Typography>
          <Button
            variant="outlined"
            startIcon={<EmailIcon />}
            href="mailto:social@neumannsworkshop.org"
            fullWidth
            sx={{ mb: 3 }}
          >
            social@neumannsworkshop.org
          </Button>

          <Typography variant="body2" color="text.secondary">
            Bug reports and feature requests via GitHub Issues.
          </Typography>

          <Box sx={{ mt: 4, p: 2, bgcolor: 'primary.50', borderRadius: 1 }}>
            <Typography variant="caption" color="text.secondary" align="center" display="block">
              PhonoLex v2.0.0 • Licensed under CC BY-SA 3.0
              <br />
              ShareAlike license required due to PHOIBLE data
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
