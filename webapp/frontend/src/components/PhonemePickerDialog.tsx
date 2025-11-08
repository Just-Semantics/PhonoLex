/**
 * Phoneme Picker Dialog
 *
 * Visual IPA chart for selecting phonemes by clicking
 * Dynamically loads phonemes from database
 */

import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  Box,
  Typography,
  Button,
  Tabs,
  Tab,
  IconButton,
  CircularProgress,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import { Close as CloseIcon } from '@mui/icons-material';
import api from '../services/phonolexApi';

interface PhonemePickerDialogProps {
  open: boolean;
  onClose: () => void;
  onSelect: (phoneme: string) => void;
  filter?: 'sonorants' | 'obstruents';
}

interface Phoneme {
  ipa: string;
  type?: string;
  segment_class?: string;
  features: Record<string, string>;
}

const PhonemePickerDialog: React.FC<PhonemePickerDialogProps> = ({
  open,
  onClose,
  onSelect,
  filter,
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const [tabIndex, setTabIndex] = React.useState(0);
  const [consonants, setConsonants] = React.useState<string[]>([]);
  const [vowels, setVowels] = React.useState<string[]>([]);
  const [loading, setLoading] = React.useState(true);

  // Load phonemes from exported data when dialog opens
  React.useEffect(() => {
    const loadPhonemes = async () => {
      setLoading(true);
      try {
        const data = await api.listPhonemes();
        const cons: string[] = [];
        const vows: string[] = [];

        data.phonemes.forEach((p: Phoneme) => {
          const phonemeType = p.type || p.segment_class;
          if (phonemeType === 'consonant') {
            // Apply sonorant/obstruent filter if specified (data-driven from Phoible features)
            if (filter === 'sonorants') {
              // Sonorants: consonantal:+ AND sonorant:+
              if (p.features.sonorant === '+') {
                cons.push(p.ipa);
              }
            } else if (filter === 'obstruents') {
              // Obstruents: consonantal:+ AND sonorant:-
              if (p.features.sonorant === '-') {
                cons.push(p.ipa);
              }
            } else {
              // No filter - include all consonants
              cons.push(p.ipa);
            }
          } else if (phonemeType === 'vowel' && !filter) {
            // Only include vowels when no consonant class filter is active
            vows.push(p.ipa);
          }
        });

        // Sort alphabetically
        setConsonants(cons.sort());
        setVowels(vows.sort());
      } catch (err) {
        console.error('Failed to load phonemes:', err);
      } finally {
        setLoading(false);
      }
    };

    if (open) {
      loadPhonemes();
    }
  }, [open, filter]);

  const handleSelect = (phoneme: string) => {
    onSelect(phoneme);
    // Don't close - allow multiple selections
  };

  return (
    <Dialog
      open={open}
      onClose={(_event, reason) => {
        // Only close on explicit button click, not backdrop or escape
        if (reason === 'backdropClick' || reason === 'escapeKeyDown') {
          return;
        }
        onClose();
      }}
      maxWidth="md"
      fullWidth
      fullScreen={isMobile}
      sx={{
        '& .MuiDialog-paper': {
          maxHeight: { xs: '90vh', sm: '80vh' },
        },
      }}
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 2 }}>
          <Box sx={{ flex: 1 }}>
            <Typography variant="h6" gutterBottom>IPA Keyboard</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              English (CMU) - {consonants.length + vowels.length} phonemes
            </Typography>

            {filter && (
              <Typography variant="caption" color="text.secondary" display="block">
                Showing {filter === 'sonorants' ? 'sonorants only' : 'obstruents only'}
              </Typography>
            )}
          </Box>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button variant="contained" onClick={onClose} size="small">
              Done
            </Button>
            <IconButton onClick={onClose} size="small" aria-label="close">
              <CloseIcon />
            </IconButton>
          </Box>
        </Box>
      </DialogTitle>
      <DialogContent sx={{ px: { xs: 2, sm: 3 } }}>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <CircularProgress />
          </Box>
        ) : (
          <>
            {!filter && (
              <Tabs value={tabIndex} onChange={(_, val) => setTabIndex(val)} sx={{ mb: 3 }}>
                <Tab label={`Consonants (${consonants.length})`} />
                <Tab label={`Vowels (${vowels.length})`} />
              </Tabs>
            )}

            {filter && (
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2, fontWeight: 500 }}>
                {filter === 'sonorants' ? 'Sonorants' : 'Obstruents'} ({consonants.length})
              </Typography>
            )}

            {(tabIndex === 0 || filter) && (
              <Box>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  {filter ? 'Tap to select' : 'Tap any consonant to select it'}
                </Typography>
                <Box
                  sx={{
                    display: 'grid',
                    gridTemplateColumns: {
                      xs: 'repeat(auto-fill, minmax(48px, 1fr))',
                      sm: 'repeat(auto-fill, minmax(56px, 1fr))',
                    },
                    gap: { xs: 1, sm: 1.5 },
                  }}
                >
                  {consonants.map((phoneme) => (
                    <Button
                      key={phoneme}
                      variant="outlined"
                      onClick={() => handleSelect(phoneme)}
                      sx={{
                        aspectRatio: '1',
                        minWidth: 0,
                        minHeight: { xs: 48, sm: 56 },
                        fontSize: { xs: '1.1rem', sm: '1.3rem' },
                        fontFamily: 'monospace',
                        p: 0,
                        '&:hover': {
                          bgcolor: 'primary.light',
                          color: 'white',
                        },
                        '&:active': {
                          transform: 'scale(0.95)',
                        },
                        transition: 'all 0.1s ease',
                      }}
                    >
                      {phoneme}
                    </Button>
                  ))}
                </Box>
              </Box>
            )}

            {tabIndex === 1 && !filter && (
              <Box>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Tap any vowel to select it
                </Typography>
                <Box
                  sx={{
                    display: 'grid',
                    gridTemplateColumns: {
                      xs: 'repeat(auto-fill, minmax(48px, 1fr))',
                      sm: 'repeat(auto-fill, minmax(56px, 1fr))',
                    },
                    gap: { xs: 1, sm: 1.5 },
                  }}
                >
                  {vowels.map((phoneme) => (
                    <Button
                      key={phoneme}
                      variant="outlined"
                      onClick={() => handleSelect(phoneme)}
                      sx={{
                        aspectRatio: '1',
                        minWidth: 0,
                        minHeight: { xs: 48, sm: 56 },
                        fontSize: { xs: '1.1rem', sm: '1.3rem' },
                        fontFamily: 'monospace',
                        p: 0,
                        '&:hover': {
                          bgcolor: 'secondary.light',
                          color: 'white',
                        },
                        '&:active': {
                          transform: 'scale(0.95)',
                        },
                        transition: 'all 0.1s ease',
                      }}
                    >
                      {phoneme}
                    </Button>
                  ))}
                </Box>
              </Box>
            )}
          </>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default PhonemePickerDialog;
