/**
 * Rhyme Sets Tool Component
 *
 * Generate rhyming word families for phonological awareness
 */

import React, { useState } from 'react';
import {
  Box,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Stack,
  Alert,
  CircularProgress,
  Typography,
  FormControlLabel,
  Checkbox,
} from '@mui/material';
import {
  PlayArrow as RunIcon,
} from '@mui/icons-material';
import api from '../../services/phonolexApi';
import { RhymeResult } from '../../types/phonology';

type RhymeMode = 'last_1' | 'last_2' | 'last_3' | 'assonance' | 'consonance';

const RhymeSetsTool: React.FC = () => {
  const [targetWord, setTargetWord] = useState('cat');
  const [rhymeMode, setRhymeMode] = useState<RhymeMode>('last_1');
  const [useEmbeddings, setUseEmbeddings] = useState(false);
  const [results, setResults] = useState<RhymeResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.generateRhymeSet({
        target_word: targetWord,
        rhyme_mode: rhymeMode,
        use_embeddings: useEmbeddings,
        limit: 100,
      });
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Stack spacing={{ xs: 1.5, sm: 2 }}>
        <TextField
          label="Target Word"
          value={targetWord}
          onChange={(e) => setTargetWord(e.target.value)}
          size="small"
          placeholder="e.g., cat"
          fullWidth
          inputProps={{
            sx: { fontSize: { xs: '0.9375rem', sm: '1rem' } },
          }}
        />
        <FormControl size="small" fullWidth>
          <InputLabel sx={{ fontSize: { xs: '0.9375rem', sm: '1rem' } }}>Rhyme Type</InputLabel>
          <Select
            value={rhymeMode}
            label="Rhyme Type"
            onChange={(e) => setRhymeMode(e.target.value as RhymeMode)}
            sx={{ '& .MuiSelect-select': { fontSize: { xs: '0.9375rem', sm: '1rem' } } }}
          >
            <MenuItem value="last_1">
              <Box>
                <Typography variant="body2" sx={{ fontSize: { xs: '0.875rem', sm: '0.9375rem' } }}>
                  Last Syllable
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ fontSize: { xs: '0.75rem', sm: '0.8125rem' } }}>
                  Matches nucleus + coda of final syllable (e.g., cat-bat, hat-mat)
                </Typography>
              </Box>
            </MenuItem>
            <MenuItem value="last_2">
              <Box>
                <Typography variant="body2" sx={{ fontSize: { xs: '0.875rem', sm: '0.9375rem' } }}>
                  Last 2 Syllables
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ fontSize: { xs: '0.75rem', sm: '0.8125rem' } }}>
                  Matches final 2 syllables (e.g., delicious-suspicious, computer-commuter)
                </Typography>
              </Box>
            </MenuItem>
            <MenuItem value="last_3">
              <Box>
                <Typography variant="body2" sx={{ fontSize: { xs: '0.875rem', sm: '0.9375rem' } }}>
                  Last 3 Syllables
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ fontSize: { xs: '0.75rem', sm: '0.8125rem' } }}>
                  Matches final 3 syllables (longer, more complex matches)
                </Typography>
              </Box>
            </MenuItem>
            <MenuItem value="assonance">
              <Box>
                <Typography variant="body2" sx={{ fontSize: { xs: '0.875rem', sm: '0.9375rem' } }}>
                  Assonance
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ fontSize: { xs: '0.75rem', sm: '0.8125rem' } }}>
                  Matches vowel sounds only (e.g., cat-back, lake-fade)
                </Typography>
              </Box>
            </MenuItem>
            <MenuItem value="consonance">
              <Box>
                <Typography variant="body2" sx={{ fontSize: { xs: '0.875rem', sm: '0.9375rem' } }}>
                  Consonance
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ fontSize: { xs: '0.75rem', sm: '0.8125rem' } }}>
                  Matches final consonants only (e.g., milk-walk, short-dirt)
                </Typography>
              </Box>
            </MenuItem>
          </Select>
        </FormControl>

        <FormControlLabel
          control={
            <Checkbox
              checked={useEmbeddings}
              onChange={(e) => setUseEmbeddings(e.target.checked)}
              sx={{ '& .MuiSvgIcon-root': { fontSize: { xs: 20, sm: 24 } } }}
            />
          }
          label={
            <Box>
              <Typography variant="body2" sx={{ fontSize: { xs: '0.875rem', sm: '0.9375rem' } }}>
                Include near-matches (quality &lt; 1.0)
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: { xs: '0.75rem', sm: '0.8125rem' } }}>
                Checked: approximate matches. Unchecked: perfect matches only (quality = 1.0)
              </Typography>
            </Box>
          }
        />

        <Button
          variant="contained"
          size="large"
          startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <RunIcon />}
          onClick={handleGenerate}
          disabled={loading}
          fullWidth
          sx={{ minHeight: 48 }}
        >
          {loading ? 'Generating...' : 'Generate Rhyme Set'}
        </Button>
      </Stack>

      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}

      {results && results.length > 0 && (
        <Box sx={{ mt: 3 }}>
          <Alert severity="success">
            Found {results.length} rhymes for "{targetWord}"
          </Alert>
          <Box sx={{ mt: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
            {results.map((r, i) => (
              <Box key={i} sx={{ display: 'inline-block', mr: 1, mb: 1 }}>
                <Box component="span" sx={{
                  px: 1.5,
                  py: 0.5,
                  bgcolor: 'white',
                  border: 1,
                  borderColor: 'divider',
                  borderRadius: 1,
                  display: 'inline-block',
                  fontSize: '0.95rem'
                }}>
                  {r.word?.word || ''}
                </Box>
              </Box>
            ))}
          </Box>
        </Box>
      )}

      {results && results.length === 0 && (
        <Alert severity="info" sx={{ mt: 2 }}>
          No rhymes found for "{targetWord}".
        </Alert>
      )}
    </Box>
  );
};

export default RhymeSetsTool;
