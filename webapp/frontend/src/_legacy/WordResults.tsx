/**
 * WordResults Component
 *
 * Displays filtered word results with phonological information
 */

import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
  Alert,
  Chip,
  Button,
  Stack,
} from '@mui/material';
import {
  Download as DownloadIcon,
  ContentCopy as CopyIcon,
  Shuffle as ShuffleIcon,
} from '@mui/icons-material';
import { useFilterStore } from '@stores/useFilterStore';
import type { WordResult } from '@types/phonology';

const WordResults: React.FC = () => {
  const { results, resultCount, isLoading, error, maxResults } = useFilterStore();

  const handleDownloadCSV = () => {
    if (results.length === 0) return;

    // Build CSV
    const headers = ['Word', 'IPA', 'Phonemes', 'Syllables', 'Frequency'];
    const rows = results.map((word) => [
      word.word,
      word.ipa,
      word.phonemes.join(' '),
      word.syllables.toString(),
      word.frequency.toFixed(2),
    ]);

    const csv = [headers, ...rows].map((row) => row.join(',')).join('\n');

    // Download
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `phonolex-wordlist-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleCopyList = () => {
    const wordList = results.map((w) => w.word).join('\n');
    navigator.clipboard.writeText(wordList);
  };

  const handleShuffle = () => {
    // TODO: Implement shuffle in store
    console.log('Shuffle not yet implemented');
  };

  return (
    <Card>
      <CardContent>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box>
            <Typography variant="h5">Results</Typography>
            {resultCount > 0 && (
              <Typography variant="body2" color="text.secondary">
                {resultCount.toLocaleString()} words matching
                {resultCount > maxResults && ` (showing first ${maxResults})`}
              </Typography>
            )}
          </Box>

          {results.length > 0 && (
            <Stack direction="row" spacing={1}>
              <Button
                size="small"
                startIcon={<CopyIcon />}
                onClick={handleCopyList}
              >
                Copy
              </Button>
              <Button
                size="small"
                startIcon={<ShuffleIcon />}
                onClick={handleShuffle}
                disabled
              >
                Shuffle
              </Button>
              <Button
                variant="contained"
                size="small"
                startIcon={<DownloadIcon />}
                onClick={handleDownloadCSV}
              >
                Download CSV
              </Button>
            </Stack>
          )}
        </Box>

        {/* Loading State */}
        {isLoading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
            <CircularProgress />
          </Box>
        )}

        {/* Error State */}
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {/* Empty State */}
        {!isLoading && !error && results.length === 0 && (
          <Paper
            sx={{
              p: 4,
              textAlign: 'center',
              bgcolor: 'background.default',
            }}
          >
            <Typography variant="body1" color="text.secondary">
              No results yet. Build a pattern to see matching words.
            </Typography>
          </Paper>
        )}

        {/* Results Table */}
        {!isLoading && results.length > 0 && (
          <TableContainer component={Paper} sx={{ maxHeight: 600 }}>
            <Table stickyHeader size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Word</TableCell>
                  <TableCell>IPA</TableCell>
                  <TableCell>Phonemes</TableCell>
                  <TableCell align="center">Syllables</TableCell>
                  <TableCell align="center">Frequency</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {results.map((word, index) => (
                  <WordRow key={`${word.word}-${index}`} word={word} />
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </CardContent>
    </Card>
  );
};

interface WordRowProps {
  word: WordResult;
}

const WordRow: React.FC<WordRowProps> = ({ word }) => {
  // Frequency badge color based on value (high = common word)
  const getFrequencyColor = (freq: number): 'success' | 'default' | 'error' => {
    if (freq >= 4.0) return 'success';  // Very common
    if (freq >= 2.0) return 'default';  // Moderate
    return 'error';  // Rare/no data
  };

  return (
    <TableRow hover>
      <TableCell>
        <Typography variant="body2" fontWeight={600}>
          {word.word}
        </Typography>
      </TableCell>
      <TableCell>
        <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
          /{word.ipa}/
        </Typography>
      </TableCell>
      <TableCell>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
          {word.phonemes.map((phoneme, idx) => (
            <Chip
              key={`${phoneme}-${idx}`}
              label={phoneme}
              size="small"
              variant="outlined"
              sx={{ fontSize: '0.75rem' }}
            />
          ))}
        </Box>
      </TableCell>
      <TableCell align="center">
        <Chip label={word.syllables} size="small" color="primary" />
      </TableCell>
      <TableCell align="center">
        <Chip
          label={word.frequency > 0 ? word.frequency.toFixed(2) : 'N/A'}
          size="small"
          color={getFrequencyColor(word.frequency)}
          variant="outlined"
        />
      </TableCell>
    </TableRow>
  );
};

export default WordResults;
