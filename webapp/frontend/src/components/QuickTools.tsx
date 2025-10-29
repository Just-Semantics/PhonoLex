/**
 * Quick Tools Component
 *
 * Premade solutions for common clinical tasks:
 * - Minimal Pairs
 * - Maximal Oppositions
 * - Rhyme Sets
 * - Complexity Lists
 * - Phoneme Position
 */

import React, { useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  CardActions,
  Typography,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Stack,
  Alert,
  CircularProgress,
} from '@mui/material';
import {
  PlayArrow as RunIcon,
} from '@mui/icons-material';
import api from '../services/phonolexApi';
import type { Word, MinimalPair } from '../services/phonolexApi';
import WordResultsDisplay from './WordResultsDisplay';

const QuickTools: React.FC = () => {
  // State for minimal pairs
  const [minimalPairs, setMinimalPairs] = useState<{
    phoneme1: string;
    phoneme2: string;
    wordLength: 'short' | 'medium' | 'long';
    complexity: 'low' | 'medium' | 'high';
  }>({
    phoneme1: 't',
    phoneme2: 'd',
    wordLength: 'short',
    complexity: 'low',
  });

  // State for rhyme sets
  const [rhymeWord, setRhymeWord] = useState('cat');
  const [perfectRhymes, setPerfectRhymes] = useState(true);

  // State for complexity lists
  const [complexityRange, setComplexityRange] = useState<{
    minWcm: number;
    maxWcm: number;
    minMsh: number;
    maxMsh: number;
  }>({
    minWcm: 0,
    maxWcm: 5,
    minMsh: 1,
    maxMsh: 3,
  });

  // State for phoneme position
  const [phonemePos, setPhonemePos] = useState<{
    phoneme: string;
    position: 'initial' | 'medial' | 'final' | 'any';
  }>({
    phoneme: 'r',
    position: 'initial',
  });

  // Results state
  const [results, setResults] = useState<Word[] | MinimalPair[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeToolMessage, setActiveToolMessage] = useState<string | null>(null);

  // Generic run handler
  const runTool = async (toolName: string, apiCall: () => Promise<any>) => {
    setLoading(true);
    setError(null);
    setActiveToolMessage(`Running ${toolName}...`);
    try {
      const data = await apiCall();
      setResults(data);
      setActiveToolMessage(`${toolName} complete: ${data.length} results`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Grid container spacing={3}>
        {/* Minimal Pairs Tool */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Minimal Pairs
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Words differing by single phoneme for discrimination therapy
              </Typography>

              <Stack spacing={2} sx={{ mt: 2 }}>
                <TextField
                  label="Phoneme 1"
                  value={minimalPairs.phoneme1}
                  onChange={(e) =>
                    setMinimalPairs({ ...minimalPairs, phoneme1: e.target.value })
                  }
                  size="small"
                  placeholder="e.g., t"
                />
                <TextField
                  label="Phoneme 2"
                  value={minimalPairs.phoneme2}
                  onChange={(e) =>
                    setMinimalPairs({ ...minimalPairs, phoneme2: e.target.value })
                  }
                  size="small"
                  placeholder="e.g., d"
                />
                <FormControl size="small">
                  <InputLabel>Word Length</InputLabel>
                  <Select
                    value={minimalPairs.wordLength}
                    label="Word Length"
                    onChange={(e) =>
                      setMinimalPairs({
                        ...minimalPairs,
                        wordLength: e.target.value as any,
                      })
                    }
                  >
                    <MenuItem value="short">Short (1-2 syllables)</MenuItem>
                    <MenuItem value="medium">Medium (3-4 syllables)</MenuItem>
                    <MenuItem value="long">Long (5+ syllables)</MenuItem>
                  </Select>
                </FormControl>
                <FormControl size="small">
                  <InputLabel>Complexity</InputLabel>
                  <Select
                    value={minimalPairs.complexity}
                    label="Complexity"
                    onChange={(e) =>
                      setMinimalPairs({
                        ...minimalPairs,
                        complexity: e.target.value as any,
                      })
                    }
                  >
                    <MenuItem value="low">Low (WCM 0-5)</MenuItem>
                    <MenuItem value="medium">Medium (WCM 5-10)</MenuItem>
                    <MenuItem value="high">High (WCM 10+)</MenuItem>
                  </Select>
                </FormControl>
              </Stack>
            </CardContent>
            <CardActions>
              <Button
                variant="contained"
                startIcon={<RunIcon />}
                onClick={() =>
                  runTool('Minimal Pairs', () =>
                    api.generateMinimalPairs({
                      ...minimalPairs,
                      word_length: minimalPairs.wordLength,
                      limit: 50,
                    })
                  )
                }
                disabled={loading}
              >
                Generate
              </Button>
            </CardActions>
          </Card>
        </Grid>

        {/* Rhyme Sets Tool */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Rhyme Sets
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Generate rhyming word families for phonological awareness
              </Typography>

              <Stack spacing={2} sx={{ mt: 2 }}>
                <TextField
                  label="Target Word"
                  value={rhymeWord}
                  onChange={(e) => setRhymeWord(e.target.value)}
                  size="small"
                  placeholder="e.g., cat"
                />
                <FormControl size="small">
                  <InputLabel>Rhyme Type</InputLabel>
                  <Select
                    value={perfectRhymes ? 'perfect' : 'near'}
                    label="Rhyme Type"
                    onChange={(e) => setPerfectRhymes(e.target.value === 'perfect')}
                  >
                    <MenuItem value="perfect">Perfect Rhymes Only</MenuItem>
                    <MenuItem value="near">Include Near Rhymes</MenuItem>
                  </Select>
                </FormControl>
              </Stack>
            </CardContent>
            <CardActions>
              <Button
                variant="contained"
                startIcon={<RunIcon />}
                onClick={() =>
                  runTool('Rhyme Sets', () =>
                    api.generateRhymeSet({
                      target_word: rhymeWord,
                      perfect_only: perfectRhymes,
                      limit: 50,
                    })
                  )
                }
                disabled={loading}
              >
                Generate
              </Button>
            </CardActions>
          </Card>
        </Grid>

        {/* Complexity-Based Lists */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Age-Appropriate Lists
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Words filtered by complexity metrics (WCM, MSH)
              </Typography>

              <Grid container spacing={2} sx={{ mt: 1 }}>
                <Grid item xs={6}>
                  <TextField
                    label="Min WCM"
                    type="number"
                    value={complexityRange.minWcm}
                    onChange={(e) =>
                      setComplexityRange({
                        ...complexityRange,
                        minWcm: Number(e.target.value),
                      })
                    }
                    size="small"
                    fullWidth
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Max WCM"
                    type="number"
                    value={complexityRange.maxWcm}
                    onChange={(e) =>
                      setComplexityRange({
                        ...complexityRange,
                        maxWcm: Number(e.target.value),
                      })
                    }
                    size="small"
                    fullWidth
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Min MSH Stage"
                    type="number"
                    value={complexityRange.minMsh}
                    onChange={(e) =>
                      setComplexityRange({
                        ...complexityRange,
                        minMsh: Number(e.target.value),
                      })
                    }
                    size="small"
                    fullWidth
                    inputProps={{ min: 1, max: 6 }}
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Max MSH Stage"
                    type="number"
                    value={complexityRange.maxMsh}
                    onChange={(e) =>
                      setComplexityRange({
                        ...complexityRange,
                        maxMsh: Number(e.target.value),
                      })
                    }
                    size="small"
                    fullWidth
                    inputProps={{ min: 1, max: 6 }}
                  />
                </Grid>
              </Grid>
            </CardContent>
            <CardActions>
              <Button
                variant="contained"
                startIcon={<RunIcon />}
                onClick={() =>
                  runTool('Complexity List', () =>
                    api.generateComplexityList({
                      min_wcm: complexityRange.minWcm,
                      max_wcm: complexityRange.maxWcm,
                      min_msh: complexityRange.minMsh,
                      max_msh: complexityRange.maxMsh,
                      limit: 100,
                    })
                  )
                }
                disabled={loading}
              >
                Generate
              </Button>
            </CardActions>
          </Card>
        </Grid>

        {/* Phoneme Position Tool */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Phoneme Position
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Words with specific phoneme in target position
              </Typography>

              <Stack spacing={2} sx={{ mt: 2 }}>
                <TextField
                  label="Phoneme"
                  value={phonemePos.phoneme}
                  onChange={(e) =>
                    setPhonemePos({ ...phonemePos, phoneme: e.target.value })
                  }
                  size="small"
                  placeholder="e.g., r"
                />
                <FormControl size="small">
                  <InputLabel>Position</InputLabel>
                  <Select
                    value={phonemePos.position}
                    label="Position"
                    onChange={(e) =>
                      setPhonemePos({
                        ...phonemePos,
                        position: e.target.value as any,
                      })
                    }
                  >
                    <MenuItem value="initial">Initial</MenuItem>
                    <MenuItem value="medial">Medial</MenuItem>
                    <MenuItem value="final">Final</MenuItem>
                    <MenuItem value="any">Any Position</MenuItem>
                  </Select>
                </FormControl>
              </Stack>
            </CardContent>
            <CardActions>
              <Button
                variant="contained"
                startIcon={<RunIcon />}
                onClick={() =>
                  runTool('Phoneme Position', () =>
                    api.findPhonemePosition({
                      phoneme: phonemePos.phoneme,
                      position: phonemePos.position,
                      limit: 100,
                    })
                  )
                }
                disabled={loading}
              >
                Find Words
              </Button>
            </CardActions>
          </Card>
        </Grid>
      </Grid>

      {/* Status Messages */}
      <Box sx={{ mt: 3 }}>
        {loading && (
          <Alert severity="info" icon={<CircularProgress size={20} />}>
            {activeToolMessage}
          </Alert>
        )}
        {error && <Alert severity="error">{error}</Alert>}
        {activeToolMessage && !loading && !error && (
          <Alert severity="success">{activeToolMessage}</Alert>
        )}
      </Box>

      {/* Results Display */}
      {results && !loading && (
        <Box sx={{ mt: 3 }}>
          <WordResultsDisplay results={results} />
        </Box>
      )}
    </Box>
  );
};

export default QuickTools;
