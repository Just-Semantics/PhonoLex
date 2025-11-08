/**
 * Text Analysis Tool
 *
 * Analyzes passages for phonological complexity, lexical properties, and psycholinguistic characteristics.
 * Provides aggregate percentile statistics and interactive text highlighting by feature.
 */

import React, { useState } from 'react';
import {
  Box,
  TextField,
  Button,
  Stack,
  Alert,
  CircularProgress,
  Typography,
  Paper,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Divider,
  SelectChangeEvent,
} from '@mui/material';
import {
  PlayArrow as AnalyzeIcon,
  Highlight as HighlightIcon,
} from '@mui/icons-material';
import api from '../../services/phonolexApi';

interface TextAnalysisResult {
  total_words: number;
  analyzed_words: number;
  unknown_words: string[];
  coverage_percent: number;
  aggregate_percentiles: {
    // Phonological
    syllable_count: number | null;
    phoneme_count: number | null;
    wcm_score: number | null;
    // Phonotactic
    phono_prob_avg: number | null;
    phono_prob_sum_log: number | null;
    positional_prob_avg: number | null;
    // Lexical
    frequency: number | null;
    aoa: number | null;
    // Semantic
    imageability: number | null;
    familiarity: number | null;
    concreteness: number | null;
    // Affective
    valence: number | null;
    arousal: number | null;
    dominance: number | null;
  };
  word_details: Array<{
    word: string;
    percentiles: Record<string, number | null>;
  }>;
}

interface FeatureConfig {
  id: string;
  label: string;
  category: string;
  interpretation: string;
  colorScheme: 'difficulty' | 'frequency' | 'diverging' | 'semantic' | 'intensity';
}

const FEATURES: FeatureConfig[] = [
  // Phonological Complexity
  { id: 'syllable_count', label: 'Syllables', category: 'Phonological Complexity', interpretation: 'Lower = simpler', colorScheme: 'difficulty' },
  { id: 'phoneme_count', label: 'Phonemes', category: 'Phonological Complexity', interpretation: 'Lower = simpler', colorScheme: 'difficulty' },
  { id: 'wcm_score', label: 'Word Complexity (WCM)', category: 'Phonological Complexity', interpretation: 'Lower = simpler', colorScheme: 'difficulty' },
  // Phonotactic Probability
  { id: 'phono_prob_avg', label: 'Phonotactic Probability (Avg)', category: 'Phonotactic Probability', interpretation: 'Higher = more typical', colorScheme: 'semantic' },
  { id: 'phono_prob_sum_log', label: 'Phonotactic Probability (Sum Log)', category: 'Phonotactic Probability', interpretation: 'Higher = more typical', colorScheme: 'semantic' },
  { id: 'positional_prob_avg', label: 'Positional Probability (Avg)', category: 'Phonotactic Probability', interpretation: 'Higher = more typical', colorScheme: 'semantic' },
  // Lexical Properties
  { id: 'frequency', label: 'Frequency', category: 'Lexical Properties', interpretation: 'Higher = more common', colorScheme: 'frequency' },
  { id: 'aoa', label: 'Age of Acquisition', category: 'Lexical Properties', interpretation: 'Lower = learned earlier', colorScheme: 'difficulty' },
  // Semantic Properties
  { id: 'imageability', label: 'Imageability', category: 'Semantic Properties', interpretation: 'Higher = more concrete', colorScheme: 'semantic' },
  { id: 'familiarity', label: 'Familiarity', category: 'Semantic Properties', interpretation: 'Higher = more familiar', colorScheme: 'semantic' },
  { id: 'concreteness', label: 'Concreteness', category: 'Semantic Properties', interpretation: 'Higher = more concrete', colorScheme: 'semantic' },
  // Affective Properties
  { id: 'valence', label: 'Valence', category: 'Affective Properties', interpretation: 'Higher = more positive', colorScheme: 'diverging' },
  { id: 'arousal', label: 'Arousal', category: 'Affective Properties', interpretation: 'Higher = more exciting', colorScheme: 'intensity' },
  { id: 'dominance', label: 'Dominance', category: 'Affective Properties', interpretation: 'Higher = more powerful', colorScheme: 'intensity' },
];

interface PassagePreset {
  id: string;
  name: string;
  text: string;
  description: string;
}

const PASSAGE_PRESETS: PassagePreset[] = [
  {
    id: 'grandfather',
    name: 'Grandfather Passage',
    text: 'You wished to know all about my grandfather. Well, he is nearly ninety-three years old. He dresses himself in an ancient black frock coat, usually minus several buttons; yet he still thinks as swiftly as ever. A long, flowing beard clings to his chin, giving those who observe him a pronounced feeling of the utmost respect. When he speaks his voice is just a bit cracked and quivers a trifle. Twice each day he plays skillfully and with zest upon our small organ. Except in the winter when the ooze or snow or ice prevents, he slowly takes a short walk in the open air each day. We have often urged him to walk more and smoke less, but he always answers, "Banana Oil!" Grandfather likes to be modern in his language.',
    description: 'Standard phonetics passage',
  },
  {
    id: 'rainbow',
    name: 'Rainbow Passage',
    text: 'When the sunlight strikes raindrops in the air, they act like a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end. People look but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow.',
    description: 'Classic phonetics passage',
  },
  {
    id: 'caterpillar',
    name: 'Caterpillar Passage',
    text: 'Do you like amusement parks? Well, I sure do. To amuse myself, I went twice last spring. My most MEMORABLE moment was riding on the Caterpillar, which is a gigantic roller coaster high above the ground. When I saw how high the Caterpillar rose into the bright blue sky I knew it was for me. After waiting in line for thirty minutes, I made it to the front where the man measured my height to see if I was tall enough. I gave the man my coins, asked for change, and jumped on the cart. Tick, tick, tick, the Caterpillar climbed slowly up the tracks. It went SO high I could see the parking lot. Boy was I SCARED! I thought to myself, "There\'s no turning back now." People were so scared they screamed as we swiftly zoomed fast, fast, and faster along the tracks. As quickly as it started, the Caterpillar came to a stop. Unfortunately, it was time to pack the car and drive home. That night I dreamt of the wild ride on the Caterpillar. Taking a trip to the amusement park and riding on the Caterpillar was my MOST memorable moment ever!',
    description: 'Pediatric speech sample',
  },
];

const TextAnalysisTool: React.FC = () => {
  const [text, setText] = useState('The quick brown fox jumps over the lazy dog.');
  const [results, setResults] = useState<TextAnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedFeature, setSelectedFeature] = useState<string>('');

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    setSelectedFeature(''); // Reset highlighting
    try {
      const data = await api.analyzeText(text);
      setResults(data);
    } catch (err) {
      console.error('Error in handleAnalyze:', err);
      setError(err instanceof Error ? err.message : 'An error occurred');
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  const handleFeatureChange = (event: SelectChangeEvent) => {
    setSelectedFeature(event.target.value);
  };

  const loadPreset = (preset: PassagePreset) => {
    setText(preset.text);
    setResults(null); // Clear previous results
    setSelectedFeature(''); // Reset highlighting
  };

  const getColorForPercentile = (percentile: number | null, colorScheme: string): string => {
    if (percentile === null) return 'transparent';

    // Normalize to 0-1
    const value = percentile / 100;

    switch (colorScheme) {
      case 'difficulty': // Low = easy (light), High = hard (red/orange)
        // Min opacity 0.15, max 0.7
        return `rgba(255, ${Math.round(165 - value * 165)}, 0, ${0.15 + value * 0.55})`;

      case 'frequency': // Low = rare (red), High = common (green)
        if (value < 0.5) {
          // 0-50th: red gradient (min opacity 0.2)
          return `rgba(255, ${Math.round(value * 255 * 2)}, 0, ${0.2 + (0.5 - value) * 1.0})`;
        } else {
          // 50-100th: green fade (min opacity 0.15)
          return `rgba(0, 200, 0, ${0.15 + (1 - value) * 0.35})`;
        }

      case 'semantic': // Low = light, High = solid blue
        // Min opacity 0.15, max 0.6
        return `rgba(30, 136, 229, ${0.15 + value * 0.45})`;

      case 'diverging': // Low = red, Mid = gray, High = green
        if (value < 0.5) {
          const intensity = (0.5 - value) * 2; // 0-0.5 → 1-0
          // Min opacity 0.2, max 0.6
          return `rgba(244, 67, 54, ${0.2 + intensity * 0.4})`; // Red
        } else {
          const intensity = (value - 0.5) * 2; // 0.5-1 → 0-1
          // Min opacity 0.2, max 0.6
          return `rgba(76, 175, 80, ${0.2 + intensity * 0.4})`; // Green
        }

      case 'intensity': // Low = light purple, High = intense purple
        // Min opacity 0.15, max 0.7
        return `rgba(156, 39, 176, ${0.15 + value * 0.55})`;

      default:
        return 'transparent';
    }
  };

  const renderHighlightedText = () => {
    if (!results || !selectedFeature) return null;

    const featureConfig = FEATURES.find(f => f.id === selectedFeature);
    if (!featureConfig) return null;

    // Split text into words (preserve punctuation)
    const tokens = text.split(/\b/);

    return (
      <Paper variant="outlined" sx={{ p: 2, mt: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          Highlighted by: {featureConfig.label}
        </Typography>
        <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
          {featureConfig.interpretation}
        </Typography>
        <Divider sx={{ my: 1 }} />
        <Box sx={{ lineHeight: 2, fontSize: '1.1rem' }}>
          {tokens.map((token, idx) => {
            const normalizedWord = token.toLowerCase().replace(/[^\w]/g, '');
            if (!normalizedWord) {
              // Non-word token (whitespace, punctuation)
              return <span key={idx}>{token}</span>;
            }

            // Find word details
            const wordDetail = results.word_details.find(
              w => w.word.toLowerCase() === normalizedWord
            );

            if (!wordDetail) {
              // Unknown word - subtle dotted underline (madlibs/fill-in-the-blank style)
              return (
                <span
                  key={idx}
                  style={{
                    borderBottom: '2px dotted #9e9e9e',
                    cursor: 'help',
                    opacity: 0.6,
                  }}
                  title="Unknown word (not in vocabulary)"
                >
                  {token}
                </span>
              );
            }

            const percentile = wordDetail.percentiles[`${selectedFeature}_percentile`];

            // If this property is N/A for this word, mark it like unknown
            if (percentile === null) {
              return (
                <span
                  key={idx}
                  style={{
                    borderBottom: '2px dotted #9e9e9e',
                    cursor: 'help',
                    opacity: 0.6,
                  }}
                  title={`${token}: No data for this property`}
                >
                  {token}
                </span>
              );
            }

            const bgColor = getColorForPercentile(percentile, featureConfig.colorScheme);

            return (
              <span
                key={idx}
                style={{
                  backgroundColor: bgColor,
                  padding: '2px 1px',
                  borderRadius: '2px',
                }}
                title={`${token}: ${percentile.toFixed(1)}th percentile`}
              >
                {token}
              </span>
            );
          })}
        </Box>
      </Paper>
    );
  };

  const renderAggregateStats = () => {
    if (!results) return null;

    const categories = [
      { name: 'Phonological Complexity', features: FEATURES.slice(0, 3) },
      { name: 'Phonotactic Probability', features: FEATURES.slice(3, 6) },
      { name: 'Lexical Properties', features: FEATURES.slice(6, 8) },
      { name: 'Semantic Properties', features: FEATURES.slice(8, 11) },
      { name: 'Affective Properties', features: FEATURES.slice(11, 14) },
    ];

    return (
      <Paper variant="outlined" sx={{ p: 2, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          Text Analysis Results
        </Typography>

        {/* Coverage */}
        <Box sx={{ mb: 3 }}>
          <Typography variant="body2" color="text.secondary">
            Coverage: {results.analyzed_words} / {results.total_words} words ({results.coverage_percent.toFixed(1)}%)
          </Typography>
          {results.unknown_words.length > 0 && (
            <Box sx={{ mt: 1 }}>
              <Typography variant="caption" color="text.secondary">
                Unknown words ({results.unknown_words.length}):
              </Typography>
              <Box sx={{ mt: 0.5 }}>
                {results.unknown_words.slice(0, 10).map((word, idx) => (
                  <Chip key={idx} label={word} size="small" sx={{ mr: 0.5, mb: 0.5 }} />
                ))}
                {results.unknown_words.length > 10 && (
                  <Typography variant="caption" color="text.secondary">
                    ... and {results.unknown_words.length - 10} more
                  </Typography>
                )}
              </Box>
            </Box>
          )}
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* Percentile Stats by Category */}
        {categories.map((category, catIdx) => (
          <Box key={catIdx} sx={{ mb: 3 }}>
            <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 'bold' }}>
              {category.name}
            </Typography>
            <Grid container spacing={2}>
              {category.features.map((feature) => {
                const value = results.aggregate_percentiles[feature.id as keyof typeof results.aggregate_percentiles];
                return (
                  <Grid item xs={12} sm={6} md={4} key={feature.id}>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        {feature.label}
                      </Typography>
                      <Typography variant="h6">
                        {value !== null ? `${value.toFixed(1)}th` : 'N/A'}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {feature.interpretation}
                      </Typography>
                    </Box>
                  </Grid>
                );
              })}
            </Grid>
          </Box>
        ))}
      </Paper>
    );
  };

  return (
    <Box>
      <Stack spacing={3}>
        {/* Text Input */}
        <TextField
          label="Text to Analyze"
          value={text}
          onChange={(e) => setText(e.target.value)}
          multiline
          rows={6}
          placeholder="Paste a passage, paragraph, or therapy script here..."
          fullWidth
        />

        {/* Preset Passages */}
        <Box>
          <Typography variant="subtitle2" gutterBottom sx={{ mb: 1 }}>
            Sample Passages
          </Typography>
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
            {PASSAGE_PRESETS.map((preset) => (
              <Chip
                key={preset.id}
                label={preset.name}
                onClick={() => loadPreset(preset)}
                color="default"
                sx={{ mb: { xs: 1.5, sm: 1 } }}
                title={preset.description}
              />
            ))}
          </Stack>
        </Box>

        {/* Analyze Button */}
        <Button
          variant="contained"
          size="large"
          startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <AnalyzeIcon />}
          onClick={handleAnalyze}
          disabled={loading || !text.trim()}
          fullWidth
          sx={{ minHeight: 48 }}
        >
          {loading ? 'Analyzing...' : 'Analyze Text'}
        </Button>
      </Stack>

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}

      {/* Results */}
      {results && (
        <>
          {renderAggregateStats()}

          {/* Feature Highlighting */}
          <Paper variant="outlined" sx={{ p: 2, mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Interactive Feature Highlighting
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Select a feature below to visualize it in the text with color-coded highlighting.
            </Typography>

            <FormControl fullWidth sx={{ mt: 2 }}>
              <InputLabel>Highlight Feature</InputLabel>
              <Select
                value={selectedFeature}
                label="Highlight Feature"
                onChange={handleFeatureChange}
                startAdornment={selectedFeature && <HighlightIcon sx={{ mr: 1, color: 'action.active' }} />}
              >
                <MenuItem value="">
                  <em>None (show plain text)</em>
                </MenuItem>
                {FEATURES.map((feature) => (
                  <MenuItem key={feature.id} value={feature.id}>
                    {feature.label} — {feature.interpretation}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {renderHighlightedText()}
          </Paper>
        </>
      )}
    </Box>
  );
};

export default TextAnalysisTool;
