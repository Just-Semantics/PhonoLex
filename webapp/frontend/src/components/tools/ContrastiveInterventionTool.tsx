/**
 * Contrastive Intervention Tool Component
 *
 * Unified tool for three research-based phonological intervention approaches:
 * - Minimal Pairs: Conventional target/substitute contrast
 * - Maximal Opposition: Two unknowns with major class difference
 * - Multiple Opposition: Global phoneme collapse treatment
 *
 * References:
 * - Gierut, J. A. (1989-1992). Maximal opposition research
 * - Storkel, H. L. (2022). Minimal, Maximal, or Multiple...
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
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  ToggleButtonGroup,
  ToggleButton,
  Stepper,
  Step,
  StepLabel,
  useMediaQuery,
  useTheme,
} from '@mui/material';
import {
  PlayArrow as RunIcon,
  Clear as ClearIcon,
  Keyboard as KeyboardIcon,
  CompareArrows as MinimalIcon,
  SwapHoriz as MaximalIcon,
  AccountTree as MultipleIcon,
} from '@mui/icons-material';
import api from '../../services/phonolexApi';
import type { Word, MinimalPair } from '../../services/phonolexApi';
import PhonemePickerDialog from '../PhonemePickerDialog';
import ContrastiveGroupsTable from '../shared/ContrastiveGroupsTable';
import { validatePhonemeInput } from '../../utils/ipaValidation';

type InterventionMode = 'minimal' | 'maximal' | 'multiple';

interface MaximalOppositionPair {
  phoneme1: string;
  phoneme2: string;
  score: number;
  major_class_diff: boolean;
  feature_diffs: number;
}

interface WordPair {
  word1: Word;
  word2: Word;
  position: number;
}

const ContrastiveInterventionTool: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  // Mode selection
  const [mode, setMode] = useState<InterventionMode>('minimal');

  // Unified input state
  const [phoneme1, setPhoneme1] = useState<string>('');
  const [phoneme2, setPhoneme2] = useState<string>('');
  const [sonorants, setSonorants] = useState<string>('');
  const [obstruents, setObstruents] = useState<string>('');
  const [substitutePhoneme, setSubstitutePhoneme] = useState<string>('');
  const [targetPhonemes, setTargetPhonemes] = useState<string>('');
  const [position, setPosition] = useState<'initial' | 'medial' | 'final' | 'any'>('any');

  // Results state
  const [minimalResults, setMinimalResults] = useState<MinimalPair[] | null>(null);
  const [maximalPairs, setMaximalPairs] = useState<MaximalOppositionPair[] | null>(null);
  const [selectedPair, setSelectedPair] = useState<MaximalOppositionPair | null>(null);
  const [wordLists, setWordLists] = useState<WordPair[] | null>(null);
  const [representativeTargets, setRepresentativeTargets] = useState<string[] | null>(null);
  const [multipleSets, setMultipleSets] = useState<Array<{
    words: Array<{ word: Word; phoneme: string }>;
    position: number;
  }> | null>(null);

  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [phonemePickerOpen, setPhonemePickerOpen] = useState(false);
  const [activeField, setActiveField] = useState<'phoneme1' | 'phoneme2' | 'sonorants' | 'obstruents' | 'substitute' | 'targets'>('phoneme1');
  const [ipaWarning1, setIpaWarning1] = useState<string | null>(null);
  const [ipaWarning2, setIpaWarning2] = useState<string | null>(null);
  const [ipaWarningSonorants, setIpaWarningSonorants] = useState<string | null>(null);
  const [ipaWarningObstruents, setIpaWarningObstruents] = useState<string | null>(null);
  const [ipaWarningSubstitute, setIpaWarningSubstitute] = useState<string | null>(null);
  const [ipaWarningTargets, setIpaWarningTargets] = useState<string | null>(null);

  // Stepper state for maximal mode
  const [maximalStep, setMaximalStep] = useState(0);

  const handleModeChange = (newMode: InterventionMode | null) => {
    if (newMode !== null) {
      setMode(newMode);
      handleClear();
    }
  };

  const handleClear = () => {
    setPhoneme1('');
    setPhoneme2('');
    setSonorants('');
    setObstruents('');
    setSubstitutePhoneme('');
    setTargetPhonemes('');
    setPosition('any');
    setMinimalResults(null);
    setMaximalPairs(null);
    setSelectedPair(null);
    setWordLists(null);
    setRepresentativeTargets(null);
    setMultipleSets(null);
    setError(null);
    setIpaWarning1(null);
    setIpaWarning2(null);
    setIpaWarningSonorants(null);
    setIpaWarningObstruents(null);
    setIpaWarningSubstitute(null);
    setIpaWarningTargets(null);
    setMaximalStep(0);
  };

  const handleGenerateMinimal = async () => {
    setLoading(true);
    setError(null);
    try {
      let data = await api.getMinimalPairs({
        phoneme1: phoneme1.trim(),
        phoneme2: phoneme2.trim(),
        limit: 200,
      });

      // Filter by position if specified
      if (position !== 'any') {
        data = data.filter(pair => {
          const pos = pair.position ?? pair.metadata?.position;
          if (pos === undefined) return true;

          const wordLength = pair.word1.phoneme_count;

          if (position === 'initial') {
            return pos === 0;
          } else if (position === 'final') {
            return pos === wordLength - 1;
          } else if (position === 'medial') {
            return pos > 0 && pos < wordLength - 1;
          }
          return true;
        });
      }

      setMinimalResults(data.slice(0, 50));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setMinimalResults(null);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateMaximalPairs = async () => {
    setLoading(true);
    setError(null);
    setMaximalPairs(null);
    setSelectedPair(null);
    setWordLists(null);

    try {
      // Parse phonemes from both fields
      const sonorantList = sonorants
        .split(/[\s,]+/)
        .map(p => p.trim())
        .filter(p => p.length > 0);

      const obstruentList = obstruents
        .split(/[\s,]+/)
        .map(p => p.trim())
        .filter(p => p.length > 0);

      if (sonorantList.length === 0 && obstruentList.length === 0) {
        throw new Error('Please enter at least one phoneme in each field');
      }

      if (sonorantList.length === 0) {
        throw new Error('Please enter at least one sonorant (m, n, ŋ, l, r, w, j)');
      }

      if (obstruentList.length === 0) {
        throw new Error('Please enter at least one obstruent (p, t, k, f, s, ʃ, etc.)');
      }

      // Combine both lists for API call
      const phonemeList = [...sonorantList, ...obstruentList];

      const data = await api.generateMaximalOppositionPairs({
        unknown_phonemes: phonemeList,
        top_n: 10,
      });

      if (data.length === 0) {
        setError('No maximal opposition pairs found. Pairs must differ by major class (sonorant vs obstruent).');
      } else {
        setMaximalPairs(data);
        setMaximalStep(1); // Move to step 2
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateWordLists = async (pair: MaximalOppositionPair) => {
    setLoading(true);
    setError(null);
    setSelectedPair(pair);
    setWordLists(null);

    try {
      const data = await api.findMaximalOppositionWordLists({
        phoneme1: pair.phoneme1,
        phoneme2: pair.phoneme2,
        position,
        max_pairs: 10,
      });

      if (data.length === 0) {
        setError(`No word pairs found for /${pair.phoneme1}/-/${pair.phoneme2}/ in ${position} position. Try a different position.`);
      } else {
        setWordLists(data);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateMultiple = async () => {
    setLoading(true);
    setError(null);
    setRepresentativeTargets(null);
    setMultipleSets(null);

    try {
      // Parse target phonemes
      const targetList = targetPhonemes
        .split(/[\s,]+/)
        .map(p => p.trim())
        .filter(p => p.length > 0);

      if (targetList.length === 0) {
        throw new Error('Please enter at least one target phoneme');
      }

      if (targetList.length === 1) {
        throw new Error('For single substitutions (e.g., r→w), use Minimal Pairs mode instead. Multiple Opposition is for global phoneme collapses with 2+ targets.');
      }

      // Select representative targets using Maximal Classification + Maximal Distinction
      const selected = api.selectRepresentativeTargets({
        substitute_phoneme: substitutePhoneme.trim(),
        target_phonemes: targetList,
        count: Math.min(4, targetList.length), // Up to 4 targets for quintuplets
      });

      setRepresentativeTargets(selected);

      // Generate minimal sets (triplets/quadruplets/quintuplets)
      const sets = await api.generateMultipleOppositionSets({
        substitute_phoneme: substitutePhoneme.trim(),
        target_phonemes: selected,
        position,
        max_sets: 10,
      });

      if (sets.length === 0) {
        setError(`No minimal sets found for this collapse in ${position} position. Try a different position.`);
      } else {
        setMultipleSets(sets);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleGenerate = () => {
    if (mode === 'minimal') {
      handleGenerateMinimal();
    } else if (mode === 'maximal') {
      handleGenerateMaximalPairs();
    } else if (mode === 'multiple') {
      handleGenerateMultiple();
    }
  };

  const handlePhonemeSelect = (phoneme: string) => {
    if (activeField === 'phoneme1') {
      setPhoneme1(prev => prev + phoneme);
    } else if (activeField === 'phoneme2') {
      setPhoneme2(prev => prev + phoneme);
    } else if (activeField === 'sonorants') {
      setSonorants(prev => prev + (prev ? ' ' : '') + phoneme);
    } else if (activeField === 'obstruents') {
      setObstruents(prev => prev + (prev ? ' ' : '') + phoneme);
    } else if (activeField === 'substitute') {
      setSubstitutePhoneme(prev => prev + phoneme);
    } else if (activeField === 'targets') {
      setTargetPhonemes(prev => prev + (prev ? ' ' : '') + phoneme);
    }
  };

  const openPhonemePicker = (field: 'phoneme1' | 'phoneme2' | 'sonorants' | 'obstruents' | 'substitute' | 'targets') => {
    setActiveField(field);
    setPhonemePickerOpen(true);
  };

  const getModeHelperText = () => {
    switch (mode) {
      case 'minimal':
        return 'Enter the two phonemes that contrast (e.g., target /θ/ vs substitute /t/)';
      case 'maximal':
        return 'Enter unknown phonemes from each major class. Algorithm will pair sonorants with obstruents for maximal contrast.';
      case 'multiple':
        return 'Enter the substitute phoneme (what child says) and all target phonemes (what they should say). Algorithm will select representative targets and generate minimal sets.';
    }
  };

  const isGenerateDisabled = () => {
    if (loading) return true;
    if (mode === 'minimal') {
      return !phoneme1.trim() || !phoneme2.trim();
    } else if (mode === 'maximal') {
      return !sonorants.trim() || !obstruents.trim();
    } else if (mode === 'multiple') {
      return !substitutePhoneme.trim() || !targetPhonemes.trim();
    }
    return false;
  };

  return (
    <Box>
      <Stack spacing={3}>
        {/* Mode Selection - Desktop: Toggle, Mobile: Dropdown */}
        {isMobile ? (
          <FormControl fullWidth>
            <InputLabel>Intervention Mode</InputLabel>
            <Select
              value={mode}
              label="Intervention Mode"
              onChange={(e) => handleModeChange(e.target.value as InterventionMode)}
            >
              <MenuItem value="minimal">
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <MinimalIcon />
                  <Typography>Minimal Pairs</Typography>
                </Box>
              </MenuItem>
              <MenuItem value="maximal">
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <MaximalIcon />
                  <Typography>Maximal Opposition</Typography>
                </Box>
              </MenuItem>
              <MenuItem value="multiple">
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <MultipleIcon />
                  <Typography>Multiple Opposition</Typography>
                </Box>
              </MenuItem>
            </Select>
          </FormControl>
        ) : (
          <ToggleButtonGroup
            value={mode}
            exclusive
            onChange={(_, val) => handleModeChange(val)}
            fullWidth
            color="primary"
          >
            <ToggleButton value="minimal">
              <MinimalIcon sx={{ mr: 1, fontSize: '1.1rem' }} />
              <Typography variant="body2">Minimal Pairs</Typography>
            </ToggleButton>
            <ToggleButton value="maximal">
              <MaximalIcon sx={{ mr: 1, fontSize: '1.1rem' }} />
              <Typography variant="body2">Maximal Opposition</Typography>
            </ToggleButton>
            <ToggleButton value="multiple">
              <MultipleIcon sx={{ mr: 1, fontSize: '1.1rem' }} />
              <Typography variant="body2">Multiple Opposition</Typography>
            </ToggleButton>
          </ToggleButtonGroup>
        )}

        {/* Stepper for Maximal Mode */}
        {mode === 'maximal' && (
          <Stepper activeStep={maximalStep} alternativeLabel={isMobile} sx={{ pt: 2, pb: 1 }}>
            <Step>
              <StepLabel>Enter Phonemes & Position</StepLabel>
            </Step>
            <Step>
              <StepLabel>Select Pair</StepLabel>
            </Step>
            <Step>
              <StepLabel>Generate Word Lists</StepLabel>
            </Step>
          </Stepper>
        )}

        {/* Input Section */}
        <Paper sx={{ p: { xs: 2, sm: 3 } }}>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2, fontSize: { xs: '0.8125rem', sm: '0.875rem' } }}>
            {getModeHelperText()}
          </Typography>

          <Stack spacing={2}>
            {/* Minimal Mode: Two separate phoneme fields */}
            {mode === 'minimal' && (
              <>
                <Box>
                  <TextField
                    fullWidth
                    label="Phoneme 1 (Target - IPA)"
                    value={phoneme1}
                    onChange={(e) => {
                      const newValue = e.target.value;
                      setPhoneme1(newValue);
                      if (newValue.trim()) {
                        const validation = validatePhonemeInput(newValue);
                        if (!validation.isValid && validation.suggestion) {
                          setIpaWarning1(validation.suggestion);
                        } else {
                          setIpaWarning1(null);
                        }
                      } else {
                        setIpaWarning1(null);
                      }
                    }}
                    placeholder="e.g., θ"
                    size="small"
                    helperText="The correct production"
                    InputProps={{
                      endAdornment: (
                        <IconButton
                          onClick={() => openPhonemePicker('phoneme1')}
                          edge="end"
                          color="primary"
                          size="small"
                          sx={{ minWidth: { xs: 48, sm: 44 }, minHeight: { xs: 48, sm: 44 } }}
                        >
                          <KeyboardIcon />
                        </IconButton>
                      )
                    }}
                  />
                  {ipaWarning1 && (
                    <Alert severity="warning" sx={{ mt: 1 }}>
                      {ipaWarning1}
                    </Alert>
                  )}
                </Box>

                <Box>
                  <TextField
                    fullWidth
                    label="Phoneme 2 (Substitute - IPA)"
                    value={phoneme2}
                    onChange={(e) => {
                      const newValue = e.target.value;
                      setPhoneme2(newValue);
                      if (newValue.trim()) {
                        const validation = validatePhonemeInput(newValue);
                        if (!validation.isValid && validation.suggestion) {
                          setIpaWarning2(validation.suggestion);
                        } else {
                          setIpaWarning2(null);
                        }
                      } else {
                        setIpaWarning2(null);
                      }
                    }}
                    placeholder="e.g., t"
                    size="small"
                    helperText="What the child says instead"
                    InputProps={{
                      endAdornment: (
                        <IconButton
                          onClick={() => openPhonemePicker('phoneme2')}
                          edge="end"
                          color="primary"
                          size="small"
                          sx={{ minWidth: { xs: 48, sm: 44 }, minHeight: { xs: 48, sm: 44 } }}
                        >
                          <KeyboardIcon />
                        </IconButton>
                      )
                    }}
                  />
                  {ipaWarning2 && (
                    <Alert severity="warning" sx={{ mt: 1 }}>
                      {ipaWarning2}
                    </Alert>
                  )}
                </Box>

                <FormControl size="small" fullWidth>
                  <InputLabel>Position in Word</InputLabel>
                  <Select
                    value={position}
                    label="Position in Word"
                    onChange={(e) => setPosition(e.target.value as 'initial' | 'medial' | 'final' | 'any')}
                  >
                    <MenuItem value="any">Any Position</MenuItem>
                    <MenuItem value="initial">Word-Initial</MenuItem>
                    <MenuItem value="medial">Word-Medial</MenuItem>
                    <MenuItem value="final">Word-Final</MenuItem>
                  </Select>
                </FormControl>
              </>
            )}

            {/* Maximal Mode: Two separate fields for sonorants/obstruents */}
            {mode === 'maximal' && maximalStep === 0 && (
              <>
                <Box>
                  <TextField
                    fullWidth
                    label="Sonorants (m, n, ŋ, l, r, w, j)"
                    value={sonorants}
                    onChange={(e) => {
                      const newValue = e.target.value;
                      setSonorants(newValue);
                      if (newValue.trim()) {
                        const validation = validatePhonemeInput(newValue);
                        if (!validation.isValid && validation.suggestion) {
                          setIpaWarningSonorants(validation.suggestion);
                        } else {
                          setIpaWarningSonorants(null);
                        }
                      } else {
                        setIpaWarningSonorants(null);
                      }
                    }}
                    placeholder="e.g., l r ŋ"
                    size="small"
                    helperText="Nasals, liquids, glides"
                    InputProps={{
                      endAdornment: (
                        <IconButton
                          onClick={() => openPhonemePicker('sonorants')}
                          edge="end"
                          color="primary"
                          size="small"
                          sx={{ minWidth: { xs: 48, sm: 44 }, minHeight: { xs: 48, sm: 44 } }}
                        >
                          <KeyboardIcon />
                        </IconButton>
                      )
                    }}
                  />
                  {ipaWarningSonorants && (
                    <Alert severity="warning" sx={{ mt: 1 }}>
                      {ipaWarningSonorants}
                    </Alert>
                  )}
                </Box>

                <Box>
                  <TextField
                    fullWidth
                    label="Obstruents (p, t, k, b, d, g, f, s, ʃ, θ, etc.)"
                    value={obstruents}
                    onChange={(e) => {
                      const newValue = e.target.value;
                      setObstruents(newValue);
                      if (newValue.trim()) {
                        const validation = validatePhonemeInput(newValue);
                        if (!validation.isValid && validation.suggestion) {
                          setIpaWarningObstruents(validation.suggestion);
                        } else {
                          setIpaWarningObstruents(null);
                        }
                      } else {
                        setIpaWarningObstruents(null);
                      }
                    }}
                    placeholder="e.g., g θ ʃ"
                    size="small"
                    helperText="Stops, fricatives, affricates"
                    InputProps={{
                      endAdornment: (
                        <IconButton
                          onClick={() => openPhonemePicker('obstruents')}
                          edge="end"
                          color="primary"
                          size="small"
                          sx={{ minWidth: { xs: 48, sm: 44 }, minHeight: { xs: 48, sm: 44 } }}
                        >
                          <KeyboardIcon />
                        </IconButton>
                      )
                    }}
                  />
                  {ipaWarningObstruents && (
                    <Alert severity="warning" sx={{ mt: 1 }}>
                      {ipaWarningObstruents}
                    </Alert>
                  )}
                </Box>

                {/* Position selector for maximal mode */}
                <FormControl size="small" fullWidth>
                  <InputLabel>Position in Word</InputLabel>
                  <Select
                    value={position}
                    label="Position in Word"
                    onChange={(e) => setPosition(e.target.value as 'initial' | 'medial' | 'final' | 'any')}
                  >
                    <MenuItem value="initial">Word-Initial</MenuItem>
                    <MenuItem value="medial">Word-Medial</MenuItem>
                    <MenuItem value="final">Word-Final</MenuItem>
                    <MenuItem value="any">Any Position</MenuItem>
                  </Select>
                </FormControl>
              </>
            )}

            {/* Multiple Mode: Substitute + Targets */}
            {mode === 'multiple' && (
              <>
                <Box>
                  <TextField
                    fullWidth
                    label="Substitute Phoneme (IPA)"
                    value={substitutePhoneme}
                    onChange={(e) => {
                      const newValue = e.target.value;
                      setSubstitutePhoneme(newValue);
                      if (newValue.trim()) {
                        const validation = validatePhonemeInput(newValue);
                        if (!validation.isValid && validation.suggestion) {
                          setIpaWarningSubstitute(validation.suggestion);
                        } else {
                          setIpaWarningSubstitute(null);
                        }
                      } else {
                        setIpaWarningSubstitute(null);
                      }
                    }}
                    placeholder="e.g., t (for t→d,k collapse)"
                    size="small"
                    helperText="What the child SAYS (the error sound)"
                    InputProps={{
                      endAdornment: (
                        <IconButton
                          onClick={() => openPhonemePicker('substitute')}
                          edge="end"
                          color="primary"
                          size="small"
                          sx={{ minWidth: { xs: 48, sm: 44 }, minHeight: { xs: 48, sm: 44 } }}
                        >
                          <KeyboardIcon />
                        </IconButton>
                      )
                    }}
                  />
                  {ipaWarningSubstitute && (
                    <Alert severity="warning" sx={{ mt: 1 }}>
                      {ipaWarningSubstitute}
                    </Alert>
                  )}
                </Box>

                <Box>
                  <TextField
                    fullWidth
                    label="Target Phonemes (IPA)"
                    value={targetPhonemes}
                    onChange={(e) => {
                      const newValue = e.target.value;
                      setTargetPhonemes(newValue);
                      if (newValue.trim()) {
                        const validation = validatePhonemeInput(newValue);
                        if (!validation.isValid && validation.suggestion) {
                          setIpaWarningTargets(validation.suggestion);
                        } else {
                          setIpaWarningTargets(null);
                        }
                      } else {
                        setIpaWarningTargets(null);
                      }
                    }}
                    placeholder="e.g., d k (for t→d,k collapse)"
                    size="small"
                    helperText="What the child SHOULD say (space or comma separated)"
                    multiline
                    rows={2}
                    InputProps={{
                      endAdornment: (
                        <IconButton
                          onClick={() => openPhonemePicker('targets')}
                          edge="end"
                          color="primary"
                          size="small"
                          sx={{ minWidth: { xs: 48, sm: 44 }, minHeight: { xs: 48, sm: 44 } }}
                        >
                          <KeyboardIcon />
                        </IconButton>
                      )
                    }}
                  />
                  {ipaWarningTargets && (
                    <Alert severity="warning" sx={{ mt: 1 }}>
                      {ipaWarningTargets}
                    </Alert>
                  )}
                </Box>

                {/* Position selector for multiple mode */}
                <FormControl size="small" fullWidth>
                  <InputLabel>Position in Word</InputLabel>
                  <Select
                    value={position}
                    label="Position in Word"
                    onChange={(e) => setPosition(e.target.value as 'initial' | 'medial' | 'final' | 'any')}
                  >
                    <MenuItem value="initial">Word-Initial</MenuItem>
                    <MenuItem value="medial">Word-Medial</MenuItem>
                    <MenuItem value="final">Word-Final</MenuItem>
                    <MenuItem value="any">Any Position</MenuItem>
                  </Select>
                </FormControl>
              </>
            )}

            {/* Action Buttons */}
            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={{ xs: 1.5, sm: 2 }}>
              <Button
                variant="outlined"
                startIcon={<ClearIcon />}
                onClick={handleClear}
                disabled={loading}
                fullWidth
                size="large"
              >
                Clear
              </Button>
              <Button
                variant="contained"
                startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <RunIcon />}
                onClick={handleGenerate}
                disabled={isGenerateDisabled()}
                fullWidth
                size="large"
              >
                {loading ? 'Generating...' : 'Generate'}
              </Button>
            </Stack>
          </Stack>
        </Paper>

        {/* Error Display */}
        {error && (
          <Alert severity="error" onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* Results: Minimal Pairs */}
        {mode === 'minimal' && minimalResults && minimalResults.length > 0 && (
          <ContrastiveGroupsTable
            pairs={minimalResults}
            mode="minimal"
            enableSelection={true}
            exportFilename={`phonolex_minimal_pairs_${phoneme1}_${phoneme2}.csv`}
          />
        )}

        {mode === 'minimal' && minimalResults && minimalResults.length === 0 && (
          <Alert severity="info">
            No minimal pairs found for this phoneme contrast.
          </Alert>
        )}

        {/* Results: Maximal Opposition Pairs */}
        {mode === 'maximal' && maximalPairs && maximalPairs.length > 0 && (
          <Paper sx={{ p: { xs: 2, sm: 3 } }}>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2, fontSize: { xs: '0.8125rem', sm: '0.875rem' } }}>
              Select a pair to generate word lists for {position} position. Each pair differs by major class (sonorant vs obstruent) for maximal phonological contrast.
            </Typography>

            <Stack spacing={1}>
              {maximalPairs.map((pair, index) => (
                <Paper
                  key={index}
                  sx={{
                    p: { xs: 1.5, sm: 2 },
                    cursor: 'pointer',
                    border: selectedPair === pair ? '2px solid' : '1px solid',
                    borderColor: selectedPair === pair ? 'primary.main' : 'divider',
                    '&:hover': { bgcolor: 'action.hover' },
                  }}
                  onClick={() => handleGenerateWordLists(pair)}
                >
                  <Stack direction="row" spacing={2} alignItems="center" justifyContent="space-between">
                    <Box>
                      <Typography variant="h6" component="span" sx={{ fontSize: { xs: '1rem', sm: '1.25rem' } }}>
                        /{pair.phoneme1}/ - /{pair.phoneme2}/
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5, fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>
                        Score: {pair.score} ({pair.feature_diffs} features)
                      </Typography>
                    </Box>
                    <Chip
                      label={`${pair.feature_diffs} features`}
                      color="primary"
                      size="small"
                    />
                  </Stack>
                </Paper>
              ))}
            </Stack>
          </Paper>
        )}

        {/* Results: Word Lists for Selected Pair */}
        {mode === 'maximal' && selectedPair && (
          <Box>
            <Alert severity="info" sx={{ mb: 2 }}>
              Word pairs for /{selectedPair.phoneme1}/ - /{selectedPair.phoneme2}/ in {position} position
            </Alert>

            {loading && (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                <CircularProgress />
              </Box>
            )}

            {wordLists && wordLists.length > 0 && (
              <ContrastiveGroupsTable
                pairs={wordLists.map(pair => ({
                  word1: pair.word1,
                  word2: pair.word2,
                  position: pair.position,
                  phoneme1: selectedPair.phoneme1,
                  phoneme2: selectedPair.phoneme2,
                }))}
                mode="maximal"
                enableSelection={true}
                exportFilename={`phonolex_maximal_${selectedPair.phoneme1}_${selectedPair.phoneme2}.csv`}
              />
            )}

            {!loading && wordLists && wordLists.length === 0 && (
              <Alert severity="warning">
                No word pairs found for this phoneme combination in {position} position. Try selecting a different position.
              </Alert>
            )}
          </Box>
        )}

        {/* Results: Multiple Opposition Sets */}
        {mode === 'multiple' && representativeTargets && (
          <Box>
            <Alert severity="info" sx={{ mb: 2 }}>
              <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>
                Collapse: [{substitutePhoneme}] → /{representativeTargets.join(', ')}/
              </Typography>
              <Typography variant="caption">
                These {representativeTargets.length} targets represent the breadth of the phonological collapse and maximize phonological distance from the substitute.
              </Typography>
            </Alert>

            {loading && (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                <CircularProgress />
              </Box>
            )}

            {multipleSets && multipleSets.length > 0 && (
              <ContrastiveGroupsTable
                groups={multipleSets.map(set => ({
                  words: set.words.map(w => ({
                    ...w,
                    position: set.position,
                  })),
                }))}
                mode="multiple"
                substitutePhoneme={substitutePhoneme}
                enableSelection={true}
                exportFilename={`phonolex_multiple_opposition_${substitutePhoneme}.csv`}
              />
            )}

            {!loading && multipleSets && multipleSets.length === 0 && (
              <Alert severity="warning">
                No minimal sets found for this collapse in {position} position. Try selecting a different position or different target phonemes.
              </Alert>
            )}
          </Box>
        )}
      </Stack>

      {/* Phoneme Picker Dialog */}
      <PhonemePickerDialog
        open={phonemePickerOpen}
        onClose={() => setPhonemePickerOpen(false)}
        onSelect={handlePhonemeSelect}
        filter={activeField === 'sonorants' ? 'sonorants' : activeField === 'obstruents' ? 'obstruents' : undefined}
      />
    </Box>
  );
};

export default ContrastiveInterventionTool;
