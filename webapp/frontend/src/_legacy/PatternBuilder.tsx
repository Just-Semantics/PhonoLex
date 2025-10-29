/**
 * PatternBuilder Component
 *
 * Interactive form for building phonological patterns position-by-position.
 * Implements progressive disclosure: simple phoneme type → specific phonemes → features
 */

import React, { useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  IconButton,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Stack,
  Paper,
  SelectChangeEvent,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';
import { useFilterStore } from '@stores/useFilterStore';
import type { PhonemeType } from '@types/phonology';

const PatternBuilder: React.FC = () => {
  const {
    pattern,
    availableVowels,
    availableConsonants,
    isPhonemesLoaded,
    addPosition,
    addContains,
    removeConstraint,
    updateConstraint,
    loadPhonemes,
  } = useFilterStore();

  // Load phonemes on mount
  useEffect(() => {
    if (!isPhonemesLoaded) {
      loadPhonemes();
    }
  }, [isPhonemesLoaded, loadPhonemes]);

  const handleAddPosition = () => {
    const nextPosition = pattern.filter(p => p.position !== null && p.position !== undefined).length;
    addPosition(nextPosition);
  };

  const handleAddContains = () => {
    addContains();
  };

  const handleRemoveConstraint = (index: number) => {
    removeConstraint(index);
  };

  const handlePhonemeTypeChange = (index: number, type: PhonemeType | '') => {
    updateConstraint(index, {
      phoneme_type: type || undefined,
      allowed_phonemes: undefined, // Reset specific selection
    });
  };

  const handleSpecificPhonemesChange = (index: number, phonemes: string[]) => {
    updateConstraint(index, {
      allowed_phonemes: phonemes.length > 0 ? phonemes : undefined,
    });
  };

  const getAvailablePhonemes = (phonemeType?: PhonemeType): string[] => {
    if (phonemeType === 'vowel') return availableVowels;
    if (phonemeType === 'consonant') return availableConsonants;
    return [...availableVowels, ...availableConsonants];
  };

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h5">Pattern Builder</Typography>
          <Stack direction="row" spacing={1}>
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={handleAddPosition}
              size="small"
            >
              Add Position
            </Button>
            <Button
              variant="outlined"
              startIcon={<AddIcon />}
              onClick={handleAddContains}
              size="small"
            >
              Contains
            </Button>
          </Stack>
        </Box>

        {pattern.length === 0 ? (
          <Paper
            sx={{
              p: 4,
              textAlign: 'center',
              bgcolor: 'background.default',
              border: '2px dashed',
              borderColor: 'divider',
            }}
          >
            <Typography variant="body1" color="text.secondary">
              Click "Add Position" to build position-by-position patterns
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Example: C + V + C (consonant-vowel-consonant)
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Or click "Contains" to find words with specific phonemes anywhere
            </Typography>
          </Paper>
        ) : (
          <Stack spacing={2}>
            {pattern.map((constraint, index) => (
              <PositionConstraint
                key={index}
                constraint={constraint}
                index={index}
                availablePhonemes={getAvailablePhonemes(constraint.phoneme_type)}
                onPhonemeTypeChange={handlePhonemeTypeChange}
                onSpecificPhonemesChange={handleSpecificPhonemesChange}
                onRemove={handleRemoveConstraint}
              />
            ))}
          </Stack>
        )}
      </CardContent>
    </Card>
  );
};

interface PositionConstraintProps {
  constraint: any; // PhonemeConstraint from store
  index: number;
  availablePhonemes: string[];
  onPhonemeTypeChange: (index: number, type: PhonemeType | '') => void;
  onSpecificPhonemesChange: (index: number, phonemes: string[]) => void;
  onRemove: (index: number) => void;
}

const PositionConstraint: React.FC<PositionConstraintProps> = ({
  constraint,
  index,
  availablePhonemes,
  onPhonemeTypeChange,
  onSpecificPhonemesChange,
  onRemove,
}) => {
  const isContainsMode = constraint.position === null || constraint.position === undefined;

  const handleTypeChange = (event: SelectChangeEvent<PhonemeType | ''>) => {
    onPhonemeTypeChange(index, event.target.value as PhonemeType | '');
  };

  const handlePhonemesChange = (event: SelectChangeEvent<string[]>) => {
    const value = event.target.value;
    onSpecificPhonemesChange(
      index,
      typeof value === 'string' ? value.split(',') : value
    );
  };

  return (
    <Paper
      sx={{
        p: 2,
        bgcolor: 'background.default',
        border: 1,
        borderColor: isContainsMode ? 'primary.main' : 'divider',
      }}
    >
      <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start' }}>
        {/* Position/Contains Label */}
        <Box
          sx={{
            minWidth: 80,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            pt: 1,
          }}
        >
          <Typography variant="caption" color="text.secondary">
            {isContainsMode ? 'Contains' : 'Position'}
          </Typography>
          <Typography variant="h6">
            {isContainsMode ? '?' : constraint.position}
          </Typography>
        </Box>

        {/* Phoneme Type Selector */}
        <FormControl sx={{ minWidth: 150 }} size="small">
          <InputLabel>Type</InputLabel>
          <Select
            value={constraint.phoneme_type || ''}
            label="Type"
            onChange={handleTypeChange}
          >
            <MenuItem value="">Any</MenuItem>
            <MenuItem value="vowel">Vowel</MenuItem>
            <MenuItem value="consonant">Consonant</MenuItem>
          </Select>
        </FormControl>

        {/* Specific Phonemes Selector */}
        {constraint.phoneme_type && (
          <FormControl sx={{ minWidth: 250, flexGrow: 1 }} size="small">
            <InputLabel>Phonemes</InputLabel>
            <Select
              multiple
              value={constraint.allowed_phonemes || []}
              label="Phonemes"
              onChange={handlePhonemesChange}
              renderValue={(selected) => (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {selected.map((value) => (
                    <Chip key={value} label={value} size="small" />
                  ))}
                </Box>
              )}
            >
              {availablePhonemes.map((phoneme) => (
                <MenuItem key={phoneme} value={phoneme}>
                  {phoneme}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        )}

        {/* Advanced Features (TODO) */}
        <IconButton size="small" disabled sx={{ mt: 0.5 }}>
          <SettingsIcon fontSize="small" />
        </IconButton>

        {/* Remove Button */}
        <IconButton
          size="small"
          color="error"
          onClick={() => onRemove(constraint.position)}
          sx={{ mt: 0.5 }}
        >
          <DeleteIcon fontSize="small" />
        </IconButton>
      </Box>

      {/* Summary */}
      {(constraint.phoneme_type || constraint.allowed_phonemes) && (
        <Box sx={{ mt: 1, ml: 10 }}>
          <Typography variant="caption" color="text.secondary">
            Pattern:{' '}
            {constraint.allowed_phonemes?.length > 0
              ? constraint.allowed_phonemes.join(', ')
              : constraint.phoneme_type === 'vowel'
              ? 'Any Vowel'
              : constraint.phoneme_type === 'consonant'
              ? 'Any Consonant'
              : 'Any Phoneme'}
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default PatternBuilder;
