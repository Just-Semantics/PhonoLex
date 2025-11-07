/**
 * Word List Table Component
 *
 * Displays single-word results with full psycholinguistic data:
 * - Checkbox selection per word
 * - All 12 psycholinguistic properties
 * - Sortable columns
 * - Mobile-responsive card view
 * - Integrated export menu
 * - Selection toolbar
 */

import React, { useState, useMemo, useRef, useEffect } from 'react';
import {
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  Button,
  Chip,
  Typography,
  Stack,
  Tooltip,
  Card,
  CardContent,
  Grid,
  Divider,
  useMediaQuery,
  useTheme,
  ToggleButton,
  ToggleButtonGroup,
  Fade,
  Checkbox,
} from '@mui/material';
import {
  ViewList as TableViewIcon,
  ViewModule as CardViewIcon,
  SwipeRounded as ScrollIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  ContentCopy as CopyIcon,
} from '@mui/icons-material';
import type { Word, SimilarWord } from '../../services/phonolexApi';
import ExportMenu from './ExportMenu';
import SelectionToolbar from './SelectionToolbar';

// ============================================================================
// Types
// ============================================================================

type SortField = 'word' | 'wcm_score' | 'msh_stage' | 'syllable_count' | 'similarity' |
  'frequency' | 'aoa' | 'imageability' | 'familiarity' | 'concreteness' |
  'valence' | 'arousal' | 'dominance';
type SortDirection = 'asc' | 'desc';

interface DisplayWord extends Word {
  similarity?: number;
}

export interface WordListTableProps {
  words: Word[] | SimilarWord[];
  showSimilarity?: boolean;
  enableSelection?: boolean;
  defaultSort?: SortField;
  onSelectionChange?: (selected: Word[]) => void;
  exportFilename?: string;
}

// ============================================================================
// Component
// ============================================================================

const WordListTable: React.FC<WordListTableProps> = ({
  words,
  showSimilarity = false,
  enableSelection = true,
  defaultSort = 'word',
  onSelectionChange,
  exportFilename = 'phonolex_words.csv',
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [sortField, setSortField] = useState<SortField>(defaultSort);
  const [sortDirection, setSortDirection] = useState<SortDirection>(
    defaultSort === 'similarity' ? 'desc' : 'asc'
  );
  const [viewMode, setViewMode] = useState<'table' | 'cards'>(isMobile ? 'cards' : 'table');
  const [showScrollHint, setShowScrollHint] = useState(true);
  const [isExpanded, setIsExpanded] = useState(false);
  const [selectedIndices, setSelectedIndices] = useState<number[]>([]);
  const tableContainerRef = useRef<HTMLDivElement>(null);

  // Threshold for showing collapse/expand
  const COLLAPSE_THRESHOLD = 50;
  const COLLAPSED_DISPLAY_COUNT = 25;

  // Determine if we have similarity results
  const isSimilarityResults = words.length > 0 && 'similarity' in words[0];

  // Extract display words
  const displayWords = useMemo((): DisplayWord[] => {
    if (isSimilarityResults) {
      return (words as SimilarWord[]).map(sr => ({
        ...sr.word,
        similarity: sr.similarity,
      }));
    }
    return words as Word[];
  }, [words, isSimilarityResults]);

  // Hide scroll hint after user scrolls
  useEffect(() => {
    const container = tableContainerRef.current;
    if (!container) return;

    const handleScroll = () => setShowScrollHint(false);
    container.addEventListener('scroll', handleScroll);
    return () => container.removeEventListener('scroll', handleScroll);
  }, []);

  // Reset scroll hint when view mode changes
  useEffect(() => {
    if (viewMode === 'table') setShowScrollHint(true);
  }, [viewMode]);

  // Notify parent of selection changes
  useEffect(() => {
    if (onSelectionChange) {
      const selected = selectedIndices.map(idx => displayWords[idx]);
      onSelectionChange(selected);
    }
  }, [selectedIndices, displayWords, onSelectionChange]);

  // Sort words
  const sortedWords = useMemo(() => {
    const sorted = [...displayWords];
    sorted.sort((a, b) => {
      let aVal: string | number | null, bVal: string | number | null;

      switch (sortField) {
        case 'word':
          aVal = a.word || '';
          bVal = b.word || '';
          break;
        case 'wcm_score':
          aVal = a.wcm_score || 0;
          bVal = b.wcm_score || 0;
          break;
        case 'msh_stage':
          aVal = a.msh_stage || 0;
          bVal = b.msh_stage || 0;
          break;
        case 'syllable_count':
          aVal = a.syllable_count || 0;
          bVal = b.syllable_count || 0;
          break;
        case 'frequency':
          aVal = a.frequency || 0;
          bVal = b.frequency || 0;
          break;
        case 'aoa':
          aVal = a.aoa || 999;
          bVal = b.aoa || 999;
          break;
        case 'imageability':
          aVal = a.imageability || 0;
          bVal = b.imageability || 0;
          break;
        case 'familiarity':
          aVal = a.familiarity || 0;
          bVal = b.familiarity || 0;
          break;
        case 'concreteness':
          aVal = a.concreteness || 0;
          bVal = b.concreteness || 0;
          break;
        case 'valence':
          aVal = a.valence || 0;
          bVal = b.valence || 0;
          break;
        case 'arousal':
          aVal = a.arousal || 0;
          bVal = b.arousal || 0;
          break;
        case 'dominance':
          aVal = a.dominance || 0;
          bVal = b.dominance || 0;
          break;
        case 'similarity':
          aVal = a.similarity || 0;
          bVal = b.similarity || 0;
          break;
        default:
          return 0;
      }

      if (sortDirection === 'asc') {
        return aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
      } else {
        return aVal > bVal ? -1 : aVal < bVal ? 1 : 0;
      }
    });
    return sorted;
  }, [displayWords, sortField, sortDirection]);

  // Display words (with collapse functionality)
  const displayedWords = useMemo(() => {
    const shouldCollapse = sortedWords.length > COLLAPSE_THRESHOLD;
    if (shouldCollapse && !isExpanded) {
      return sortedWords.slice(0, COLLAPSED_DISPLAY_COUNT);
    }
    return sortedWords;
  }, [sortedWords, isExpanded, COLLAPSE_THRESHOLD, COLLAPSED_DISPLAY_COUNT]);

  const showCollapseControls = sortedWords.length > COLLAPSE_THRESHOLD;

  // Handle sort
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  // Selection handlers
  const handleSelectAll = () => {
    setSelectedIndices(sortedWords.map((_, idx) => idx));
  };

  const handleClearAll = () => {
    setSelectedIndices([]);
  };

  const handleToggleRow = (idx: number) => {
    setSelectedIndices(prev =>
      prev.includes(idx)
        ? prev.filter(i => i !== idx)
        : [...prev, idx]
    );
  };

  // Copy words to clipboard (simple copy - no selection awareness)
  const copyWords = () => {
    const text = sortedWords.map(w => w.word).join('\n');
    navigator.clipboard.writeText(text);
  };

  if (words.length === 0) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography color="text.secondary">No results found</Typography>
      </Paper>
    );
  }

  return (
    <Box>
      {/* Selection Toolbar */}
      {enableSelection && (
        <SelectionToolbar
          totalCount={sortedWords.length}
          selectedCount={selectedIndices.length}
          selectedIndices={selectedIndices}
          onSelectAll={handleSelectAll}
          onClearAll={handleClearAll}
          data={sortedWords}
          dataType="words"
          exportFilename={exportFilename}
        />
      )}

      {/* Header */}
      <Stack
        direction={{ xs: 'column', sm: 'row' }}
        justifyContent="space-between"
        alignItems={{ xs: 'stretch', sm: 'center' }}
        spacing={{ xs: 1.5, sm: 2 }}
        sx={{ mb: { xs: 1.5, sm: 2 } }}
      >
        <Stack direction="row" alignItems="center" spacing={1}>
          <Typography variant="h6" sx={{ fontSize: { xs: '1rem', sm: '1.25rem' } }}>
            {words.length} Words Found
          </Typography>
          {showCollapseControls && (
            <Chip
              label={isExpanded ? `Showing all ${words.length}` : `Showing ${COLLAPSED_DISPLAY_COUNT} of ${words.length}`}
              size="small"
              color={isExpanded ? 'primary' : 'default'}
              sx={{ fontSize: '0.75rem' }}
            />
          )}
        </Stack>

        <Stack
          direction={{ xs: 'column', sm: 'row' }}
          spacing={{ xs: 1, sm: 1 }}
          alignItems={{ xs: 'stretch', sm: 'center' }}
        >
          {/* View Mode Toggle (Mobile only) */}
          <ToggleButtonGroup
            value={viewMode}
            exclusive
            onChange={(_, newMode) => newMode && setViewMode(newMode)}
            size="small"
            fullWidth
            sx={{
              display: { xs: 'flex', md: 'none' },
              '& .MuiToggleButton-root': {
                minHeight: 44,
                flex: 1,
              },
            }}
          >
            <ToggleButton value="table">
              <Tooltip title="Table view">
                <TableViewIcon fontSize="small" />
              </Tooltip>
            </ToggleButton>
            <ToggleButton value="cards">
              <Tooltip title="Card view">
                <CardViewIcon fontSize="small" />
              </Tooltip>
            </ToggleButton>
          </ToggleButtonGroup>

          {/* Export Buttons */}
          <Stack direction="row" spacing={{ xs: 1, sm: 1 }} sx={{ width: { xs: '100%', sm: 'auto' } }}>
            {!enableSelection && (
              <Tooltip title="Copy words to clipboard">
                <Button
                  size="small"
                  startIcon={<CopyIcon />}
                  onClick={copyWords}
                  sx={{ minHeight: 44, width: { xs: '100%', sm: 'auto' } }}
                >
                  Copy
                </Button>
              </Tooltip>
            )}

            <ExportMenu
              data={sortedWords}
              dataType="words"
              selectedIndices={enableSelection ? selectedIndices : undefined}
              filename={exportFilename}
            />
          </Stack>
        </Stack>
      </Stack>

      {/* Card View for Mobile */}
      {viewMode === 'cards' ? (
        <Stack spacing={{ xs: 1.5, sm: 2 }}>
          {displayedWords.map((word, idx) => (
            <Card
              key={idx}
              variant="outlined"
              sx={{
                transition: 'all 0.2s ease',
                bgcolor: selectedIndices.includes(idx) ? 'primary.light' : 'background.paper',
                borderColor: selectedIndices.includes(idx) ? 'primary.main' : 'divider',
                '&:hover': {
                  boxShadow: 2,
                  borderColor: 'primary.main',
                },
              }}
            >
              <CardContent sx={{ px: { xs: 1.5, sm: 2 }, py: { xs: 1.5, sm: 2 }, '&:last-child': { pb: { xs: 1.5, sm: 2 } } }}>
                <Stack spacing={{ xs: 1.5, sm: 2 }}>
                  {/* Header: Checkbox + Word + IPA */}
                  <Stack direction="row" alignItems="flex-start" spacing={1}>
                    {enableSelection && (
                      <Checkbox
                        checked={selectedIndices.includes(idx)}
                        onChange={() => handleToggleRow(idx)}
                        size="small"
                      />
                    )}
                    <Box flex={1}>
                      <Typography variant="h6" fontWeight={600} sx={{ fontSize: { xs: '1.125rem', sm: '1.25rem' }, color: 'primary.main' }}>
                        {word.word}
                      </Typography>
                      <Typography
                        variant="body2"
                        fontFamily="monospace"
                        color="text.secondary"
                        sx={{ fontSize: { xs: '0.8125rem', sm: '0.875rem' }, mt: 0.25 }}
                      >
                        {word.ipa}
                      </Typography>
                    </Box>
                  </Stack>

                  <Divider />

                  {/* Key Metrics */}
                  <Grid container spacing={{ xs: 1, sm: 1.5 }}>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="caption" color="text.secondary">Syllables</Typography>
                      <Box>
                        <Chip label={word.syllable_count || 0} size="small" />
                      </Box>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="caption" color="text.secondary">WCM</Typography>
                      <Box>
                        <Chip
                          label={word.wcm_score?.toFixed(1) || '0.0'}
                          size="small"
                          color={
                            (word.wcm_score || 0) < 5 ? 'success' :
                            (word.wcm_score || 0) < 10 ? 'warning' : 'error'
                          }
                        />
                      </Box>
                    </Grid>
                    {(showSimilarity || word.similarity !== undefined) && (
                      <Grid item xs={6} sm={3}>
                        <Typography variant="caption" color="text.secondary">Similarity</Typography>
                        <Box>
                          <Chip
                            label={word.similarity?.toFixed(3) || 'N/A'}
                            size="small"
                            color="primary"
                          />
                        </Box>
                      </Grid>
                    )}
                  </Grid>

                  {/* Psycholinguistic Properties */}
                  {(word.frequency || word.aoa || word.imageability) && (
                    <>
                      <Divider />
                      <Grid container spacing={1}>
                        {word.frequency !== null && word.frequency !== undefined && (
                          <Grid item xs={4}>
                            <Typography variant="caption" color="text.secondary">Freq</Typography>
                            <Typography variant="body2" fontFamily="monospace">
                              {word.frequency.toFixed(1)}
                            </Typography>
                          </Grid>
                        )}
                        {word.aoa !== null && word.aoa !== undefined && (
                          <Grid item xs={4}>
                            <Typography variant="caption" color="text.secondary">AoA</Typography>
                            <Typography variant="body2" fontFamily="monospace">
                              {word.aoa.toFixed(1)}
                            </Typography>
                          </Grid>
                        )}
                        {word.imageability !== null && word.imageability !== undefined && (
                          <Grid item xs={4}>
                            <Typography variant="caption" color="text.secondary">Image</Typography>
                            <Typography variant="body2" fontFamily="monospace">
                              {word.imageability.toFixed(1)}
                            </Typography>
                          </Grid>
                        )}
                      </Grid>
                    </>
                  )}
                </Stack>
              </CardContent>
            </Card>
          ))}

          {/* Expand/Collapse Button */}
          {showCollapseControls && (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
              <Button
                variant="outlined"
                onClick={() => setIsExpanded(!isExpanded)}
                startIcon={isExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                size="small"
              >
                {isExpanded
                  ? `Collapse (showing ${COLLAPSED_DISPLAY_COUNT})`
                  : `Show All ${sortedWords.length} Results`}
              </Button>
            </Box>
          )}
        </Stack>
      ) : (
        /* Table View */
        <Box sx={{ position: 'relative' }}>
          {/* Scroll Hint Overlay */}
          <Fade in={showScrollHint && isMobile} timeout={1000}>
            <Box
              sx={{
                position: 'absolute',
                top: '50%',
                right: 16,
                transform: 'translateY(-50%)',
                zIndex: 10,
                bgcolor: 'primary.main',
                color: 'white',
                px: 2,
                py: 1,
                borderRadius: 2,
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                boxShadow: 3,
                pointerEvents: 'none',
              }}
            >
              <ScrollIcon />
              <Typography variant="caption" fontWeight={600}>
                Scroll to see more
              </Typography>
            </Box>
          </Fade>

          <TableContainer
            ref={tableContainerRef}
            component={Paper}
            sx={{
              overflowX: 'auto',
              overflowY: 'auto',
              maxHeight: '70vh',
              WebkitOverflowScrolling: 'touch',
              position: 'relative',
              '&::-webkit-scrollbar': { height: 8, width: 8 },
              '&::-webkit-scrollbar-thumb': {
                backgroundColor: 'rgba(0,0,0,0.2)',
                borderRadius: 4,
              },
              '&::-webkit-scrollbar-track': {
                backgroundColor: 'rgba(0,0,0,0.05)',
              },
            }}
          >
            <Table size="small" sx={{ minWidth: { xs: 800, sm: 'auto' } }} stickyHeader>
              <TableHead>
                <TableRow>
                  {/* Checkbox Column */}
                  {enableSelection && (
                    <TableCell
                      padding="checkbox"
                      sx={{
                        position: 'sticky',
                        left: 0,
                        zIndex: 3,
                        bgcolor: 'background.paper',
                        borderRight: 1,
                        borderColor: 'divider',
                      }}
                    >
                      <Checkbox
                        indeterminate={selectedIndices.length > 0 && selectedIndices.length < sortedWords.length}
                        checked={sortedWords.length > 0 && selectedIndices.length === sortedWords.length}
                        onChange={(e) => e.target.checked ? handleSelectAll() : handleClearAll()}
                      />
                    </TableCell>
                  )}

                  {/* Word Column */}
                  <TableCell
                    sx={{
                      position: 'sticky',
                      left: enableSelection ? 58 : 0,
                      zIndex: 3,
                      bgcolor: 'background.paper',
                      borderRight: 1,
                      borderColor: 'divider',
                      boxShadow: '2px 0 4px rgba(0,0,0,0.05)',
                    }}
                  >
                    <TableSortLabel
                      active={sortField === 'word'}
                      direction={sortField === 'word' ? sortDirection : 'asc'}
                      onClick={() => handleSort('word')}
                    >
                      Word
                    </TableSortLabel>
                  </TableCell>

                  {/* Other Columns */}
                  <TableCell sx={{ whiteSpace: 'nowrap' }}>IPA</TableCell>
                  {(showSimilarity || isSimilarityResults) && (
                    <TableCell align="center">
                      <TableSortLabel
                        active={sortField === 'similarity'}
                        direction={sortField === 'similarity' ? sortDirection : 'asc'}
                        onClick={() => handleSort('similarity')}
                      >
                        Similarity
                      </TableSortLabel>
                    </TableCell>
                  )}
                  <TableCell align="center">
                    <TableSortLabel
                      active={sortField === 'syllable_count'}
                      direction={sortField === 'syllable_count' ? sortDirection : 'asc'}
                      onClick={() => handleSort('syllable_count')}
                    >
                      Syllables
                    </TableSortLabel>
                  </TableCell>
                  <TableCell align="center">
                    <Tooltip title="Word Complexity Measure">
                      <TableSortLabel
                        active={sortField === 'wcm_score'}
                        direction={sortField === 'wcm_score' ? sortDirection : 'asc'}
                        onClick={() => handleSort('wcm_score')}
                      >
                        WCM
                      </TableSortLabel>
                    </Tooltip>
                  </TableCell>
                  <TableCell align="center">
                    <Tooltip title="Motor Speech Hierarchy (1-6)">
                      <TableSortLabel
                        active={sortField === 'msh_stage'}
                        direction={sortField === 'msh_stage' ? sortDirection : 'asc'}
                        onClick={() => handleSort('msh_stage')}
                      >
                        MSH
                      </TableSortLabel>
                    </Tooltip>
                  </TableCell>
                  <TableCell align="center">
                    <Tooltip title="Word frequency (per million words)">
                      <TableSortLabel
                        active={sortField === 'frequency'}
                        direction={sortField === 'frequency' ? sortDirection : 'asc'}
                        onClick={() => handleSort('frequency')}
                      >
                        Freq
                      </TableSortLabel>
                    </Tooltip>
                  </TableCell>
                  <TableCell align="center">
                    <Tooltip title="Age of Acquisition (1-7)">
                      <TableSortLabel
                        active={sortField === 'aoa'}
                        direction={sortField === 'aoa' ? sortDirection : 'asc'}
                        onClick={() => handleSort('aoa')}
                      >
                        AoA
                      </TableSortLabel>
                    </Tooltip>
                  </TableCell>
                  <TableCell align="center">
                    <Tooltip title="Imageability (1-7)">
                      <TableSortLabel
                        active={sortField === 'imageability'}
                        direction={sortField === 'imageability' ? sortDirection : 'asc'}
                        onClick={() => handleSort('imageability')}
                      >
                        Image
                      </TableSortLabel>
                    </Tooltip>
                  </TableCell>
                  <TableCell align="center">
                    <Tooltip title="Familiarity (1-7)">
                      <TableSortLabel
                        active={sortField === 'familiarity'}
                        direction={sortField === 'familiarity' ? sortDirection : 'asc'}
                        onClick={() => handleSort('familiarity')}
                      >
                        Famil
                      </TableSortLabel>
                    </Tooltip>
                  </TableCell>
                  <TableCell align="center">
                    <Tooltip title="Concreteness (1-5)">
                      <TableSortLabel
                        active={sortField === 'concreteness'}
                        direction={sortField === 'concreteness' ? sortDirection : 'asc'}
                        onClick={() => handleSort('concreteness')}
                      >
                        Concr
                      </TableSortLabel>
                    </Tooltip>
                  </TableCell>
                  <TableCell align="center">
                    <Tooltip title="Valence (1-9)">
                      <TableSortLabel
                        active={sortField === 'valence'}
                        direction={sortField === 'valence' ? sortDirection : 'asc'}
                        onClick={() => handleSort('valence')}
                      >
                        Val
                      </TableSortLabel>
                    </Tooltip>
                  </TableCell>
                  <TableCell align="center">
                    <Tooltip title="Arousal (1-9)">
                      <TableSortLabel
                        active={sortField === 'arousal'}
                        direction={sortField === 'arousal' ? sortDirection : 'asc'}
                        onClick={() => handleSort('arousal')}
                      >
                        Aro
                      </TableSortLabel>
                    </Tooltip>
                  </TableCell>
                  <TableCell align="center">
                    <Tooltip title="Dominance (1-9)">
                      <TableSortLabel
                        active={sortField === 'dominance'}
                        direction={sortField === 'dominance' ? sortDirection : 'asc'}
                        onClick={() => handleSort('dominance')}
                      >
                        Dom
                      </TableSortLabel>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {displayedWords.map((word, idx) => (
                  <TableRow
                    key={idx}
                    hover
                    selected={selectedIndices.includes(idx)}
                    onClick={() => enableSelection && handleToggleRow(idx)}
                    sx={{ cursor: enableSelection ? 'pointer' : 'default' }}
                  >
                    {/* Checkbox */}
                    {enableSelection && (
                      <TableCell
                        padding="checkbox"
                        sx={{
                          position: 'sticky',
                          left: 0,
                          zIndex: 1,
                          bgcolor: 'background.paper',
                          borderRight: 1,
                          borderColor: 'divider',
                        }}
                      >
                        <Checkbox checked={selectedIndices.includes(idx)} />
                      </TableCell>
                    )}

                    {/* Word */}
                    <TableCell
                      sx={{
                        position: 'sticky',
                        left: enableSelection ? 58 : 0,
                        zIndex: 1,
                        bgcolor: 'background.paper',
                        borderRight: 1,
                        borderColor: 'divider',
                        boxShadow: '2px 0 4px rgba(0,0,0,0.05)',
                      }}
                    >
                      <Typography variant="body2" fontWeight={500}>
                        {word.word}
                      </Typography>
                    </TableCell>

                    {/* IPA */}
                    <TableCell sx={{ whiteSpace: 'nowrap' }}>
                      <Typography variant="body2" fontFamily="monospace" color="text.secondary">
                        {word.ipa}
                      </Typography>
                    </TableCell>

                    {/* Similarity */}
                    {(showSimilarity || isSimilarityResults) && (
                      <TableCell align="center">
                        <Chip
                          label={word.similarity?.toFixed(3) || 'N/A'}
                          size="small"
                          color="primary"
                        />
                      </TableCell>
                    )}

                    {/* Syllables */}
                    <TableCell align="center">
                      <Chip label={word.syllable_count || 0} size="small" color="default" />
                    </TableCell>

                    {/* WCM */}
                    <TableCell align="center">
                      <Chip
                        label={word.wcm_score?.toFixed(1) || '0.0'}
                        size="small"
                        color={
                          (word.wcm_score || 0) < 5 ? 'success' :
                          (word.wcm_score || 0) < 10 ? 'warning' : 'error'
                        }
                      />
                    </TableCell>

                    {/* MSH */}
                    <TableCell align="center">
                      <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
                        {word.msh_stage?.toString() || '-'}
                      </Typography>
                    </TableCell>

                    {/* Frequency */}
                    <TableCell align="center">
                      <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
                        {word.frequency ? word.frequency.toFixed(1) : '-'}
                      </Typography>
                    </TableCell>

                    {/* AoA */}
                    <TableCell align="center">
                      <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
                        {word.aoa ? word.aoa.toFixed(1) : '-'}
                      </Typography>
                    </TableCell>

                    {/* Imageability */}
                    <TableCell align="center">
                      <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
                        {word.imageability ? word.imageability.toFixed(1) : '-'}
                      </Typography>
                    </TableCell>

                    {/* Familiarity */}
                    <TableCell align="center">
                      <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
                        {word.familiarity ? word.familiarity.toFixed(1) : '-'}
                      </Typography>
                    </TableCell>

                    {/* Concreteness */}
                    <TableCell align="center">
                      <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
                        {word.concreteness ? word.concreteness.toFixed(1) : '-'}
                      </Typography>
                    </TableCell>

                    {/* Valence */}
                    <TableCell align="center">
                      <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
                        {word.valence ? word.valence.toFixed(1) : '-'}
                      </Typography>
                    </TableCell>

                    {/* Arousal */}
                    <TableCell align="center">
                      <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
                        {word.arousal ? word.arousal.toFixed(1) : '-'}
                      </Typography>
                    </TableCell>

                    {/* Dominance */}
                    <TableCell align="center">
                      <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
                        {word.dominance ? word.dominance.toFixed(1) : '-'}
                      </Typography>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>

          {/* Expand/Collapse Button */}
          {showCollapseControls && (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
              <Button
                variant="outlined"
                onClick={() => setIsExpanded(!isExpanded)}
                startIcon={isExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                size="small"
              >
                {isExpanded
                  ? `Collapse (showing ${COLLAPSED_DISPLAY_COUNT})`
                  : `Show All ${sortedWords.length} Results`}
              </Button>
            </Box>
          )}
        </Box>
      )}
    </Box>
  );
};

export default WordListTable;
