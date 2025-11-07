/**
 * Contrastive Groups Table Component
 *
 * Displays contrastive intervention results (pairs, triplets, quadruplets, quintuplets):
 * - Unified table format for all group sizes
 * - Checkbox selection per group
 * - Position and phoneme contrast indicators
 * - Compact property display
 * - Integrated export menu
 * - Selection toolbar
 */

import React, { useState, useMemo } from 'react';
import {
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  Stack,
  Chip,
  Checkbox,
  Card,
  CardContent,
  Grid,
  Divider,
  useMediaQuery,
  useTheme,
  Button,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
} from '@mui/icons-material';
import type { MinimalPair } from '../../services/phonolexApi';
import ExportMenu, { type ContrastiveGroup } from './ExportMenu';
import SelectionToolbar from './SelectionToolbar';

// ============================================================================
// Types
// ============================================================================

export interface ContrastiveGroupsTableProps {
  // For minimal/maximal pairs
  pairs?: MinimalPair[];

  // For multiple opposition sets (triplets, quadruplets, quintuplets)
  groups?: ContrastiveGroup[];

  mode: 'minimal' | 'maximal' | 'multiple';
  substitutePhoneme?: string;  // For multiple opposition mode
  enableSelection?: boolean;
  exportFilename?: string;
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Format position as readable string (Initial/Medial/Final)
 */
function formatPosition(position: number | undefined): string {
  if (position === undefined || position === null) return 'Any';
  if (position === 0) return 'Initial';
  if (position === -1) return 'Final';
  return `Medial (${position})`;
}

/**
 * Get position label with color coding
 */
function getPositionChip(position: number | undefined) {
  const label = formatPosition(position);
  const color = position === 0 ? 'primary' : position === -1 ? 'secondary' : 'default';
  return <Chip label={label} size="small" color={color} />;
}

// ============================================================================
// Component
// ============================================================================

const ContrastiveGroupsTable: React.FC<ContrastiveGroupsTableProps> = ({
  pairs,
  groups,
  mode,
  substitutePhoneme,
  enableSelection = true,
  exportFilename = 'phonolex_contrastive.csv',
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [selectedIndices, setSelectedIndices] = useState<number[]>([]);
  const [isExpanded, setIsExpanded] = useState(false);

  // Threshold for collapse
  const COLLAPSE_THRESHOLD = 50;
  const COLLAPSED_DISPLAY_COUNT = 25;

  // Convert pairs to groups format for unified rendering
  const allGroups = useMemo((): ContrastiveGroup[] => {
    if (pairs) {
      return pairs.map(pair => ({
        words: [
          {
            word: pair.word1,
            phoneme: pair.phoneme1 || '',
            position: pair.position || 0,
          },
          {
            word: pair.word2,
            phoneme: pair.phoneme2 || '',
            position: pair.position || 0,
          },
        ],
      }));
    }
    return groups || [];
  }, [pairs, groups]);

  const totalCount = allGroups.length;

  // Display groups (with collapse for long lists)
  const displayedGroups = useMemo(() => {
    const shouldCollapse = totalCount > COLLAPSE_THRESHOLD;
    if (shouldCollapse && !isExpanded) {
      return allGroups.slice(0, COLLAPSED_DISPLAY_COUNT);
    }
    return allGroups;
  }, [allGroups, isExpanded, totalCount, COLLAPSE_THRESHOLD, COLLAPSED_DISPLAY_COUNT]);

  const showCollapseControls = totalCount > COLLAPSE_THRESHOLD;

  // Selection handlers
  const handleSelectAll = () => {
    setSelectedIndices(allGroups.map((_, idx) => idx));
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

  // Export data type
  const exportDataType = mode === 'multiple' ? 'groups' : 'pairs';
  const exportData = pairs || allGroups;

  if (allGroups.length === 0) {
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
          totalCount={totalCount}
          selectedCount={selectedIndices.length}
          selectedIndices={selectedIndices}
          onSelectAll={handleSelectAll}
          onClearAll={handleClearAll}
          data={exportData}
          dataType={exportDataType}
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
        <Typography variant="h6" sx={{ fontSize: { xs: '1rem', sm: '1.25rem' } }}>
          {totalCount} {mode === 'multiple' ? 'Sets' : 'Pairs'} Found
        </Typography>

        <ExportMenu
          data={exportData}
          dataType={exportDataType}
          selectedIndices={enableSelection ? selectedIndices : undefined}
          filename={exportFilename}
        />
      </Stack>

      {/* Mobile Card View */}
      {isMobile ? (
        <Stack spacing={2}>
          {displayedGroups.map((group, groupIdx) => (
            <Card
              key={groupIdx}
              variant="outlined"
              sx={{
                bgcolor: selectedIndices.includes(groupIdx) ? 'primary.light' : 'background.paper',
                borderColor: selectedIndices.includes(groupIdx) ? 'primary.main' : 'divider',
              }}
            >
              <CardContent>
                <Stack spacing={2}>
                  {/* Header: Checkbox + Group Number */}
                  <Stack direction="row" alignItems="center" spacing={1}>
                    {enableSelection && (
                      <Checkbox
                        checked={selectedIndices.includes(groupIdx)}
                        onChange={() => handleToggleRow(groupIdx)}
                      />
                    )}
                    <Typography variant="h6" fontWeight={600}>
                      {mode === 'multiple' ? 'Set' : 'Pair'} {groupIdx + 1}
                    </Typography>
                  </Stack>

                  <Divider />

                  {/* Words in Group */}
                  <Grid container spacing={2}>
                    {group.words.map((w, wordIdx) => (
                      <Grid item xs={12} sm={6} key={wordIdx}>
                        <Box
                          sx={{
                            p: 1.5,
                            bgcolor: 'background.default',
                            borderRadius: 1,
                            border: 1,
                            borderColor: 'divider',
                          }}
                        >
                          <Stack spacing={1}>
                            {/* Word + IPA */}
                            <Box>
                              <Typography variant="body1" fontWeight={600} color="primary.main">
                                {w.word.word}
                              </Typography>
                              <Typography variant="body2" fontFamily="monospace" color="text.secondary">
                                {w.word.ipa}
                              </Typography>
                            </Box>

                            {/* Phoneme Badge */}
                            <Box>
                              <Chip
                                label={w.phoneme}
                                size="small"
                                color={
                                  mode === 'multiple' && substitutePhoneme && w.phoneme === substitutePhoneme
                                    ? 'error'
                                    : 'primary'
                                }
                              />
                              {getPositionChip(w.position)}
                            </Box>

                            {/* Key Properties */}
                            <Stack direction="row" spacing={1} flexWrap="wrap">
                              {w.word.wcm_score && (
                                <Chip label={`WCM: ${w.word.wcm_score.toFixed(1)}`} size="small" variant="outlined" />
                              )}
                              {w.word.aoa && (
                                <Chip label={`AoA: ${w.word.aoa.toFixed(1)}`} size="small" variant="outlined" />
                              )}
                            </Stack>
                          </Stack>
                        </Box>
                      </Grid>
                    ))}
                  </Grid>
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
                  : `Show All ${totalCount} Results`}
              </Button>
            </Box>
          )}
        </Stack>
      ) : (
        /* Desktop Table View */
        <Box>
          <TableContainer component={Paper}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  {/* Checkbox */}
                  {enableSelection && (
                    <TableCell padding="checkbox">
                      <Checkbox
                        indeterminate={selectedIndices.length > 0 && selectedIndices.length < totalCount}
                        checked={totalCount > 0 && selectedIndices.length === totalCount}
                        onChange={(e) => e.target.checked ? handleSelectAll() : handleClearAll()}
                      />
                    </TableCell>
                  )}

                  {/* Group Number */}
                  <TableCell sx={{ fontWeight: 600 }}>
                    {mode === 'multiple' ? 'Set' : 'Pair'}
                  </TableCell>

                  {/* Words */}
                  <TableCell sx={{ fontWeight: 600 }}>Words</TableCell>

                  {/* Position */}
                  <TableCell sx={{ fontWeight: 600 }}>Position</TableCell>

                  {/* Phoneme Contrast */}
                  <TableCell sx={{ fontWeight: 600 }}>Phoneme</TableCell>

                  {/* Key Properties */}
                  <TableCell sx={{ fontWeight: 600 }} align="center">WCM</TableCell>
                  <TableCell sx={{ fontWeight: 600 }} align="center">AoA</TableCell>
                  <TableCell sx={{ fontWeight: 600 }} align="center">Frequency</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {displayedGroups.map((group, groupIdx) => (
                  <TableRow
                    key={groupIdx}
                    hover
                    selected={selectedIndices.includes(groupIdx)}
                    onClick={() => enableSelection && handleToggleRow(groupIdx)}
                    sx={{ cursor: enableSelection ? 'pointer' : 'default' }}
                  >
                    {/* Checkbox */}
                    {enableSelection && (
                      <TableCell padding="checkbox">
                        <Checkbox checked={selectedIndices.includes(groupIdx)} />
                      </TableCell>
                    )}

                    {/* Group Number */}
                    <TableCell>
                      <Typography variant="body2" fontWeight={500}>
                        {groupIdx + 1}
                      </Typography>
                    </TableCell>

                    {/* Words Column */}
                    <TableCell>
                      <Stack spacing={0.5}>
                        {group.words.map((w, wordIdx) => (
                          <Box key={wordIdx}>
                            <Typography variant="body2" fontWeight={500}>
                              {w.word.word}
                            </Typography>
                            <Typography variant="caption" fontFamily="monospace" color="text.secondary">
                              {w.word.ipa}
                            </Typography>
                          </Box>
                        ))}
                      </Stack>
                    </TableCell>

                    {/* Position */}
                    <TableCell>
                      {getPositionChip(group.words[0].position)}
                    </TableCell>

                    {/* Phoneme */}
                    <TableCell>
                      <Stack direction="row" spacing={0.5} flexWrap="wrap">
                        {group.words.map((w, wordIdx) => (
                          <Chip
                            key={wordIdx}
                            label={w.phoneme}
                            size="small"
                            color={
                              mode === 'multiple' && substitutePhoneme && w.phoneme === substitutePhoneme
                                ? 'error'
                                : 'primary'
                            }
                          />
                        ))}
                      </Stack>
                    </TableCell>

                    {/* WCM */}
                    <TableCell align="center">
                      <Stack spacing={0.25}>
                        {group.words.map((w, wordIdx) => (
                          <Typography key={wordIdx} variant="body2" fontFamily="monospace" color="text.secondary">
                            {w.word.wcm_score?.toFixed(1) || '-'}
                          </Typography>
                        ))}
                      </Stack>
                    </TableCell>

                    {/* AoA */}
                    <TableCell align="center">
                      <Stack spacing={0.25}>
                        {group.words.map((w, wordIdx) => (
                          <Typography key={wordIdx} variant="body2" fontFamily="monospace" color="text.secondary">
                            {w.word.aoa?.toFixed(1) || '-'}
                          </Typography>
                        ))}
                      </Stack>
                    </TableCell>

                    {/* Frequency */}
                    <TableCell align="center">
                      <Stack spacing={0.25}>
                        {group.words.map((w, wordIdx) => (
                          <Typography key={wordIdx} variant="body2" fontFamily="monospace" color="text.secondary">
                            {w.word.frequency?.toFixed(1) || '-'}
                          </Typography>
                        ))}
                      </Stack>
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
                  : `Show All ${totalCount} Results`}
              </Button>
            </Box>
          )}
        </Box>
      )}
    </Box>
  );
};

export default ContrastiveGroupsTable;
