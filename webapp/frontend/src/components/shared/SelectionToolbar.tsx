/**
 * Selection Toolbar Component
 *
 * Displays selection status and provides bulk selection controls:
 * - Shows count of selected items
 * - Select All button
 * - Clear selection button
 * - Export selected button
 */

import React from 'react';
import {
  Box,
  Stack,
  Typography,
  Button,
  Chip,
  Fade,
} from '@mui/material';
import {
  SelectAll as SelectAllIcon,
  Clear as ClearIcon,
} from '@mui/icons-material';
import ExportMenu, { type ExportData } from './ExportMenu';

// ============================================================================
// Types
// ============================================================================

export interface SelectionToolbarProps {
  totalCount: number;
  selectedCount: number;
  selectedIndices: number[];
  onSelectAll: () => void;
  onClearAll: () => void;
  data: ExportData;
  dataType: 'words' | 'pairs' | 'groups';
  exportFilename?: string;
}

// ============================================================================
// Component
// ============================================================================

const SelectionToolbar: React.FC<SelectionToolbarProps> = ({
  totalCount,
  selectedCount,
  selectedIndices,
  onSelectAll,
  onClearAll,
  data,
  dataType,
  exportFilename = 'phonolex_export.csv',
}) => {
  const hasSelection = selectedCount > 0;
  const allSelected = selectedCount === totalCount;

  return (
    <Fade in={hasSelection} unmountOnExit>
      <Box
        sx={{
          bgcolor: 'primary.light',
          color: 'primary.contrastText',
          px: 2,
          py: 1.5,
          borderRadius: 1,
          mb: 2,
        }}
      >
        <Stack
          direction={{ xs: 'column', sm: 'row' }}
          alignItems={{ xs: 'stretch', sm: 'center' }}
          justifyContent="space-between"
          spacing={2}
        >
          {/* Selection Count */}
          <Stack direction="row" alignItems="center" spacing={1}>
            <Chip
              label={selectedCount}
              size="small"
              sx={{
                bgcolor: 'background.paper',
                color: 'primary.main',
                fontWeight: 600,
              }}
            />
            <Typography variant="body2" fontWeight={500}>
              of {totalCount} selected
            </Typography>
          </Stack>

          {/* Action Buttons */}
          <Stack
            direction={{ xs: 'column', sm: 'row' }}
            spacing={1}
            sx={{ width: { xs: '100%', sm: 'auto' } }}
          >
            {!allSelected && (
              <Button
                size="small"
                startIcon={<SelectAllIcon />}
                onClick={onSelectAll}
                sx={{
                  bgcolor: 'background.paper',
                  color: 'primary.main',
                  minHeight: 36,
                  '&:hover': {
                    bgcolor: 'grey.100',
                  },
                }}
              >
                Select All
              </Button>
            )}

            <Button
              size="small"
              startIcon={<ClearIcon />}
              onClick={onClearAll}
              sx={{
                bgcolor: 'background.paper',
                color: 'text.primary',
                minHeight: 36,
                '&:hover': {
                  bgcolor: 'grey.100',
                },
              }}
            >
              Clear
            </Button>

            {/* Export Selected */}
            <Box
              sx={{
                '& button': {
                  bgcolor: 'background.paper',
                  color: 'primary.main',
                  '&:hover': {
                    bgcolor: 'grey.100',
                  },
                },
              }}
            >
              <ExportMenu
                data={data}
                dataType={dataType}
                selectedIndices={selectedIndices}
                filename={exportFilename}
              />
            </Box>
          </Stack>
        </Stack>
      </Box>
    </Fade>
  );
};

export default SelectionToolbar;
