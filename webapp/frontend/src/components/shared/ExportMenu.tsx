/**
 * Export Menu Component
 *
 * Provides unified export functionality for word lists and contrastive groups:
 * - Copy as plain text (newline-separated)
 * - Copy as numbered list
 * - Download CSV (full metadata)
 * - Download CSV (selected only)
 */

import React, { useState } from 'react';
import {
  Button,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  Download as DownloadIcon,
  ContentCopy as CopyIcon,
  FormatListNumbered as NumberedListIcon,
  Check as CheckIcon,
} from '@mui/icons-material';
import type { Word, MinimalPair } from '../../services/phonolexApi';

// ============================================================================
// Types
// ============================================================================

export interface ContrastiveGroup {
  words: Array<{
    word: Word;
    phoneme: string;
    position: number;
  }>;
}

export type ExportData = Word[] | MinimalPair[] | ContrastiveGroup[];

export interface ExportMenuProps {
  data: ExportData;
  dataType: 'words' | 'pairs' | 'groups';
  selectedIndices?: number[];
  filename?: string;
  disabled?: boolean;
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Extract words from data based on type
 */
function extractWords(data: ExportData, dataType: string): string[] {
  if (dataType === 'words') {
    return (data as Word[]).map(w => w.word);
  } else if (dataType === 'pairs') {
    return (data as MinimalPair[]).flatMap(p => [p.word1.word, p.word2.word]);
  } else {
    // groups
    return (data as ContrastiveGroup[]).flatMap(g => g.words.map(w => w.word.word));
  }
}

/**
 * Export words as plain text (newline-separated)
 */
function exportPlainText(words: string[]): void {
  const text = words.join('\n');
  navigator.clipboard.writeText(text);
}

/**
 * Export words as numbered list
 */
function exportNumberedList(words: string[]): void {
  const text = words.map((w, idx) => `${idx + 1}. ${w}`).join('\n');
  navigator.clipboard.writeText(text);
}

/**
 * Export groups as delimiter-separated (cat / tat)
 */
function exportGroupsPlainText(data: MinimalPair[] | ContrastiveGroup[], dataType: string): void {
  let text = '';

  if (dataType === 'pairs') {
    const pairs = data as MinimalPair[];
    text = pairs.map(p => `${p.word1.word} / ${p.word2.word}`).join('\n');
  } else {
    const groups = data as ContrastiveGroup[];
    text = groups.map(g => g.words.map(w => w.word.word).join(' / ')).join('\n');
  }

  navigator.clipboard.writeText(text);
}

/**
 * Download CSV for single words
 */
function downloadWordsCSV(words: Word[], filename: string): void {
  const headers = [
    'Word', 'IPA', 'Syllables', 'Phonemes', 'WCM',
    'Frequency', 'AoA', 'Imageability', 'Familiarity', 'Concreteness',
    'Valence', 'Arousal', 'Dominance'
  ];

  const rows = words.map(w => [
    w.word || '',
    w.ipa || '',
    w.syllable_count?.toString() || '',
    w.phoneme_count?.toString() || '',
    w.wcm_score?.toFixed(2) || '',
    w.frequency?.toFixed(1) || '',
    w.aoa?.toFixed(1) || '',
    w.imageability?.toFixed(1) || '',
    w.familiarity?.toFixed(1) || '',
    w.concreteness?.toFixed(1) || '',
    w.valence?.toFixed(1) || '',
    w.arousal?.toFixed(1) || '',
    w.dominance?.toFixed(1) || '',
  ]);

  downloadCSV(headers, rows, filename);
}

/**
 * Download CSV for pairs (long format with Group column)
 */
function downloadPairsCSV(pairs: MinimalPair[], filename: string): void {
  const headers = [
    'Group', 'Word', 'IPA', 'Phoneme', 'Position',
    'Syllables', 'Phonemes', 'WCM',
    'Frequency', 'AoA', 'Imageability', 'Familiarity', 'Concreteness',
    'Valence', 'Arousal', 'Dominance'
  ];

  const rows = pairs.flatMap((pair, idx) => [
    // Word 1
    [
      (idx + 1).toString(),
      pair.word1.word || '',
      pair.word1.ipa || '',
      pair.phoneme1 || '',
      pair.position?.toString() || '',
      pair.word1.syllable_count?.toString() || '',
      pair.word1.phoneme_count?.toString() || '',
      pair.word1.wcm_score?.toFixed(2) || '',
      pair.word1.frequency?.toFixed(1) || '',
      pair.word1.aoa?.toFixed(1) || '',
      pair.word1.imageability?.toFixed(1) || '',
      pair.word1.familiarity?.toFixed(1) || '',
      pair.word1.concreteness?.toFixed(1) || '',
      pair.word1.valence?.toFixed(1) || '',
      pair.word1.arousal?.toFixed(1) || '',
      pair.word1.dominance?.toFixed(1) || '',
    ],
    // Word 2
    [
      (idx + 1).toString(),
      pair.word2.word || '',
      pair.word2.ipa || '',
      pair.phoneme2 || '',
      pair.position?.toString() || '',
      pair.word2.syllable_count?.toString() || '',
      pair.word2.phoneme_count?.toString() || '',
      pair.word2.wcm_score?.toFixed(2) || '',
      pair.word2.frequency?.toFixed(1) || '',
      pair.word2.aoa?.toFixed(1) || '',
      pair.word2.imageability?.toFixed(1) || '',
      pair.word2.familiarity?.toFixed(1) || '',
      pair.word2.concreteness?.toFixed(1) || '',
      pair.word2.valence?.toFixed(1) || '',
      pair.word2.arousal?.toFixed(1) || '',
      pair.word2.dominance?.toFixed(1) || '',
    ],
  ]);

  downloadCSV(headers, rows, filename);
}

/**
 * Download CSV for groups (long format)
 */
function downloadGroupsCSV(groups: ContrastiveGroup[], filename: string): void {
  const headers = [
    'Group', 'Word', 'IPA', 'Phoneme', 'Position',
    'Syllables', 'Phonemes', 'WCM',
    'Frequency', 'AoA', 'Imageability', 'Familiarity', 'Concreteness',
    'Valence', 'Arousal', 'Dominance'
  ];

  const rows = groups.flatMap((group, groupIdx) =>
    group.words.map(w => [
      (groupIdx + 1).toString(),
      w.word.word || '',
      w.word.ipa || '',
      w.phoneme || '',
      w.position?.toString() || '',
      w.word.syllable_count?.toString() || '',
      w.word.phoneme_count?.toString() || '',
      w.word.wcm_score?.toFixed(2) || '',
      w.word.frequency?.toFixed(1) || '',
      w.word.aoa?.toFixed(1) || '',
      w.word.imageability?.toFixed(1) || '',
      w.word.familiarity?.toFixed(1) || '',
      w.word.concreteness?.toFixed(1) || '',
      w.word.valence?.toFixed(1) || '',
      w.word.arousal?.toFixed(1) || '',
      w.word.dominance?.toFixed(1) || '',
    ])
  );

  downloadCSV(headers, rows, filename);
}

/**
 * Generic CSV download helper
 */
function downloadCSV(headers: string[], rows: string[][], filename: string): void {
  const csv = [headers, ...rows]
    .map(row => row.map(cell => `"${cell}"`).join(','))
    .join('\n');

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// ============================================================================
// Component
// ============================================================================

const ExportMenu: React.FC<ExportMenuProps> = ({
  data,
  dataType,
  selectedIndices,
  filename = 'phonolex_export.csv',
  disabled = false,
}) => {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [copied, setCopied] = useState(false);

  const open = Boolean(anchorEl);

  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
    setCopied(false);
  };

  const handleCopyPlainText = () => {
    let dataToExport: ExportData = data;

    // Filter by selected indices if provided
    if (selectedIndices && selectedIndices.length > 0) {
      dataToExport = selectedIndices.map(idx => data[idx]) as ExportData;
    }

    if (dataType === 'words') {
      const words = extractWords(dataToExport, dataType);
      exportPlainText(words);
    } else {
      // For pairs/groups, use delimiter format
      exportGroupsPlainText(dataToExport as MinimalPair[] | ContrastiveGroup[], dataType);
    }

    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleCopyNumberedList = () => {
    let dataToExport: ExportData = data;

    if (selectedIndices && selectedIndices.length > 0) {
      dataToExport = selectedIndices.map(idx => data[idx]) as ExportData;
    }

    if (dataType === 'words') {
      const words = extractWords(dataToExport, dataType);
      exportNumberedList(words);
    } else {
      // For pairs/groups, use delimiter format with numbering
      exportGroupsPlainText(dataToExport as MinimalPair[] | ContrastiveGroup[], dataType);
    }

    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadCSV = () => {
    let dataToExport: ExportData = data;

    if (selectedIndices && selectedIndices.length > 0) {
      dataToExport = selectedIndices.map(idx => data[idx]) as ExportData;
    }

    if (dataType === 'words') {
      downloadWordsCSV(dataToExport as Word[], filename);
    } else if (dataType === 'pairs') {
      downloadPairsCSV(dataToExport as MinimalPair[], filename);
    } else {
      downloadGroupsCSV(dataToExport as ContrastiveGroup[], filename);
    }

    handleClose();
  };

  const hasSelection = selectedIndices && selectedIndices.length > 0;

  return (
    <>
      <Button
        size="small"
        startIcon={<DownloadIcon />}
        onClick={handleClick}
        disabled={disabled || data.length === 0}
        sx={{ minHeight: 44 }}
      >
        Export
      </Button>

      <Menu
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'right',
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
      >
        <MenuItem onClick={handleCopyPlainText}>
          <ListItemIcon>
            {copied ? <CheckIcon fontSize="small" color="success" /> : <CopyIcon fontSize="small" />}
          </ListItemIcon>
          <ListItemText>
            {dataType === 'words' ? 'Copy Words (Plain Text)' : 'Copy Groups (Delimited)'}
          </ListItemText>
        </MenuItem>

        {dataType === 'words' && (
          <MenuItem onClick={handleCopyNumberedList}>
            <ListItemIcon>
              <NumberedListIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText>Copy Numbered List</ListItemText>
          </MenuItem>
        )}

        <Divider />

        <MenuItem onClick={handleDownloadCSV}>
          <ListItemIcon>
            <DownloadIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>
            {hasSelection ? `Download CSV (${selectedIndices!.length} selected)` : 'Download CSV (All Data)'}
          </ListItemText>
        </MenuItem>
      </Menu>
    </>
  );
};

export default ExportMenu;
