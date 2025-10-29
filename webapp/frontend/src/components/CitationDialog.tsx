/**
 * Citation Dialog Component
 *
 * Displays detailed citations for specific measurement categories
 * Organized by psycholinguistic properties and data sources
 */

import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Divider,
  IconButton,
  Link,
} from '@mui/material';
import {
  Close as CloseIcon,
  OpenInNew as ExternalIcon,
} from '@mui/icons-material';

interface Citation {
  authors: string;
  year: string;
  title: string;
  journal?: string;
  volume?: string;
  issue?: string;
  pages?: string;
  publisher?: string;
  doi?: string;
  url?: string;
}

interface CitationDialogProps {
  open: boolean;
  onClose: () => void;
  category: 'phonological' | 'lexical' | 'semantic' | 'affective' | 'embeddings' | 'data-sources' | null;
}

const citations: Record<string, Citation[]> = {
  phonological: [
    {
      authors: 'Moran, S., & McCloy, D.',
      year: '2019',
      title: 'PHOIBLE 2.0',
      publisher: 'Max Planck Institute for the Science of Human History',
      url: 'https://phoible.org/',
    },
    {
      authors: 'Hayes, B.',
      year: '2009',
      title: 'Introductory Phonology',
      publisher: 'Wiley-Blackwell',
    },
    {
      authors: 'Moisik, S. R., & Esling, J. H.',
      year: '2011',
      title: 'The \'whole larynx\' approach to laryngeal features',
      journal: 'Proceedings of the International Congress of Phonetic Sciences XVII',
      pages: '1406-1409',
    },
    {
      authors: 'Stoel-Gammon, C.',
      year: '2010',
      title: 'The Word Complexity Measure: Description and application to developmental phonology and disorders',
      journal: 'Clinical Linguistics & Phonetics',
      volume: '24',
      issue: '4-5',
      pages: '271-282',
      doi: '10.3109/02699200903581059',
    },
  ],
  lexical: [
    {
      authors: 'Brysbaert, M., & New, B.',
      year: '2009',
      title: 'Moving beyond Kučera and Francis: A critical evaluation of current word frequency norms and the introduction of a new and improved word frequency measure for American English',
      journal: 'Behavior Research Methods',
      volume: '41',
      issue: '4',
      pages: '977-990',
      doi: '10.3758/BRM.41.4.977',
    },
    {
      authors: 'Kuperman, V., Stadthagen-Gonzalez, H., & Brysbaert, M.',
      year: '2012',
      title: 'Age-of-acquisition ratings for 30,000 English words',
      journal: 'Behavior Research Methods',
      volume: '44',
      issue: '4',
      pages: '978-990',
      doi: '10.3758/s13428-012-0210-4',
    },
  ],
  semantic: [
    {
      authors: 'Brysbaert, M., Warriner, A. B., & Kuperman, V.',
      year: '2014',
      title: 'Concreteness ratings for 40 thousand generally known English word lemmas',
      journal: 'Behavior Research Methods',
      volume: '46',
      issue: '3',
      pages: '904-911',
      doi: '10.3758/s13428-013-0403-5',
    },
    {
      authors: 'Cortese, M. J., & Fugett, A.',
      year: '2004',
      title: 'Imageability ratings for 3,000 monosyllabic words',
      journal: 'Behavior Research Methods, Instruments, & Computers',
      volume: '36',
      issue: '3',
      pages: '384-387',
      doi: '10.3758/BF03195585',
    },
  ],
  affective: [
    {
      authors: 'Warriner, A. B., Kuperman, V., & Brysbaert, M.',
      year: '2013',
      title: 'Norms of valence, arousal, and dominance for 13,915 English lemmas',
      journal: 'Behavior Research Methods',
      volume: '45',
      issue: '4',
      pages: '1191-1207',
      doi: '10.3758/s13428-012-0314-x',
    },
  ],
  embeddings: [
    {
      authors: 'Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K.',
      year: '2019',
      title: 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
      journal: 'Proceedings of NAACL-HLT 2019',
      pages: '4171-4186',
      doi: '10.18653/v1/N19-1423',
    },
  ],
  'data-sources': [
    {
      authors: 'Carnegie Mellon University',
      year: '2014',
      title: 'The CMU Pronouncing Dictionary (125,764 words)',
      url: 'http://www.speech.cs.cmu.edu/cgi-bin/cmudict',
    },
    {
      authors: 'Open Dict Data',
      year: '2020',
      title: 'ipa-dict: Monolingual wordlists with pronunciation information in IPA',
      url: 'https://github.com/open-dict-data/ipa-dict',
    },
    {
      authors: 'Moran, S., & McCloy, D.',
      year: '2019',
      title: 'PHOIBLE 2.0 (2,716 languages, 38 distinctive features)',
      publisher: 'Max Planck Institute for the Science of Human History',
      url: 'https://phoible.org/',
    },
    {
      authors: 'Brysbaert, M., & New, B.',
      year: '2009',
      title: 'SUBTLEX-US: American English word frequencies from film subtitles',
      journal: 'Behavior Research Methods',
      volume: '41',
      issue: '4',
      pages: '977-990',
      doi: '10.3758/BRM.41.4.977',
    },
  ],
};

const categoryTitles: Record<string, string> = {
  phonological: 'Phonological Complexity',
  lexical: 'Lexical Properties',
  semantic: 'Semantic Properties',
  affective: 'Affective/Emotional Norms',
  embeddings: 'Embedding Architecture',
  'data-sources': 'Data Sources',
};

const categoryDescriptions: Record<string, string> = {
  phonological: 'Measurements of phonological and articulatory complexity',
  lexical: 'Word frequency and age of acquisition measures',
  semantic: 'Imageability, familiarity, and concreteness ratings',
  affective: 'Emotional valence, arousal, and dominance norms',
  embeddings: 'Hierarchical syllable embedding architecture',
  'data-sources': 'Primary datasets and corpora',
};

const CitationDialog: React.FC<CitationDialogProps> = ({ open, onClose, category }) => {
  if (!category) return null;

  const citationList = citations[category] || [];

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      scroll="paper"
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <Box>
            <Typography variant="h6" fontWeight={600}>
              {categoryTitles[category]}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {categoryDescriptions[category]}
            </Typography>
          </Box>
          <IconButton onClick={onClose} size="small" aria-label="close">
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>
      <DialogContent dividers>
        <Box sx={{ '& > *:not(:last-child)': { mb: 3 } }}>
          {citationList.map((citation, index) => (
            <Box key={index}>
              <Typography variant="body1" paragraph sx={{ mb: 1 }}>
                <strong>{citation.authors}</strong> ({citation.year}).{' '}
                {citation.title}.
                {citation.journal && (
                  <>
                    {' '}<em>{citation.journal}</em>
                    {citation.volume && `, ${citation.volume}`}
                    {citation.issue && `(${citation.issue})`}
                    {citation.pages && `, ${citation.pages}`}.
                  </>
                )}
                {citation.publisher && ` ${citation.publisher}.`}
              </Typography>
              {citation.doi && (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
                  <Typography variant="caption" color="text.secondary">
                    DOI:
                  </Typography>
                  <Link
                    href={`https://doi.org/${citation.doi}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    variant="caption"
                    sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}
                  >
                    {citation.doi}
                    <ExternalIcon sx={{ fontSize: '0.875rem' }} />
                  </Link>
                </Box>
              )}
              {citation.url && (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Typography variant="caption" color="text.secondary">
                    URL:
                  </Typography>
                  <Link
                    href={citation.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    variant="caption"
                    sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}
                  >
                    {citation.url}
                    <ExternalIcon sx={{ fontSize: '0.875rem' }} />
                  </Link>
                </Box>
              )}
              {index < citationList.length - 1 && <Divider sx={{ mt: 2 }} />}
            </Box>
          ))}
        </Box>
      </DialogContent>
      <DialogActions sx={{ px: 3, py: 2 }}>
        <Button onClick={onClose} variant="contained">
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default CitationDialog;
