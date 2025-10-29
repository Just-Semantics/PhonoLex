# PhonoLex Web Interface

A simple client-side React application for loading and verifying the PhonoLex phonological lexicon data.

## Getting Started

### Prerequisites

- Node.js (v14+)
- npm or yarn

### Installation

```bash
# Install dependencies
npm install
```

### Running the Application

```bash
# Start the development server
npm start
```

This will:
1. Run the setup script to copy data files to the public directory
2. Start the React development server
3. Open the application in your browser at http://localhost:3000

When the app loads, it will attempt to load the phoneme and word data from the processed data files. If successful, it will display statistics about the loaded data:

- **English Phonemes** - The 38 distinct phonemes used in the English language
- **Words (Gold Standard)** - The ~52k words in the high-quality dataset
- **Phoneme Features** - The 37 phonological features used to describe each phoneme

### Running Tests

```bash
# Run tests
npm test
```

The tests verify that the data loading functionality works correctly.

## Data Files

This application uses the following data files from the PhonoLex dataset:

- `phoneme_vectors.json` - Vector representations of the 38 English phonemes (filtered from 2100+ phonemes)
- `phoneme_features.json` - Phonological features for each English phoneme
- `gold_standard.json` - Reference dataset of ~52k high-quality words

These files are automatically copied from the `data/processed` directory when you run the application. The phoneme files have been filtered by the data processing pipeline to include only the phonemes supported by PhonoLex for English, reducing the data size by over 98%.

## Data Filtering Process

The phoneme data has been filtered to include only the phonemes that are supported in the English IPA to ARPA mappings. This filtering happens in the main data processing pipeline with the `filter_supported_phonemes.py` script, which:

1. Reads the IPA to ARPA mapping to identify supported phonemes
2. Filters the phoneme vectors and features files to only include those phonemes
3. Creates backups of the original full files
4. Results in much smaller, more efficient data files (22KB vs ~1MB)

## Next Steps

This minimalist app confirms that the data is loading correctly. The next steps would be to build UI components for:

- Visualizing phoneme relationships
- Searching the lexicon
- Comparing phonological properties
- Testing phoneme transformations 

## Features

The PhonoLex web interface now includes several interactive components:

### 1. Phoneme Explorer
- Select individual phonemes to view their feature specifications
- See color-coded feature values based on strength
- View the phoneme's vector representation

### 2. Word Search
- Search for words in the lexicon
- View pronunciation, part of speech, syllable structure, and stress patterns
- Access detailed phonological information for matching words

### 3. Phoneme Visualization
- Interactive 2D/3D visualization of phoneme vectors
- Multiple ways to explore phonological relationships:
  - **Multiselect Category Filter**: Select multiple phoneme categories simultaneously
  - **Dimension Viewing**: Toggle between 2D and 3D visualization modes
  - **Quick Filters**: One-click filters for "Vowels Only" or "Consonants Only"
  - **Visual Encoding**: Color-coded points based on phoneme category
  - **Interactive Selection**: Click on phonemes to view detailed feature information

The phoneme visualization uses Principal Component Analysis (PCA) to project high-dimensional phoneme vectors into a 2D or 3D space, placing similar phonemes closer together. This provides an intuitive way to explore relationships between different sound categories. 