# Changelog

All notable changes to PhonoLex will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.3.0-beta] - 2025-11-06

### Added
- **🎯 Phoneme-Sequence Soft Levenshtein Architecture**
  - Revolutionary approach: NO AVERAGING - preserves sequential structure within onset/nucleus/coda
  - Onset/coda: sequences of 76-dim phoneme vectors compared with soft Levenshtein distance
  - Nucleus: 152-dim for diphthongs (trajectory), 76-dim for monophthongs
  - Three-level hierarchy: phoneme → component → syllable → word
  - User-adjustable onset/nucleus/coda weights at query time
  - Complete documentation: `docs/PHONEME_SEQUENCE_ARCHITECTURE_V2.3.md`

### Fixed
- **🐛 Critical: Component-wise averaging bug (v2.2.1)**
  - v2.2.1 averaged consonant clusters: `[k, ɹ]` → single vector (destroyed structure!)
  - v2.2.1 averaged diphthongs: trajectory collapsed to single vector
  - v2.3 preserves sequences: `[k, ɹ]` → sequence of 2 vectors
  - **Result**: Proper discrimination of consonant clusters and complex syllables
- **📊 Improved cat-crest discrimination**
  - Old (v2.2.1 averaging): cat-crest = 0.20 (too similar due to averaging)
  - New (v2.3 sequences): cat-crest = 0.74 (correct - crest has extra phonemes)
  - Onset comparison: `[k]` vs `[k, ɹ]` → 0.50 (properly penalizes length difference)
  - Coda comparison: `[t]` vs `[s, t]` → 0.90 (soft match on shared /t/)

### Changed
- **Syllable Structure Data Format**
  - Each component now stores sequences of phoneme vectors (not averaged embeddings)
  - `onset: number[][]` - sequence of 76-dim vectors
  - `nucleus: number[][]` - sequence of 76-dim or 152-dim vectors
  - `coda: number[][]` - sequence of 76-dim vectors
- **Similarity Computation**
  - Component-level: Soft Levenshtein DP on phoneme sequences
  - Syllable-level: Weighted average of component similarities
  - Word-level: Soft Levenshtein on syllable sequences
- **Build Pipeline**
  - New script: `scripts/build_phase3_syllable_embeddings_v2.py`
  - New export: `scripts/export_clientside_data_v2.py`
  - New frontend service: `webapp/frontend/src/services/clientSideData_v2.ts`

### Performance
- **🚀 File Size**: 99% compression ratio
  - Uncompressed: 56.7 MB JSON
  - Gzipped: 0.6 MB (99% compression!)
  - Total download: ~1.4 MB (structures + metadata)
- **Query Performance**
  - Load time: ~500ms (one-time, cached)
  - Full-vocab similarity scan: ~50-100ms (17,920 words)
  - Single comparison: ~0.005ms
  - Phoneme sequence DP: ~0.001ms (typical onset/coda length)

### Examples
- **cat** vs **crest**: 0.74 (onset [k] vs [k,ɹ], coda [t] vs [s,t])
- **cat** vs **bat**: 0.90 (rhyme, different onset)
- **cat** vs **act**: 0.20 (anagrams, different syllable structure)

### Technical Details
- Soft Levenshtein: Standard edit distance DP with soft substitution costs
- Match cost: `1.0 - cosineSimilarity(phoneme1, phoneme2)`
- Normalization: `similarity = 1.0 - (editDistance / maxLength)`
- Symmetric: dist(A, B) = dist(B, A)
- Length-sensitive: Longer sequences properly penalized
- Bounded: [0, 1] range (0 = unrelated, 1 = identical)

### Deprecated
- **⚠️ v2.2.1 Component-wise averaging approach is DEPRECATED**
  - Do not use `build_phase3_syllable_embeddings.py` (old version)
  - Do not use `export_clientside_data.py` (old version)
  - Averaging destroyed sequential structure (consonant clusters, diphthongs)
  - Use v2.3 scripts instead (with `_v2` suffix)

## [2.2.1-beta] - 2025-01-06 **[DEPRECATED]**

### ⚠️ DEPRECATION WARNING
This version has a critical bug where component-wise averaging destroys sequential structure. DO NOT USE. Replaced by v2.3.0.

### Changed
- **🏗️ Rebranded "Layers" to "Phases" (3-Phase Pipeline)**
  - Renamed architecture from confusing "4-layer" to clear "3-phase" deterministic pipeline
  - **Phase 1** (Extract): Phoible features (38-dim ternary) - `embeddings/phase1/`
  - **Phase 2** (Normalize): Continuous vectors (76-dim) - `embeddings/phase2/`
  - **Phase 3** (Aggregate): Syllable embeddings (228-dim) - `embeddings/phase3/`
  - Removed "Layer 3" (training step - no longer exists)
  - Removed "Layer 4" (was confusing - now Phase 3)
  - Scripts renamed: `compute_phase1/2_*.py`, `build_phase3_*.py`
  - **Rationale**: "Phases" better describes deterministic transformations; "Layers" implied ML model layers
  - All documentation updated (CLAUDE.md, architecture docs, README)
  - Archived deprecated Layer 3 training scripts to `archive/deprecated_layer_scripts/`

- **🎯 Major Architecture Shift: Pure Phoible Embeddings**
  - Replaced MLM-trained embeddings with pure Phoible phonological feature vectors
  - No training required - embeddings built directly from linguistic theory
  - Reduced file size: 23.8 MB → 1.5 MB gzipped (94% reduction!)
  - Reduced dimensions: 384-dim → 228-dim per syllable (40% reduction)
  - **User-adjustable weights**: Onset/nucleus/coda weights now controllable at query time
  - Linguistically transparent: Based on 38 universal phonological features from Phoible

### Fixed
- **Unicode Normalization**: Fixed IPA character mismatch for /g/ phoneme
  - Updated `data/mappings/arpa_to_ipa.json` to use correct IPA character `ɡ` (U+0261) instead of ASCII `g` (U+0067)
  - Fixes "dog", "fog", and other words with /g/ failing to build Phoible embeddings
  - All 17,920 filtered words now have complete Phoible embeddings

### Improved
- **Better Onset Discrimination**: Fixed rhyme bias in general similarity
  - Old: make-take (0.79) scored higher than make-bake (0.83) ❌
  - New: make-bake (0.95) correctly scores higher than make-take (0.85) ✅
  - Onset similarity now properly considered: /m/-/b/ (0.84) > /m/-/t/ (0.55)
  - Component-wise normalization ensures equal weighting of onset, nucleus, and coda

### Removed
- **Quantization**: No longer needed with Phoible's excellent compressibility
  - Direct float32 embeddings compress 99% with gzip
  - Removed int8 quantization and dequantization logic
  - Simpler data pipeline: build → export → load

### Technical Details
- Syllable structure: onset(76) + nucleus(76) + coda(76) = 228 dims
- Each component individually normalized to unit length
- Default weighting: 1/3 each (onset, nucleus, coda)
- Users can adjust weights for specific use cases (rhymes, alliteration, etc.)

### Documentation
- Added `docs/PHASE_ARCHITECTURE.md`
  - Complete guide to the new 3-phase pipeline
  - Detailed explanation of each phase (Extract → Normalize → Aggregate)
  - Command reference and rebuild instructions
  - Comparison: Phases vs old Layer architecture
  - Migration guide from layers to phases
- Added `docs/PHOIBLE_EMBEDDINGS_AND_WEIGHTED_SIMILARITY.md`
  - Complete guide to Phoible embeddings and weighted similarity
  - UI implementation guide with code examples
  - Use case examples (rhymes, alliteration, assonance, etc.)
  - Comparison: Phoible vs MLM embeddings
- Added `docs/MIGRATION_SUMMARY_V2.2.1.md`
  - Comprehensive migration notes from v2.2.0 to v2.2.1
  - Technical changes, file size comparisons, performance metrics

### Migration Notes
- **Breaking**: Embedding dimension changed from 384 → 228
- **Breaking**: File name changed from `embeddings_quantized.json.gz` → `embeddings.json.gz`
- **Breaking**: Removed `scales` property (no quantization)
- Frontend automatically loads new format - no API changes

### Next Steps (UI Implementation)
- [ ] Add onset/nucleus/coda weight sliders to similarity search
- [ ] Add preset buttons (Balanced, Rhymes, Alliteration, etc.)
- [ ] Update all tools to use weighted similarity
- [ ] Save user weight preferences

## [2.2.0-beta] - 2025-11-06

### Added
- **Multiple Opposition Intervention**: Implemented third contrastive intervention mode based on Storkel (2022) research
  - Representative target selection using Maximal Classification + Maximal Distinction algorithms
  - Automatic generation of minimal sets (triplets/quadruplets/quintuplets)
  - Clinical validation support for global phoneme collapse treatment (severe SSD)
  - Duplicate phoneme detection to ensure valid minimal sets
  - Example: t→d,k generates "bat - bad - back", "top - drop - cop"
- **Unified Contrastive Intervention Tool**: Consolidated three intervention approaches into one tool
  - Mode toggle: Minimal Pairs | Maximal Opposition | Multiple Opposition
  - Responsive design: ToggleButtonGroup (desktop) / Select dropdown (mobile)
  - Clinical context alerts explaining when to use each mode
  - Consistent input patterns across all three modes
- **Comprehensive Test Suite**: 35 passing tests covering all Multiple Opposition algorithms
  - Unit tests for representative target selection (Maximal Distinction)
  - Integration tests for minimal set generation with real phoneme data
  - API adapter tests for all public methods
  - Edge case handling (phonotactic constraints, duplicate detection)
- **Example Verification Script**: Added `scripts/verify_examples.py` to programmatically test all documentation examples
  - Validates all 8 Custom Word Lists examples against actual word data
  - Ensures documentation accuracy by checking expected keywords appear in results

### Changed
- **Unified Tool Architecture**: Replaced separate MinimalPairsTool and MaximalOppositionTool with ContrastiveInterventionTool
  - Single entry point with mode selection following SearchTool pattern
  - Shared position selector and phoneme picker across all modes
  - Consistent validation and error messaging
- **Improved Example Placeholders**: Updated Multiple Opposition mode to use t→d,k example (works in all positions)
- **TypeScript Type Safety**: Fixed 3 `any` type warnings in ContrastiveInterventionTool

### Improved
- **🎯 Logarithmic Frequency Slider**: Fixed severe usability issue with frequency filtering
  - Old: Linear slider made low frequencies (0-100) impossible to select - first tick at 1,300+
  - New: Log10 scale with fine control at low end while preserving full range (0-2.1M)
  - Added frequency marks: 0, 10, 100, 1K, 10K for intuitive reference
  - Displays actual frequency values (not log values) for user clarity
- **Unified Exclusion Field UX**: Made phoneme exclusions consistent with pattern fields
  - Removed "Add" button requirement - now space-separated like all other phoneme inputs
  - Removed chip display for consistency
  - Auto-spacing when selecting multiple phonemes from IPA keyboard
  - Same placeholder text as pattern fields: "Use keyboard icon → to select IPA"

### Fixed
- **🐛 Critical: Concreteness Data Loading Bug**
  - Was reading wrong column (Bigram instead of Conc.M) from psycholinguistic norms
  - Fixed column index from row[1] to row[2] in `english_data_loader.py`
  - Coverage increased from 0% to 97.8% (24,189 words now have concreteness values)
  - Re-exported all client-side data with corrected values
- Removed unused `beforeAll` import from test files (ESLint error)
- Fixed duplicate phonemes appearing in Multiple Opposition sets
- Added proper type annotations for Select onChange handlers

### Documentation
- **Updated Examples for Accuracy**: Fixed 4 examples to reflect actual app behavior
  - Example 3: Updated expected words (hear, hair, air, fair, bear, chair, fear)
  - Example 4: Updated for negative valence thresholds (afraid, scared, mad, evil, dangerous, attack)
  - Example 16: Updated for concreteness after bug fix (justice, theory, particular, professional, affair, destiny, former, instance)
  - Example 17: Updated for neutral valence/arousal (table, paper, floor, time, thing, work, wait, put)
- All 8 Custom Word Lists examples verified with real data (✅ passing)

### Research
- Implemented algorithms from Storkel (2022): "Minimal, maximal, or multiple: Which contrastive intervention approach to use with children with speech sound disorder?"
- Validated Maximal Classification for target diversity
- Validated Maximal Distinction for phonological distance maximization

### Code Quality
- **Python Formatting**: Applied Black formatting to 24 files (100% compliance)
  - All scripts and source files now follow consistent PEP 8 style
  - Automated formatting for maintainability
- **Python Linting**: Fixed all Ruff linting issues
  - Auto-fixed 49 style violations
  - Removed unused variable in `export_phoneme_data.py`
  - Added proper `noqa` directives for intentional late imports (E402)
- **Frontend Quality**: All checks passing
  - TypeScript type checking: ✅ No errors
  - ESLint: ✅ 5 warnings (within limit of 50)
  - Production build: ✅ 672.86 kB → 199.17 kB gzipped

### Technical
- All tests passing (35/35)
- Zero breaking changes to existing APIs
- All example verification tests passing (8/8)

## [2.1.1-beta] - 2025-11-04

### Improved
- **IPA Keyboard Discoverability**: Updated all phoneme input placeholders to guide users to the keyboard icon
  - Changed from generic examples ("e.g., t, k, s") to action-oriented prompts ("Use keyboard icon → to select IPA")
  - Added missing IPA keyboard button to Search component's phoneme search mode
  - All 5 components with phoneme inputs now have consistent, discoverable keyboard access
  - Components updated: MinimalPairsTool, Compare, Builder, SearchTool, Search

### Fixed
- Added missing PhonemePickerDialog to Search.tsx phoneme search mode
- Added IPA validation and warning alerts to Search.tsx

## [Unreleased] - 2025-11-01

### Added
- Position filtering for Minimal Pairs (word-initial, medial, final) - clinically relevant for SLPs
- Auto-include typed exclusions in Custom Word List Builder (no "Add" button required)
- Comprehensive testing infrastructure:
  - **Vitest** unit tests for phoneme tokenization, exclusion filtering, Unicode handling (17 tests)
  - **Playwright** E2E tests for Builder exclusions and Minimal Pairs position filtering
  - Test coverage reporting and UI test runners
  - CI integration with automated test runs on push/PR

### Changed
- Simplified Minimal Pairs tool: removed property filters (syllables, WCM, frequency)
- Unified UI paradigm: all input fields now "type and go" without explicit add/submit buttons
- Updated CI workflow to run unit tests and E2E tests before build

### Removed
- Norm-Filtered Lists tool (functionality fully covered by Custom Word List Builder)

### Fixed
- Exclusion filtering bug where typed exclusions weren't applied without clicking Add button
- Unicode phoneme handling in exclusion filters (dʒ, ð, θ, etc.)

## [2.1.0-beta] - 2025-10-31

### Added
- Fully client-side architecture - no backend required
- Static JSON data files for all word embeddings and metadata (~88 MB, gzips to ~45 MB)
- Mobile-optimized responsive design
- Sticky table headers (vertical scroll) and sticky Word column (horizontal scroll)
- Scroll hint indicator for mobile table view
- Phoneme picker keyboard for easy IPA input
- Placeholder text for all phoneme input fields (replaces default values)
- Enhanced card view with hover effects and improved visual hierarchy
- Touch-friendly UI with 44x44px minimum touch targets (WCAG Level AAA)

### Changed
- **Breaking**: Migrated from FastAPI + PostgreSQL backend to client-side-only React app
- Improved mobile layout with responsive typography and spacing
- Updated header to show "Phono" on mobile, "PhonoLex" on desktop
- Optimized all form controls to stack vertically on mobile
- Enhanced table view with horizontal scrolling and sticky columns
- IPA column now displays on single line with `whiteSpace: nowrap`
- All phoneme input fields now start empty with helpful placeholder text

### Fixed
- Fixed phoneme comparison crash: `similarity_score` undefined error
- Fixed data structure mismatch in `comparePhonemes()` function
- Fixed minimal pairs data structure to match TypeScript interface
- Added null checks for all `.toFixed()` calls throughout the application
- Fixed Word column and IPA column wrapping issues in table view

### Technical
- Type-safe client-side data service with full TypeScript coverage
- Optimized bundle size: 641 KB minified → 190 KB gzipped
- Zero server costs, zero database maintenance
- Offline-capable architecture (PWA-ready)
- All components pass TypeScript strict mode checks

### Performance
- Instant data loading (no network latency)
- Sub-millisecond query response times
- Efficient in-memory filtering and pattern matching
- Smooth 60fps scrolling with hardware acceleration

## [2.0.0] - 2025-10-01

### Added
- FastAPI backend with PostgreSQL + pgvector
- Database-backed word storage with vector similarity search
- RESTful API endpoints for all operations
- Server-side filtering and pattern matching

### Changed
- Migrated from Flask (v1) to FastAPI (v2)
- New database schema with psycholinguistic properties
- Improved API design with Pydantic validation

## [1.0.0] - 2024-06-01

### Added
- Initial release with Flask backend
- Basic phonological tools (minimal pairs, rhyme sets)
- CMU Dictionary integration
- Simple web interface

---

**Note**: v2.0.0 backend code archived in `archive/webapp_v2_backend/` (October 2025)
