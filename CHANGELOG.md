# Changelog

All notable changes to PhonoLex will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
