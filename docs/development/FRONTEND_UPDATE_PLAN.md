# Frontend Update Plan - V2.0 Data Integration

## Overview
This document tracks the work needed to update the frontend to use the v2.0 backend with full data exposure.

## Current Status
- **Backend**: v2.0 database-centric architecture with PostgreSQL + pgvector
- **Frontend**: Using outdated types and API client pointing to partial v1.0 endpoints
- **Gap**: Frontend not exposing ~50% of available word data and 100% of edge/graph data

## Required Updates

### 1. TypeScript Type Definitions ✅ (To Do)
**File**: `webapp/frontend/src/types/phonology.ts`

#### Add Missing Word Properties
- `msh_stage` (Motor Speech Hierarchy stage)
- `aoa` (Age of Acquisition)
- `imageability`
- `familiarity`
- `valence`
- `arousal`
- `dominance`

#### Add Edge Types
```typescript
export interface WordEdge {
  word1: string;
  word2: string;
  relation_type: 'MINIMAL_PAIR' | 'RHYME' | 'NEIGHBOR' | 'MAXIMAL_OPP' | 'SIMILAR' | 'MORPHOLOGICAL';
  metadata: EdgeMetadata;
  weight: number;
}

export interface MinimalPairMetadata {
  position: number;
  phoneme1: string;
  phoneme2: string;
  feature_diff?: number;
}

export interface RhymeMetadata {
  rhyme_type: 'perfect' | 'slant' | 'assonance';
  nucleus: string;
  coda: string[];
  quality: number;
}

export type EdgeMetadata = MinimalPairMetadata | RhymeMetadata | Record<string, any>;
```

#### Add Phoneme Types
```typescript
export interface PhonemeDetail {
  phoneme_id: number;
  ipa: string;
  segment_class: 'consonant' | 'vowel' | 'tone';
  features: Record<string, string>;
  has_trajectory: boolean;
  trajectory_features?: string[];
}
```

### 2. API Client Updates ✅ (To Do)
**File**: `webapp/frontend/src/services/phonolexApi.ts`

#### Update Word Interface
Match backend `WordResponse` schema exactly

#### Add Missing Endpoints
- `GET /api/graph/neighbors/{word}`
- `GET /api/graph/minimal-pairs?phoneme1={p1}&phoneme2={p2}`
- `GET /api/graph/rhymes/{word}`
- `GET /api/graph/export`
- `POST /api/words/filter`
- `POST /api/words/pattern-search`
- `POST /api/similarity/search`

#### Update Existing Endpoints
- Change base paths from v1.0 to v2.0 format
- Update response types to match v2.0 schemas

### 3. Component Updates (To Do Later)
Once types and API are updated, update UI components to display:
- Psycholinguistic properties in word cards
- Edge relationships in graph visualizations
- MSH stage for clinical users
- Age of acquisition for educational contexts

## Implementation Order

1. **Phase 1: Type Definitions** (30 min)
   - Update `webapp/frontend/src/types/phonology.ts`
   - Add comprehensive Word, Edge, and Phoneme types
   - Ensure 100% backend schema coverage

2. **Phase 2: API Client** (1 hour)
   - Update `webapp/frontend/src/services/phonolexApi.ts`
   - Add all v2.0 endpoints
   - Update method signatures and return types
   - Test API calls

3. **Phase 3: Validation** (30 min)
   - Type-check frontend: `npm run type-check`
   - Test API integration
   - Verify data flow from backend to frontend

4. **Phase 4: UI Updates** (Future)
   - Update components to display new data
   - Add visualization for edge types
   - Enhance word detail views

## Testing Checklist

- [ ] Frontend compiles without TypeScript errors
- [ ] API client methods return correct types
- [ ] Word data includes all psycholinguistic properties
- [ ] Edge data includes relation types and metadata
- [ ] Similarity search returns complete word objects
- [ ] Graph endpoints return neighbor/rhyme/minimal pair data

## Migration Notes

### Breaking Changes
- `Word` interface has 9 new required properties
- All API methods now return richer data structures
- Existing components may need null checks for new properties

### Backward Compatibility
- New properties are nullable in TypeScript for gradual migration
- Old endpoints still work (for now) but are deprecated
- Frontend should gracefully handle missing data from v1.0

## Files to Update

1. `webapp/frontend/src/types/phonology.ts` - Type definitions
2. `webapp/frontend/src/services/phonolexApi.ts` - API client
3. (Later) UI components that display word data

## Success Criteria

✅ Frontend types match backend schemas 100%
✅ All v2.0 endpoints accessible from frontend
✅ Word results include all available properties
✅ Edge/graph data properly typed and accessible
✅ Type checking passes
✅ No runtime errors when fetching data
