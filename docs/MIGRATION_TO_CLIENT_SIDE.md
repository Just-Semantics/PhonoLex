# Migration Guide: Backend API → Client-Side Data

This guide explains how to migrate the PhonoLex frontend from using the backend API to the new fully client-side data paradigm.

## Why Migrate?

✅ **No backend server** - Deploy as static site
✅ **No database** - Everything runs in browser
✅ **No server costs** - Free hosting on Netlify/Cloudflare Pages
✅ **Faster** - No network latency for queries
✅ **Offline-capable** - Progressive Web App potential

## What's Changed

### Before (v2.0 with Backend)
```typescript
import { api } from './services/phonolexApi';

// Make HTTP request to backend
const words = await api.findMinimalPairs('t', 'd', { limit: 10 });
```

### After (v2.1 Client-Side)
```typescript
import { clientSideData } from './services/clientSideData';

// Load data once on app startup
await clientSideData.loadData();

// Compute locally in browser
const words = await clientSideData.findMinimalPairs('t', 'd', 10);
```

## Migration Steps

### 1. Update App Initialization

Add data loading to your app startup:

```typescript
// src/App.tsx or src/main.tsx

import { clientSideData } from './services/clientSideData';
import { useState, useEffect } from 'react';

function App() {
  const [dataLoaded, setDataLoaded] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    // Load data on app mount
    clientSideData.loadData()
      .then(() => {
        console.log('✓ PhonoLex data loaded!');
        setDataLoaded(true);
      })
      .catch((err) => {
        console.error('Failed to load data:', err);
        setLoadError(err.message);
      });
  }, []);

  if (loadError) {
    return <div>Error loading data: {loadError}</div>;
  }

  if (!dataLoaded) {
    return <div>Loading PhonoLex data...</div>;
  }

  return <YourApp />;
}
```

### 2. Replace API Calls

Find all uses of `phonolexApi` and replace with `clientSideData`:

**Before:**
```typescript
import { api } from '../services/phonolexApi';

const results = await api.findMinimalPairs('t', 'd', {
  word_length: 'short',
  complexity: 'low',
  limit: 5
});
```

**After:**
```typescript
import { clientSideData } from '../services/clientSideData';

const results = await clientSideData.findMinimalPairs('t', 'd', 5);
// Note: Client-side doesn't support filters yet - filter results manually if needed
```

### 3. Update Component Patterns

#### Pattern Matching

**Before:**
```typescript
const results = await api.patternSearch({
  starts_with: 'k',
  word_length: 'short',
  limit: 10
});
```

**After:**
```typescript
const results = await clientSideData.patternSearch({
  starts_with: 'k',
  limit: 10
});

// Apply filters client-side
const filtered = results.filter(w =>
  w.phoneme_count <= 4 // short words
);
```

#### Similarity Search

**Before:**
```typescript
const similar = await api.getSimilarWords('cat', { threshold: 0.85, limit: 20 });
```

**After:**
```typescript
const similar = await clientSideData.findSimilarWords('cat', 0.85, 20);
```

#### Rhyme Detection

**Before:**
```typescript
const rhymes = await api.getRhymes('cat', {
  rhyme_mode: 'last_1',
  use_embeddings: false,
  limit: 20
});
```

**After:**
```typescript
const rhymes = await clientSideData.findRhymes('cat', 'last_1', 20);
```

### 4. Remove Backend Dependencies

Once migration is complete:

1. **Remove backend API service:**
   ```bash
   rm webapp/frontend/src/services/phonolexApi.ts
   ```

2. **Remove backend environment variable:**
   ```bash
   # Remove from .env
   # VITE_API_URL=http://localhost:8000
   ```

3. **Update deployment config:**
   - No need for backend server
   - Deploy frontend as static site only

## Data Loading Strategy

### Option 1: Load on App Startup (Recommended)

Best for single-page apps where user will use multiple features:

```typescript
// Load all data upfront
useEffect(() => {
  clientSideData.loadData();
}, []);
```

**Pros:** All features work immediately
**Cons:** 88 MB initial load (but gzipped to ~45 MB)

### Option 2: Lazy Loading

For multi-page apps or if initial load is too slow:

```typescript
// Load only when needed
const handleMinimalPairsClick = async () => {
  await clientSideData.loadData(); // Loads only once
  const results = await clientSideData.findMinimalPairs('t', 'd', 10);
};
```

**Pros:** Faster initial page load
**Cons:** First feature use has delay

### Option 3: Progressive Loading (Future)

Split data by word frequency and load high-frequency words first:

```typescript
// TODO: Implement in future version
await clientSideData.loadHighFrequencyWords(); // ~10 MB
// ... user can start using app ...
await clientSideData.loadAllWords(); // ~88 MB total
```

## Performance Optimization

### Use Web Workers for Heavy Computations

Similarity search can be slow for large result sets. Move to worker:

```typescript
// src/workers/similarity.worker.ts
import { clientSideData } from '../services/clientSideData';

self.onmessage = async (e) => {
  const { word, threshold, limit } = e.data;
  const results = await clientSideData.findSimilarWords(word, threshold, limit);
  self.postMessage(results);
};
```

```typescript
// src/hooks/useSimilarity.ts
const worker = new Worker(new URL('../workers/similarity.worker.ts', import.meta.url));

export function useSimilarWords(word: string) {
  const [results, setResults] = useState([]);

  useEffect(() => {
    worker.postMessage({ word, threshold: 0.85, limit: 50 });
    worker.onmessage = (e) => setResults(e.data);
  }, [word]);

  return results;
}
```

### Enable Compression

Make sure your static host serves JSON with Brotli or gzip:

```nginx
# Netlify (automatic)
# Cloudflare Pages (automatic)

# Custom nginx
gzip on;
gzip_types application/json;
gzip_comp_level 6;
```

## API Compatibility Table

| Feature | Backend API | Client-Side | Notes |
|---------|-------------|-------------|-------|
| Get word | ✅ | ✅ | Identical |
| Filter words | ✅ | ✅ | Client-side applies filters manually |
| Pattern search | ✅ | ✅ | Starts with, ends with, contains |
| Minimal pairs | ✅ | ✅ | Identical results |
| Rhymes | ✅ | ⚠️ | Only `last_1` mode implemented (others TODO) |
| Similarity | ✅ | ✅ | Same algorithm, computed client-side |
| Phoneme features | ✅ | ❌ | TODO: Add phoneme data file |
| Graph neighbors | ✅ | ❌ | Not applicable (no pre-computed graph) |
| Statistics | ✅ | ✅ | Basic stats only |

## Deployment

### Netlify

```bash
# Build frontend
cd webapp/frontend
npm run build

# Deploy
netlify deploy --prod --dir=dist
```

### Cloudflare Pages

```bash
# Build frontend
cd webapp/frontend
npm run build

# Deploy
wrangler pages deploy dist
```

### GitHub Pages

```bash
# Build frontend
cd webapp/frontend
npm run build

# Deploy
gh-pages -d dist
```

## Testing

Test the migration locally:

```bash
# Build frontend
cd webapp/frontend
npm run build

# Serve locally
npx serve dist

# Open http://localhost:3000
# Verify all features work without backend
```

## Rollback Plan

If issues arise, you can temporarily revert to backend API:

1. Keep `phonolexApi.ts` file
2. Add environment variable check:
   ```typescript
   const USE_CLIENT_SIDE = import.meta.env.VITE_USE_CLIENT_SIDE === 'true';

   const dataService = USE_CLIENT_SIDE ? clientSideData : api;
   ```

3. Set `VITE_USE_CLIENT_SIDE=false` to use backend

## Next Steps

After migration:

1. ✅ Remove backend server code
2. ✅ Remove PostgreSQL database
3. ✅ Cancel backend hosting (if any)
4. ✅ Update documentation
5. ✅ Celebrate! 🎉

You now have a fully static, serverless PhonoLex app!
