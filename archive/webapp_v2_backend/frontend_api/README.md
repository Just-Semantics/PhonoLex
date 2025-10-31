# Frontend Backend API Client (Archived)

This directory contains the TypeScript API client that was used to communicate with the v2.0 FastAPI backend.

## File

- `phonolexApi.backend.ts` - HTTP client for backend API endpoints

## Usage (Historical)

```typescript
import { api } from './services/phonolexApi.backend';

// Make HTTP request to backend
const words = await api.findMinimalPairs('t', 'd', { limit: 10 });
```

## Replacement

This has been replaced by the client-side data adapter:

```typescript
import { api } from './services/phonolexApi';

// Now uses client-side computation
const words = await api.getMinimalPairs({
  phoneme1: 't',
  phoneme2: 'd',
  limit: 10
});
```

The new `phonolexApi.ts` re-exports the `clientSideApiAdapter.ts` which wraps `clientSideData.ts` to provide the same interface without network calls.
