# Testing Guide

PhonoLex uses a comprehensive testing strategy with both **unit tests** (Vitest) and **end-to-end tests** (Playwright) to ensure reliability and prevent regressions.

## Test Structure

```
webapp/frontend/
├── src/
│   └── services/
│       ├── clientSideData.test.ts       # Unit tests for phoneme tokenization & pattern matching
│       └── clientSideApiAdapter.test.ts # Unit tests for exclusion filtering logic
├── e2e/
│   ├── builder-exclusions.spec.ts       # E2E tests for Builder exclusion feature
│   └── minimal-pairs-position.spec.ts   # E2E tests for Minimal Pairs position filtering
├── src/test/
│   └── setup.ts                         # Vitest test setup (jsdom, testing-library)
├── vitest.config.ts                     # Vitest configuration
└── playwright.config.ts                 # Playwright configuration
```

## Running Tests

### Unit Tests (Vitest)

```bash
# Run all unit tests
npm run test

# Run with watch mode (auto-rerun on changes)
npm test

# Run with coverage report
npm run test:coverage

# Run with UI (visual test runner)
npm run test:ui
```

### E2E Tests (Playwright)

```bash
# Run all E2E tests (headless)
npm run test:e2e

# Run with Playwright UI (visual test runner)
npm run test:e2e:ui

# Run all tests (unit + E2E)
npm run test:all
```

## What's Tested

### Unit Tests (17 tests)

**Phoneme Tokenization** (5 tests):
- Space-separated phoneme parsing
- Multi-character phoneme handling (dʒ, tʃ, aɪ, etc.)
- Whitespace handling
- Empty input handling

**Sequence Matching** (5 tests):
- Single phoneme matching
- Multi-phoneme sequence matching
- Order-sensitive matching
- Multi-character phoneme sequences

**Unicode Handling** (3 tests):
- Multi-byte Unicode IPA characters
- Complex IPA characters
- Unicode preservation in matching

**Exclusion Filtering** (4 tests):
- Single phoneme exclusion
- Multiple phoneme exclusions
- Unicode phoneme exclusions
- Empty exclusion list handling

### E2E Tests (Playwright)

**Builder - Exclusion Filtering**:
- Auto-include typed exclusions (no "Add" button required)
- Multiple exclusion handling
- Clear functionality
- IPA keyboard integration

**Minimal Pairs - Position Filtering**:
- Default "any position" behavior
- Word-initial filtering
- Word-final filtering
- Word-medial filtering
- Clear functionality

## Critical Bugs Prevented

The test suite specifically guards against these bugs we fixed:

1. **Exclusion not applied** - Users typed exclusions but they weren't being applied because auto-include logic was missing
2. **Unicode mismatch** - dʒ, ð, θ and other multi-byte characters not properly compared
3. **Position filtering** - Ensuring minimal pairs can be filtered by word position (clinical requirement)

## Adding New Tests

### Unit Test Example

```typescript
// src/services/myService.test.ts
import { describe, it, expect } from 'vitest';

describe('My Service', () => {
  it('should do something', () => {
    expect(true).toBe(true);
  });
});
```

### E2E Test Example

```typescript
// e2e/my-feature.spec.ts
import { test, expect } from '@playwright/test';

test('should interact with feature', async ({ page }) => {
  await page.goto('/');
  await page.click('button');
  await expect(page.locator('text=Result')).toBeVisible();
});
```

## CI/CD Integration

Tests can be run in CI with:

```bash
# Run all tests in CI mode (stricter, no watch mode)
CI=true npm run test:all
```

Playwright will:
- Use headless mode
- Retry failed tests 2x
- Run tests serially (not parallel)
- Generate HTML reports on failure

## Coverage

Run `npm run test:coverage` to generate a coverage report. Coverage files are excluded from version control.

Target coverage goals:
- **Critical paths**: 100% (exclusion filtering, position filtering)
- **Data services**: 80%+ (clientSideData, clientSideApiAdapter)
- **UI components**: 60%+ (functional behavior, not visual)

## Debugging Tests

### Vitest
```bash
# Run specific test file
npm test -- src/services/clientSideData.test.ts

# Run tests matching pattern
npm test -- --grep "exclusion"
```

### Playwright
```bash
# Run specific test file
npm run test:e2e -- builder-exclusions

# Debug mode (headed browser, step through)
npm run test:e2e:ui
```

## Test Philosophy

1. **Test user-facing behavior**, not implementation details
2. **Prioritize critical paths** that affect data correctness
3. **Keep tests fast** - unit tests should run in <1s
4. **Make tests readable** - clear test names, minimal setup
5. **Prevent regressions** - add tests when fixing bugs

## Resources

- [Vitest Documentation](https://vitest.dev/)
- [Playwright Documentation](https://playwright.dev/)
- [Testing Library](https://testing-library.com/)
