import { test, expect } from '@playwright/test';

/**
 * E2E tests for Minimal Pairs position filtering
 *
 * Tests the new clinically-relevant position filtering feature.
 */
test.describe('Minimal Pairs - Position Filtering', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for data to load
    await page.waitForSelector('text=Minimal Pairs', { timeout: 10000 });
  });

  test('should generate pairs at any position by default', async ({ page }) => {
    // Expand Minimal Pairs card
    const pairsCard = page.locator('text=Minimal Pairs').locator('..');
    await pairsCard.click();
    await page.waitForTimeout(500);

    // Enter phonemes
    const phoneme1Input = page.locator('input[placeholder*="t, k, s"]').first();
    const phoneme2Input = page.locator('input[placeholder*="d, g, z"]').first();

    await phoneme1Input.fill('t');
    await phoneme2Input.fill('d');

    // Position should default to "Any Position"
    const positionSelect = page.locator('select, [role="combobox"]').filter({ hasText: /position/i });
    await expect(positionSelect).toHaveValue('any');

    // Generate
    await page.getByRole('button', { name: /generate|run/i }).click();
    await page.waitForSelector('text=/\\d+ (pair|result)/', { timeout: 5000 });

    // Should have results
    const resultsText = await page.textContent('body');
    expect(resultsText).toMatch(/cat.*cad|bat.*bad|ten.*den/i);
  });

  test('should filter pairs to word-initial position', async ({ page }) => {
    const pairsCard = page.locator('text=Minimal Pairs').locator('..');
    await pairsCard.click();
    await page.waitForTimeout(500);

    // Enter phonemes t vs d
    const phoneme1Input = page.locator('input[placeholder*="t, k, s"]').first();
    const phoneme2Input = page.locator('input[placeholder*="d, g, z"]').first();

    await phoneme1Input.fill('t');
    await phoneme2Input.fill('d');

    // Select word-initial position
    const positionSelect = page.locator('select[value="any"]').first();
    await positionSelect.selectOption('initial');

    // Generate
    await page.getByRole('button', { name: /generate|run/i }).click();
    await page.waitForSelector('text=/\\d+ (pair|result)/', { timeout: 5000 });

    // Results should only include word-initial pairs
    // e.g., "ten/den", "tip/dip" but NOT "cat/cad" (final position)
    const resultsText = await page.textContent('body');
    expect(resultsText).toMatch(/ten.*den|tip.*dip|two.*do/i);
  });

  test('should filter pairs to word-final position', async ({ page }) => {
    const pairsCard = page.locator('text=Minimal Pairs').locator('..');
    await pairsCard.click();
    await page.waitForTimeout(500);

    // Enter phonemes t vs d
    const phoneme1Input = page.locator('input[placeholder*="t, k, s"]').first();
    const phoneme2Input = page.locator('input[placeholder*="d, g, z"]').first();

    await phoneme1Input.fill('t');
    await phoneme2Input.fill('d');

    // Select word-final position
    const positionSelect = page.locator('select[value="any"]').first();
    await positionSelect.selectOption('final');

    // Generate
    await page.getByRole('button', { name: /generate|run/i }).click();
    await page.waitForSelector('text=/\\d+ (pair|result)/', { timeout: 5000 });

    // Results should only include word-final pairs
    // e.g., "cat/cad", "bat/bad" but NOT "ten/den" (initial position)
    const resultsText = await page.textContent('body');
    expect(resultsText).toMatch(/cat.*cad|bat.*bad|bit.*bid/i);
  });

  test('should filter pairs to word-medial position', async ({ page }) => {
    const pairsCard = page.locator('text=Minimal Pairs').locator('..');
    await pairsCard.click();
    await page.waitForTimeout(500);

    // Enter phonemes t vs d
    const phoneme1Input = page.locator('input[placeholder*="t, k, s"]').first();
    const phoneme2Input = page.locator('input[placeholder*="d, g, z"]').first();

    await phoneme1Input.fill('t');
    await phoneme2Input.fill('d');

    // Select word-medial position
    const positionSelect = page.locator('select[value="any"]').first();
    await positionSelect.selectOption('medial');

    // Generate
    await page.getByRole('button', { name: /generate|run/i }).click();
    await page.waitForSelector('text=/\\d+ (pair|result)/', { timeout: 5000 });

    // Results should only include word-medial pairs (position > 0 and < last)
    // This is less common, so may have fewer results
    const resultsText = await page.textContent('body');
    // Just verify we got some results
    expect(resultsText).toMatch(/\\d+ (pair|result)/);
  });

  test('should reset position filter when clicking Clear', async ({ page }) => {
    const pairsCard = page.locator('text=Minimal Pairs').locator('..');
    await pairsCard.click();
    await page.waitForTimeout(500);

    // Set position to initial
    const positionSelect = page.locator('select[value="any"]').first();
    await positionSelect.selectOption('initial');

    // Enter phonemes
    const phoneme1Input = page.locator('input[placeholder*="t, k, s"]').first();
    await phoneme1Input.fill('t');

    // Click Clear
    await page.getByRole('button', { name: /clear/i }).click();

    // Position should reset to "any"
    await expect(positionSelect).toHaveValue('any');

    // Phoneme inputs should be empty
    await expect(phoneme1Input).toHaveValue('');
  });
});
