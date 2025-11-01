import { test, expect } from '@playwright/test';

/**
 * E2E tests for Custom Word List Builder exclusion filtering
 *
 * Tests the critical bug fix where exclusions weren't being applied
 * if the user typed them but didn't click Add button.
 */
test.describe('Builder - Exclusion Filtering', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for data to load
    await page.waitForSelector('text=Custom Word List Builder', { timeout: 10000 });
  });

  test('should exclude words with typed exclusion (auto-include)', async ({ page }) => {
    // Expand Builder if collapsed
    const builderCard = page.locator('text=Custom Word List Builder').locator('..');
    await builderCard.click();

    // Wait for Builder to expand
    await page.waitForTimeout(500);

    // Add pattern: starts with "m"
    const phonemeInput = page.locator('input[placeholder*="e.g., k, t, s"]').first();
    await phonemeInput.fill('m');

    // Type exclusion but DON'T click Add button
    const exclusionInput = page.locator('input[placeholder*="r, l, θ"]');
    await exclusionInput.fill('dʒ');

    // Click Build (should auto-include the typed exclusion)
    await page.getByRole('button', { name: /build/i }).click();

    // Wait for results
    await page.waitForSelector('text=/\\d+ results/', { timeout: 5000 });

    // Check that "magenta" is NOT in results (contains dʒ)
    const resultsText = await page.textContent('body');
    expect(resultsText).not.toContain('magenta');

    // But "make", "mat", "man" should be present (no dʒ)
    expect(resultsText).toMatch(/make|mat|man/);
  });

  test('should handle multiple exclusions', async ({ page }) => {
    const builderCard = page.locator('text=Custom Word List Builder').locator('..');
    await builderCard.click();
    await page.waitForTimeout(500);

    // Pattern: starts with "t"
    const phonemeInput = page.locator('input[placeholder*="e.g., k, t, s"]').first();
    await phonemeInput.fill('t');

    // Exclude both "dʒ" and "ð"
    const exclusionInput = page.locator('input[placeholder*="r, l, θ"]');
    await exclusionInput.fill('dʒ ð');

    await page.getByRole('button', { name: /build/i }).click();
    await page.waitForSelector('text=/\\d+ results/', { timeout: 5000 });

    const resultsText = await page.textContent('body');

    // Should NOT contain words with dʒ or ð
    expect(resultsText).not.toContain('judge');
    expect(resultsText).not.toContain('them');
    expect(resultsText).not.toContain('that');

    // Should contain words starting with "t" without excluded phonemes
    expect(resultsText).toMatch(/take|time|top/);
  });

  test('should clear exclusions when clicking Clear', async ({ page }) => {
    const builderCard = page.locator('text=Custom Word List Builder').locator('..');
    await builderCard.click();
    await page.waitForTimeout(500);

    // Add pattern and exclusion
    const phonemeInput = page.locator('input[placeholder*="e.g., k, t, s"]').first();
    await phonemeInput.fill('m');

    const exclusionInput = page.locator('input[placeholder*="r, l, θ"]');
    await exclusionInput.fill('dʒ');

    // Click Clear
    await page.getByRole('button', { name: /clear/i }).click();

    // Inputs should be empty
    await expect(phonemeInput).toHaveValue('');
    await expect(exclusionInput).toHaveValue('');
  });

  test('should work with IPA keyboard for exclusions', async ({ page }) => {
    const builderCard = page.locator('text=Custom Word List Builder').locator('..');
    await builderCard.click();
    await page.waitForTimeout(500);

    // Open IPA keyboard for exclusion input
    const exclusionInputWrapper = page.locator('input[placeholder*="r, l, θ"]').locator('..');
    const keyboardButton = exclusionInputWrapper.locator('button[aria-label*="phoneme picker"]');
    await keyboardButton.click();

    // Wait for phoneme picker dialog
    await page.waitForSelector('text=Select Phonemes', { timeout: 2000 });

    // Click on "dʒ" in the phoneme picker
    await page.locator('button:has-text("dʒ")').first().click();

    // Close picker
    await page.locator('button:has-text("Close")').click();

    // The exclusion input should now have "dʒ"
    const exclusionInput = page.locator('input[placeholder*="r, l, θ"]');
    await expect(exclusionInput).toHaveValue('dʒ');
  });
});
