# Contrastive Intervention Tool - Unified Architecture Proposal

**Date**: January 6, 2025
**Status**: Proposal for v2.2
**Goal**: Consolidate three contrastive intervention approaches into one tool with mode selection

## Current State (v2.1)

Three separate tools with distinct UIs:

1. **MinimalPairsTool** - Conventional minimal pairs (target + substitute)
2. **MaximalOppositionTool** - Two unknowns with major class difference
3. *(Not yet implemented)* Multiple Opposition - Global phoneme collapse

**Problem**:
- Redundant UI components
- Inconsistent user experience
- Difficult to add new intervention modes
- Users must understand which tool to use for which clinical scenario

## Proposed Architecture (v2.2)

### Single Unified Tool: `ContrastiveInterventionTool.tsx`

Following the SearchTool pattern: **ToggleButtonGroup for mode selection** → **Conditional inputs** → **Single action button**

```typescript
type InterventionMode = 'minimal' | 'maximal' | 'multiple';

const [mode, setMode] = useState<InterventionMode>('minimal');
```

---

## Three Intervention Modes

### Mode 1: Minimal Pairs (Conventional)

**Clinical Indication**: Mild SSD, 1-2 error patterns
**Research**: Traditional phonological therapy

**Inputs**:
```
┌─────────────────────────────────────────┐
│ Phoneme 1 (Target):    [  θ  ] [🎹]    │
│ Phoneme 2 (Substitute): [  t  ] [🎹]    │
│ Position: [Any ▼]                        │
└─────────────────────────────────────────┘
```

**Output**: Direct word pairs
```
1. thin - tin
2. thick - tick
3. thought - taught
```

**Implementation**: Use existing MinimalPairsTool logic

---

### Mode 2: Maximal Opposition

**Clinical Indication**: Moderate-severe SSD, multiple error classes
**Research**: Gierut (1989-1992), Storkel (2022)

**Inputs**:
```
┌─────────────────────────────────────────┐
│ Unknown Phonemes: [  g θ ʃ l r ŋ  ] [🎹]│
│ (Enter all error sounds)                 │
└─────────────────────────────────────────┘
```

**Step 1 Output**: Scored pairs (major class + features)
```
Select a pair to generate word lists:

□ /θ/ - /ŋ/   Score: 114 (14 features, major class ✓)
□ /ʃ/ - /l/   Score: 108 (8 features, major class ✓)
□ /g/ - /l/   Score: 104 (4 features, major class ✓)

Position: [Initial ▼]
```

**Step 2 Output**: Word pairs for selected pair
```
/θ/ - /ŋ/ in Initial position:

1. thick - Nick
2. thought - not
```

**Implementation**: Use existing MaximalOppositionTool logic, but:
- Single input field (not separate sonorant/obstruent fields)
- Algorithm automatically filters by major class internally
- UI shows why pairs don't appear if no major class difference exists

---

### Mode 3: Multiple Opposition

**Clinical Indication**: Severe SSD, global phoneme collapse (age 3-6)
**Research**: Gierut (1989-1992), Storkel (2022)

**Inputs**:
```
┌─────────────────────────────────────────┐
│ Substitute Phoneme: [  t  ] [🎹]        │
│ (What the child SAYS)                    │
│                                          │
│ Target Phonemes: [  θ k l kr tr  ] [🎹] │
│ (What the child SHOULD say)              │
│ (Space/comma separated)                  │
│                                          │
│ Position: [Initial ▼]                    │
└─────────────────────────────────────────┘
```

**Step 1 Output**: Show selected representative targets
```
Based on Maximal Classification + Maximal Distinction:

Selected targets: /θ/ /l/ /kr/
(3 targets representing the breadth of the collapse)

Continue to generate word sets?
```

**Step 2 Output**: Minimal triplets/quadruplets
```
Minimal Triplets for [t] → /t θ l kr/:

1. toes - those - lows - crows
2. tin - thin - Lynn - grin
3. toe - though - low - grow
```

**Implementation**: New algorithms needed:
1. `selectRepresentativeTargets()`: Maximal Classification + Maximal Distinction
2. `generateMinimalSets()`: Find words differing only at target position across 3-5 targets

---

## Unified UI Structure

```tsx
<Box>
  {/* Mode Toggle */}
  <ToggleButtonGroup value={mode} exclusive onChange={handleModeChange} fullWidth>
    <ToggleButton value="minimal">
      <Icon /> Minimal Pairs
    </ToggleButton>
    <ToggleButton value="maximal">
      <Icon /> Maximal Opposition
    </ToggleButton>
    <ToggleButton value="multiple">
      <Icon /> Multiple Opposition
    </ToggleButton>
  </ToggleButtonGroup>

  {/* Clinical Context Helper */}
  <Alert severity="info" sx={{ mt: 2 }}>
    {mode === 'minimal' && "For mild SSD with 1-2 error patterns"}
    {mode === 'maximal' && "For moderate-severe SSD with multiple error classes"}
    {mode === 'multiple' && "For severe SSD with global phoneme collapse"}
  </Alert>

  {/* Conditional Inputs */}
  {mode === 'minimal' && <MinimalInputs />}
  {mode === 'maximal' && <MaximalInputs />}
  {mode === 'multiple' && <MultipleInputs />}

  {/* Action Buttons */}
  <Button variant="contained" onClick={handleGenerate}>
    Generate
  </Button>

  {/* Results */}
  {results && <ResultsDisplay />}
</Box>
```

---

## Alignment Considerations

### Shared Components
- **PhonemePickerDialog**: Already supports filtering (sonorants/obstruents)
- **Position selector**: All three modes use position filtering
- **WordResultsDisplay**: Reusable for pairs/triplets/quadruplets

### Shared State
```typescript
interface ContrastiveState {
  mode: InterventionMode;
  phonemes: {
    phoneme1?: string;       // Minimal mode
    phoneme2?: string;       // Minimal mode
    unknowns?: string[];     // Maximal mode
    substitute?: string;     // Multiple mode
    targets?: string[];      // Multiple mode
  };
  position: 'any' | 'initial' | 'medial' | 'final';
  results: MinimalPair[] | MaximalPair[] | MultiplePair[];
}
```

### Input Alignment Strategy

**Option A: Unified Phoneme Input** (Recommended)
- All modes use same "Phonemes" field with helper text varying by mode
- Validation logic changes based on mode
- Simpler UI, less code duplication

**Option B: Mode-Specific Input Sections**
- Each mode has completely separate input fields
- Clearer separation, easier to understand
- More code duplication, larger component

**Recommendation**: Option A with strong helper text and validation

---

## Implementation Phases

### Phase 1: Refactor Existing Tools (Week 1)
1. Create `ContrastiveInterventionTool.tsx`
2. Extract `MinimalInputs`, `MaximalInputs` as sub-components
3. Add mode toggle with minimal/maximal modes
4. Migrate logic from MinimalPairsTool and MaximalOppositionTool
5. Update routes/navigation
6. Deprecate old tools

### Phase 2: Implement Multiple Opposition (Week 2)
1. Implement `selectRepresentativeTargets()` algorithm
   - Maximal Classification: Select targets representing breadth
   - Maximal Distinction: Maximize phonological distance from substitute
2. Implement `generateMinimalSets()` algorithm
   - Find words differing at exactly N target positions
   - Return triplets/quadruplets/quintuplets
3. Create `MultipleInputs` component
4. Add 'multiple' mode to toggle
5. Test with research examples (Storkel 2022)

### Phase 3: Polish & Documentation (Week 3)
1. Add clinical decision guidance (when to use which mode)
2. Update all documentation
3. Add examples to Info drawer
4. User testing with SLPs

---

## Research Validation

Test cases from literature:

### Minimal Pairs (Baseline)
```
Input: /θ/ - /t/, initial
Output: thin-tin, thick-tick, thought-taught
```

### Maximal Opposition (Gierut 1990, Storkel 2022)
```
Input: g θ ʃ l r ŋ
Output: /θ/-/ŋ/ (score 114), /ʃ/-/l/ (score 108), /g/-/l/ (score 104)
Word pairs: gore-lore, game-lame, gab-lab
```

### Multiple Opposition (Storkel 2022, Case "Ethan")
```
Input: [t] → /t θ k l kr tr θr sk st skr str/
Selected: /kl θr st/ (maximal classification + distinction)
Output: toes-close-throws-stows, toe-claw-Thor-stall
```

---

## Benefits

1. **Single entry point** - Users don't need to know which tool to use upfront
2. **Educational** - Mode descriptions teach clinical decision-making
3. **Consistent UX** - Same UI patterns across all modes
4. **Extensible** - Easy to add future modes (e.g., "Cycles", "Complexity")
5. **Reduced code** - Shared components and validation logic

---

## Migration Strategy

1. **v2.2-alpha**: New tool exists alongside old tools (feature flag)
2. **v2.2-beta**: Default to new tool, old tools available via URL
3. **v2.2-stable**: Remove old tools completely

Old tool URLs can redirect:
- `/tools/minimal-pairs` → `/tools/contrastive-intervention?mode=minimal`
- `/tools/maximal-opposition` → `/tools/contrastive-intervention?mode=maximal`

---

## Questions for Discussion

1. **Input field design**: Single unified field vs. mode-specific fields?
2. **Default mode**: Start with 'minimal' (most common) or show all three equally?
3. **Multi-step UI**: Should maximal/multiple modes use stepper component for clarity?
4. **Feature naming**: "Contrastive Intervention" vs. "Phoneme Contrasts" vs. "Intervention Planner"?
5. **Mobile experience**: How to handle 3 toggle buttons on small screens? (Vertical stack vs. dropdown)

---

## Next Steps

**Immediate**: Get user approval on:
1. Overall architecture direction
2. Input alignment strategy (Option A vs. B)
3. Implementation phase priority

**Then**: Begin Phase 1 refactoring
