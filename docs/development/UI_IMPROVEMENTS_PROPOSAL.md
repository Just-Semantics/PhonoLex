# PhonoLex UI Improvements Proposal

## Current State Analysis

### Existing UI Components (✅ Well-Designed)
- **4 Main Tools**: Quick Tools, Search, Builder, Compare
- **Clean MUI Design**: Professional, accessible, responsive
- **Table-Based Results**: Sortable, exportable (CSV), copyable
- **Current Data Display**: Word, IPA, Syllables, WCM, Complexity, AoA, Similarity

### What's Missing (🆕 Now Available from v2.0 Backend)
- **9 Psycholinguistic properties** not shown in UI
- **Graph relationships** (edges) not visualized
- **Syllable structure** (onset-nucleus-coda) not displayed
- **Edge metadata** (minimal pair positions, rhyme types) not shown
- **MSH stage** mentioned but underutilized
- **No interactive graph explorer**

---

## 🎯 Proposed UI Improvements

### **Priority 1: Enhance Word Cards & Tables** (High Impact, Low Effort)

#### 1.1 Expand WordResultsDisplay Table

**Current**: Shows 6 columns (Word, IPA, Syllables, WCM, Complexity, AoA, Similarity)

**Proposed**: Add toggle-able column groups:

```typescript
Column Groups (user can show/hide):
├── Basic (always visible)
│   ├── Word
│   ├── IPA
│   └── Syllables
├── Complexity Measures
│   ├── WCM
│   ├── Complexity
│   └── MSH Stage ⭐ NEW
├── Psycholinguistic ⭐ NEW
│   ├── AoA (Age of Acquisition)
│   ├── Imageability
│   ├── Concreteness
│   ├── Familiarity
│   └── Frequency
├── Emotional ⭐ NEW (for researchers)
│   ├── Valence
│   ├── Arousal
│   └── Dominance
└── Advanced
    ├── Phoneme Count
    └── Similarity (when applicable)
```

**Rationale**: SLPs care about AoA/Imageability for therapy planning. Researchers want emotional norms. Not everyone needs all columns.

**UI Mockup**:
```
[Show Columns ▼] [Basic ✓] [Complexity ✓] [Psycholinguistic ○] [Emotional ○]
```

---

#### 1.2 Word Detail Modal/Expandable Rows

**Problem**: Table is getting crowded with new data

**Solution**: Click word → opens detailed card with:

```
╔═══════════════════════════════════════════╗
║  cat  /kæt/                               ║
║  ═══════════════════════════════════════  ║
║                                           ║
║  📊 Phonological Structure                ║
║  ┌─────────────────────────────────────┐ ║
║  │ Syllable 1: k-æ-t                   │ ║
║  │  Onset:  [k]                        │ ║
║  │  Nucleus: æ                         │ ║
║  │  Coda:   [t]                        │ ║
║  └─────────────────────────────────────┘ ║
║                                           ║
║  🎓 Developmental (for SLPs)              ║
║    Age of Acquisition: 2.4 years          ║
║    MSH Stage: 2 (Early developing)        ║
║    WCM Score: 3 (Low complexity)          ║
║                                           ║
║  🧠 Psycholinguistic (for Researchers)    ║
║    Imageability:  5.8/7  ⭐⭐⭐⭐⭐⭐       ║
║    Concreteness:  6.2/7  ⭐⭐⭐⭐⭐⭐⭐      ║
║    Familiarity:   6.9/7  ⭐⭐⭐⭐⭐⭐⭐      ║
║    Frequency:     8,234 (log: 3.9)        ║
║                                           ║
║  😊 Emotional Norms                       ║
║    Valence:   6.1/9  (Positive)           ║
║    Arousal:   3.8/9  (Calm)               ║
║    Dominance: 5.2/9  (Neutral)            ║
║                                           ║
║  🔗 Graph Relationships  [Explore →]      ║
║    12 Minimal Pairs  •  45 Rhymes         ║
║    234 Similar Words (>0.85)              ║
║                                           ║
║  [Export Card] [Find Similar] [See Graph] ║
╚═══════════════════════════════════════════╝
```

---

### **Priority 2: Add Graph Explorer Tab** (High Impact, Medium Effort)

#### 2.1 New 5th Tab: "Graph Explorer"

**Concept**: Interactive visualization of phonological relationships

**Features**:
1. **Central Node**: User selects a word (e.g., "cat")
2. **Radial Layout**: Shows neighbors by edge type
3. **Edge Filtering**: Toggle edge types (MINIMAL_PAIR, RHYME, SIMILAR, etc.)
4. **Click to Expand**: Click neighbor → becomes new center

**Visual Layout**:
```
                    bat (RHYME)
                      ↗
        mat (RHYME) ↗    ↘ sat (RHYME)
                  ↗        ↘
      hat ← [CAT] → rat → vat
                ↓
              cut (MINIMAL_PAIR: æ→ʌ)
                ↓
              cot (MINIMAL_PAIR: æ→ɑ)
```

**Edge Type Legend** (with counts):
```
🔵 Minimal Pairs (12)     [Show/Hide]
🟢 Rhymes (45)            [Show/Hide]
🟡 Similar (>0.85) (234)  [Show/Hide]
🟠 Neighbors (8)          [Show/Hide]
```

**Implementation**: Use `react-force-graph-2d` or `react-flow`

---

#### 2.2 Graph Query Builder

**Use Case**: SLPs want to find "all minimal pairs for /t/ vs /d/ that are early-acquired and imageable"

**UI**:
```
┌────────────────────────────────────────┐
│ Graph Query Builder                    │
├────────────────────────────────────────┤
│ Find words that:                       │
│                                        │
│ [Relationship Type ▼]                  │
│   ● Minimal Pairs                      │
│   ○ Rhymes                             │
│   ○ Similar (embedding)                │
│                                        │
│ With contrast:                         │
│   Phoneme 1: [t▼]  Phoneme 2: [d▼]   │
│                                        │
│ Filtered by:                           │
│   □ AoA < 5 years                      │
│   □ Imageability > 5.0                 │
│   □ WCM < 7                            │
│   □ MSH Stage ≤ 3                      │
│                                        │
│ [Run Query →]                          │
└────────────────────────────────────────┘
```

**API Call**:
```typescript
const pairs = await api.getMinimalPairs({
  phoneme1: 't',
  phoneme2: 'd',
  word_length: 'short',
  complexity: 'low'
});

// Then filter client-side by psycholinguistic properties
const filtered = pairs.filter(p =>
  p.word1.aoa < 5 &&
  p.word1.imageability > 5.0 &&
  p.word2.aoa < 5 &&
  p.word2.imageability > 5.0
);
```

---

### **Priority 3: Smart Filtering & Recommendations** (Medium Impact, Medium Effort)

#### 3.1 Filter Presets for Clinical Populations

**Problem**: SLPs don't know optimal filter values for different client types

**Solution**: Pre-configured filter sets

**UI Addition to Quick Tools**:
```
┌─────────────────────────────────────────┐
│ Quick Filter Presets                    │
├─────────────────────────────────────────┤
│ [👶 Early Intervention]                 │
│   AoA < 3 years                         │
│   Imageability > 6.0                    │
│   MSH Stage 1-2                         │
│                                         │
│ [🎒 School-Age]                         │
│   AoA 4-8 years                         │
│   Complexity: Low-Medium                │
│   MSH Stage 3-5                         │
│                                         │
│ [🔬 Research (Adult Norms)]             │
│   Frequency > 5 (log)                   │
│   All psycholinguistic data available   │
│                                         │
│ [🏥 Aphasia Therapy]                    │
│   High Imageability (>5.5)              │
│   High Frequency                        │
│   Low Complexity                        │
│                                         │
│ [Custom...]                             │
└─────────────────────────────────────────┘
```

---

#### 3.2 "Optimal Pair Finder" (AI-Powered)

**Concept**: Given a target word, find the BEST contrasts for a specific goal

**Example**:
```
Target Word: cat

Goal: Minimal pair therapy for /k/ deletion

Optimal Pairs:
1. cat vs. at     (initial /k/)     AoA: 2.1, Img: 6.8 ⭐ Best
2. cat vs. bat    (k→b contrast)    AoA: 2.4, Img: 6.5
3. cat vs. hat    (k→h contrast)    AoA: 2.6, Img: 6.2

Rationale:
- All pairs are early-acquired (AoA < 3)
- High imageability for picture-based therapy
- "cat vs. at" isolates /k/ in onset position
```

**Backend**: Simple scoring algorithm:
```python
def score_pair(word1, word2, goal='therapy'):
    score = 0

    # Prefer early-acquired words
    if word1.aoa and word1.aoa < 4:
        score += 10

    # Prefer high imageability
    if word1.imageability and word1.imageability > 5.5:
        score += 8

    # Prefer common words
    if word1.frequency and word1.frequency > 100:
        score += 5

    # Prefer low complexity
    if word1.wcm_score and word1.wcm_score < 6:
        score += 5

    return score
```

---

### **Priority 4: Enhanced Visualizations** (Medium Impact, High Effort)

#### 4.1 Psycholinguistic Property Radars

**For Word Comparison**:
```
Compare "cat" vs "dog":

       AoA (2.4)           Imageability (6.8)
            ╱ ╲                    ╱ ╲
           ╱   ╲                  ╱   ╲
Frequency ●─────● Concreteness  ●─────●
(8234)    ╲   ╱  (6.2)         (7234) ╲   ╱  (6.5)
           ╲ ╱                         ╲ ╱
        Familiarity (6.9)           Familiarity (7.1)

  🐱 cat            🐶 dog
```

---

#### 4.2 Syllable Structure Visualizer

**For Phonological Awareness Teaching**:
```
Word: computer

Syllable Breakdown:
┌─────────┬─────────┬─────────┐
│   C1    │   C2    │   C3    │
├─────────┼─────────┼─────────┤
│ k-ə-m   │ p-juː-  │ t-ə-ɹ   │
│         │         │         │
│ O: k    │ O: p    │ O: t    │
│ N: ə    │ N: juː  │ N: ə    │
│ C: m    │ C: ∅    │ C: ɹ    │
└─────────┴─────────┴─────────┘

Onset:  [k]   [p]   [t]
Nucleus: ə     juː   ə
Coda:   [m]   [∅]   [ɹ]
```

---

#### 4.3 Embedding Space Visualization (Advanced)

**For Researchers**:
```
t-SNE Projection of Syllable Embeddings

Similar words cluster together:

    mat •
  cat • • bat
    • sat
     rat

              • dog
            • • log
          • fog • hog

[Color by: AoA ▼]  [3D View]  [Export Data]
```

---

### **Priority 5: UX Quality-of-Life** (Low-Medium Impact, Low Effort)

#### 5.1 Smart Defaults & Memory

- **Remember last query**: Store in localStorage
- **Recent searches**: Quick access to last 5 queries
- **Favorite words**: Star words for quick reference

#### 5.2 Keyboard Shortcuts

```
Ctrl+K: Quick search
Ctrl+1-4: Switch tabs
Ctrl+E: Export results
Esc: Clear filters
```

#### 5.3 Progressive Disclosure

**Problem**: Too many options overwhelm new users

**Solution**:
- **Beginner Mode**: Show only essential filters
- **Advanced Mode**: Show all psycholinguistic properties
- Toggle in header: `[Simple View] / [Advanced View]`

#### 5.4 Contextual Help Tooltips

Add `(?)` icons with explanations:
```
AoA (?) → "Age of Acquisition: The typical age (in years)
          when children learn this word. Lower = earlier."

Imageability (?) → "How easily the word can be pictured
                   (1-7 scale). Higher = more concrete."

MSH Stage (?) → "Motor Speech Hierarchy stage (1-8).
                Lower stages are earlier-developing sounds."
```

---

## 📊 Implementation Roadmap

### Phase 1: Low-Hanging Fruit (1-2 days)
1. ✅ Add MSH Stage column to WordResultsDisplay
2. ✅ Add column visibility toggles
3. ✅ Add tooltips for psycholinguistic properties
4. ✅ Add filter presets for clinical populations

### Phase 2: Word Detail Enhancement (2-3 days)
1. ✅ Create WordDetailModal component
2. ✅ Display syllable structure breakdown
3. ✅ Show all psycholinguistic properties
4. ✅ Add "Explore Graph" button

### Phase 3: Graph Explorer (1 week)
1. ✅ Create GraphExplorer tab
2. ✅ Implement force-directed graph layout
3. ✅ Add edge type filtering
4. ✅ Add click-to-expand interaction
5. ✅ Add graph query builder

### Phase 4: Advanced Features (1-2 weeks)
1. ✅ Implement "Optimal Pair Finder"
2. ✅ Add psycholinguistic radar charts
3. ✅ Add syllable structure visualizer
4. ✅ Add keyboard shortcuts

### Phase 5: Polish & Research Tools (ongoing)
1. ✅ Add embedding space visualization (t-SNE)
2. ✅ Add batch export for research
3. ✅ Add API usage dashboard
4. ✅ Add documentation wiki

---

## 🎨 Visual Design Principles

### Color Coding
- **AoA**: Green (early) → Yellow (school-age) → Red (late)
- **Imageability**: Blue gradient (low → high)
- **Complexity**: Traffic light (green/yellow/red)
- **Edge Types**: Distinct colors for each relationship type

### Typography
- **IPA**: Monospace font (Charis SIL, Doulos SIL)
- **Numbers**: Tabular figures for alignment
- **Word**: Sans-serif, medium weight

### Accessibility
- **WCAG AA compliance**: All color contrasts ≥4.5:1
- **Keyboard navigation**: Full keyboard support
- **Screen readers**: ARIA labels on all interactive elements
- **Focus indicators**: Clear visual focus states

---

## 💡 My Top 3 Recommendations

### 1. **Word Detail Modal** (Highest ROI)
- **Why**: Exposes all new data without cluttering the table
- **Effort**: Low (1-2 days)
- **Impact**: Immediately useful for all users

### 2. **Graph Explorer Tab** (Game Changer)
- **Why**: Unique feature, no competitor has this
- **Effort**: Medium (1 week)
- **Impact**: Differentiates PhonoLex from other tools

### 3. **Clinical Filter Presets** (SLP-Focused)
- **Why**: Makes complex filters accessible to non-technical users
- **Effort**: Low (1 day)
- **Impact**: Drastically improves UX for target audience

---

## 🤔 Questions for You

1. **Target Audience Balance**: What's the split between:
   - SLPs (clinical) → want simplicity, therapy-relevant filters
   - Researchers (academic) → want all data, export, visualizations
   - Students (learning) → want explanations, examples

2. **Graph Explorer Priority**: Is graph visualization worth 1 week of dev time?

3. **Emotional Norms**: Are valence/arousal/dominance useful for your users, or too niche?

4. **Embedding Visualization**: Do researchers want to explore embedding space, or is this over-engineering?

5. **Mobile Support**: Do users need mobile-responsive design, or is this desktop-only?

---

## 📦 Component Reusability

All proposed components should be:
- **Composable**: Small, single-purpose components
- **Themed**: Use MUI theme system
- **Tested**: Unit tests for logic, visual tests for UI
- **Documented**: Storybook stories for each component
- **Accessible**: ARIA labels, keyboard nav, screen reader support

---

## Next Steps

Let me know:
1. Which priorities resonate most with your vision?
2. Any features you'd add/remove?
3. Timeline constraints?
4. Should I start implementing any of these?

I can create:
- Working prototypes for any feature
- Detailed component specs
- API integration code
- Visual mockups (if you have design tools)

What would be most helpful? 🚀
