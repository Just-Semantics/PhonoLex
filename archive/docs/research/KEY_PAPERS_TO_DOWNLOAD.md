# Key Research Papers for PhonoLex SLP Frontend

This document lists essential research papers that inform the design of our clinical word list generation system.

## Priority 1: Already Have ✅

1. **Namasivayam et al. (2021)** - Development and Validation of a Probe Word List
   - Location: `/research/papers/namasivayam-et-al-2021-development-and-validation-of-a-probe-word-list.pdf`
   - **Why Critical**: Motor Speech Hierarchy framework, validated probe word list methodology
   - **Key Findings**:
     - Hierarchical developmental stages (mandibular → labial → lingual → sequenced)
     - Motor complexity scoring system
     - Controlled linguistic variables (frequency, neighborhood density, biphone frequency)
     - 42-57% variance in intelligibility explained by motor scores

## Priority 2: Essential Papers to Download

### A. Complexity Approach

2. **Gierut (2001)** - Complexity in Phonological Treatment: Clinical Factors
   - **Download**: https://www.speech-language-therapy.com/pdf/gierut2001.pdf
   - **Why Critical**: Foundational paper on complexity-based target selection
   - **Key Concepts**:
     - Treating complex targets → broader generalization
     - Implicational universals (clusters → singletons)
     - Target selection based on developmental norms and stimulability

3. **Gierut (2018)** - The Complexity Approach to Phonological Treatment: How to Select Treatment Targets
   - **Download**: https://pubs.asha.org/doi/pdf/10.1044/2017_LSHSS-17-0082
   - **Why Critical**: Updated clinical tutorial with evidence synthesis
   - **Key Concepts**:
     - System-wide generalization patterns
     - Evidence-based target selection criteria
     - Clinical decision-making framework

### B. Minimal Pairs

4. **Barlow & Gierut (2002)** - Minimal Pair Approaches to Phonological Remediation
   - **Download**: https://www.speech-language-therapy.com/pdf/barlowgierut2002.pdf
   - **Why Critical**: Comprehensive review of minimal pair efficacy
   - **Key Concepts**:
     - Hierarchy of minimal pair treatment efficacy
     - Number of featural differences matters
     - Type of featural differences (place vs. manner vs. voicing)

5. **Baker & Williams (2021)** - Minimal, Maximal, or Multiple: Which Contrastive Intervention
   - **Download**: https://pubs.asha.org/doi/pdf/10.1044/2021_LSHSS-21-00105
   - **Why Critical**: Recent evidence comparing contrastive approaches
   - **Key Concepts**:
     - Minimal pairs vs. maximal oppositions vs. multiple oppositions
     - Evidence for when to use each approach
     - Clinical decision tree

### C. Word Complexity Measures

6. **Stoel-Gammon (2010)** - The Word Complexity Measure
   - **Search**: PubMed or ResearchGate
   - **Citation**: *Clinical Linguistics & Phonetics*, 24(4-5), 271-282
   - **Why Critical**: Original word complexity scoring system (basis for Namasivayam's modified version)
   - **Key Concepts**:
     - 8 phonological complexity parameters
     - Word patterns, syllable structures, sound classes
     - Independent of accuracy (production-based)

### D. Lexical Factors

7. **Storkel et al. (2018)** - Implementing Evidence-Based Practice: Selecting Treatment Words to Boost Phonological Learning
   - **Download**: https://pubs.asha.org/doi/10.1044/2017_LSHSS-17-0080
   - **Why Critical**: Evidence on word frequency, neighborhood density, phonotactic probability
   - **Key Concepts**:
     - High frequency words → better learning
     - Dense neighborhoods → easier acquisition in children
     - Interaction with motor complexity

### E. Developmental Norms

8. **McLeod & Crowe (2018)** - Children's Consonant Acquisition in 27 Languages
   - **Download**: https://pubs.asha.org/doi/10.1044/2018_AJSLP-17-0100
   - **Why Critical**: Cross-linguistic developmental norms (gold standard)
   - **Key Findings**:
     - Age of acquisition data for all English consonants
     - /b,n,m,p,h,w,d/ by age 2-3
     - /r,ð,ʒ/ by age 5-6
     - /θ/ by age 6-7

9. **McLeod et al. (2020)** - Children's English Consonant Acquisition in the United States
   - **Download**: https://pubs.asha.org/doi/10.1044/2020_AJSLP-19-00168
   - **Why Critical**: US-specific norms (more recent than Sander 1972)
   - **Key Findings**:
     - Updates to traditional developmental charts
     - Children acquire sounds earlier than previously thought
     - Evidence-based eligibility criteria

## Priority 3: Supporting Papers

### Imageability & Concreteness

10. **Word Characteristics in Vocabulary Learning** - Multiple studies found
    - Imageability ratings and their role in word learning
    - Concreteness effects in children with language disorders
    - Practical application to word selection

### Motor Speech Development

11. **Green & Nip (2010)** - Organization Principles in Early Speech Development
    - Motor control development evidence supporting MSH
    - Lip-mandible coordination timeline
    - Tongue-mandible coordination development

## How These Papers Inform PhonoLex

### 1. Motor Complexity Scoring
- **From Namasivayam + Stoel-Gammon**: Implement WCM scoring
- **Application**: Score every word in CMU dictionary, filter by complexity level

### 2. Hierarchical Word Lists
- **From Namasivayam MSH**: Generate stage-specific lists
- **From Gierut**: Add complexity-based recommendations
- **Application**: "Stage III words" vs "Stage V words" with smart recommendations

### 3. Minimal Pair Generation
- **From Barlow & Gierut**: Feature-based minimal pair finding
- **From Baker & Williams**: Provide minimal vs. maximal options
- **Application**: Smart minimal pair finder that suggests maximal oppositions for severe cases

### 4. Psycholinguistic Filtering
- **From Storkel**: Add frequency/neighborhood filters
- **From AssocNet data**: Add concreteness/imageability/valence filters
- **Application**: "Child-friendly, high-frequency minimal pairs"

### 5. Age-Appropriate Recommendations
- **From McLeod & Crowe**: Developmental norms database
- **Application**: "Words appropriate for age 4" filter

## Download Strategy

### Immediate (This Week)
1. ✅ Namasivayam et al. 2021 (HAVE IT)
2. Gierut 2001 (freely available PDF)
3. Barlow & Gierut 2002 (freely available PDF)
4. Gierut 2018 ASHA tutorial

### Next Week
5. McLeod & Crowe 2018 (may need university access)
6. Stoel-Gammon 2010 (may need purchase)
7. Storkel 2018 ASHA article

### Research Access Needed
- Check if you have university library access (Toronto?)
- ASHA members get free access to ASHA journals
- ResearchGate often has author-uploaded PDFs

## Citation Management

All papers should be:
1. Saved to `/research/papers/` with consistent naming: `author-year-short-title.pdf`
2. Cited in our implementation code where algorithms are based on research
3. Referenced in our documentation/README
4. Used to validate our approach matches clinical best practices

---

**Last Updated**: 2025-10-27
**Maintained By**: PhonoLex Development Team
