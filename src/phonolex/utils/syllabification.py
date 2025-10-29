#!/usr/bin/env python3
"""
Syllable structure extraction from phoneme sequences with stress.

Extracts onset-nucleus-coda structure for each syllable.
"""

from dataclasses import dataclass
from typing import List, Optional
from ..embeddings.english_data_loader import PhonemeWithStress


@dataclass
class Syllable:
    """A syllable with onset-nucleus-coda structure"""
    onset: List[str]  # Consonants before nucleus
    nucleus: str  # Vowel (required)
    coda: List[str]  # Consonants after nucleus
    stress: int  # 0=unstressed, 1=primary, 2=secondary

    def __str__(self):
        onset_str = ''.join(self.onset) if self.onset else ''
        coda_str = ''.join(self.coda) if self.coda else ''
        return f"[{onset_str}·{self.nucleus}{self.stress}·{coda_str}]"

    def to_phoneme_list(self) -> List[str]:
        """Return flat phoneme list"""
        return self.onset + [self.nucleus] + self.coda


# Define vowels (have stress markers in CMU)
VOWELS = {
    'i', 'ɪ', 'ɛ', 'æ', 'ɑ', 'ɔ', 'ʊ', 'u', 'ʌ', 'ə', 'ɝ', 'ɚ',  # Monophthongs
    'eɪ', 'aɪ', 'ɔɪ', 'aʊ', 'oʊ'  # Diphthongs
}


def is_vowel(phoneme: str) -> bool:
    """Check if phoneme is a vowel"""
    return phoneme in VOWELS


def syllabify(phonemes_with_stress: List[PhonemeWithStress]) -> List[Syllable]:
    """
    Extract syllables from phoneme sequence with stress.

    Algorithm:
    1. Find vowels (syllable nuclei)
    2. Assign consonants to syllables using maximal onset principle
    3. Create Syllable objects with onset-nucleus-coda

    Args:
        phonemes_with_stress: Phonemes with stress markers

    Returns:
        List of Syllable objects
    """
    if not phonemes_with_stress:
        return []

    # Find vowel positions (nuclei)
    vowel_positions = []
    for i, p in enumerate(phonemes_with_stress):
        if is_vowel(p.phoneme):
            vowel_positions.append(i)

    if not vowel_positions:
        # No vowels - treat as single syllable with no nucleus (shouldn't happen)
        return []

    syllables = []

    for syll_idx, vowel_pos in enumerate(vowel_positions):
        # Get nucleus
        nucleus_phone = phonemes_with_stress[vowel_pos]
        nucleus = nucleus_phone.phoneme
        stress = nucleus_phone.stress if nucleus_phone.stress is not None else 0

        # Determine onset (consonants before this vowel)
        if syll_idx == 0:
            # First syllable: onset is everything before first vowel
            onset_start = 0
        else:
            # Later syllables: onset starts after previous vowel's coda
            # Use maximal onset principle: assign as many consonants to onset as possible
            prev_vowel_pos = vowel_positions[syll_idx - 1]

            # Find consonants between previous vowel and this vowel
            consonants_between = []
            for i in range(prev_vowel_pos + 1, vowel_pos):
                consonants_between.append(i)

            # Maximal onset: if multiple consonants, split them
            # Simple heuristic: give last consonant(s) to onset, rest to previous coda
            # For now, split in middle (can be improved with phonotactic rules)
            if len(consonants_between) <= 1:
                onset_start = prev_vowel_pos + 1
            else:
                # Split consonant cluster
                split_point = len(consonants_between) // 2
                onset_start = prev_vowel_pos + 1 + split_point

        onset = [phonemes_with_stress[i].phoneme for i in range(onset_start, vowel_pos)]

        # Determine coda (consonants after this vowel, before next vowel)
        if syll_idx == len(vowel_positions) - 1:
            # Last syllable: coda is everything after vowel
            coda_end = len(phonemes_with_stress)
        else:
            # Not last syllable: consonants belong to onset of next syllable
            next_vowel_pos = vowel_positions[syll_idx + 1]

            # Find split point (same logic as onset)
            consonants_between = []
            for i in range(vowel_pos + 1, next_vowel_pos):
                consonants_between.append(i)

            if len(consonants_between) <= 1:
                coda_end = vowel_pos + 1
            else:
                split_point = len(consonants_between) // 2
                coda_end = vowel_pos + 1 + split_point

        coda = [phonemes_with_stress[i].phoneme for i in range(vowel_pos + 1, coda_end)]

        syllables.append(Syllable(
            onset=onset,
            nucleus=nucleus,
            coda=coda,
            stress=stress
        ))

    return syllables


def get_rhyme_part(syllables: List[Syllable]) -> Optional[str]:
    """
    Get rhyme part (nucleus + coda of final stressed syllable).

    For rhyme detection: cat /kæt/ -> /æt/, bat /bæt/ -> /æt/
    """
    if not syllables:
        return None

    # Find last syllable with primary stress, or just use last syllable
    stressed = [s for s in syllables if s.stress == 1]
    target_syll = stressed[-1] if stressed else syllables[-1]

    return target_syll.nucleus + ''.join(target_syll.coda)


def demo():
    """Demo syllabification"""
    from ..embeddings.english_data_loader import EnglishPhonologyLoader
    import io, contextlib

    # Load data
    with contextlib.redirect_stdout(io.StringIO()):
        loader = EnglishPhonologyLoader()

    print("=" * 70)
    print("SYLLABIFICATION DEMO")
    print("=" * 70)

    examples = ['cat', 'bat', 'computer', 'banana', 'photograph', 'photography', 'understand']

    for word in examples:
        if word not in loader.lexicon_with_stress:
            continue

        phonemes = loader.lexicon_with_stress[word]
        syllables = syllabify(phonemes)

        print(f"\n{word}:")
        print(f"  Phonemes: {' '.join(str(p) for p in phonemes)}")
        print(f"  Syllables: {' . '.join(str(s) for s in syllables)}")
        print(f"  Rhyme part: {get_rhyme_part(syllables)}")
        print(f"  Structure:")
        for i, syll in enumerate(syllables):
            stress_marker = "'" if syll.stress == 1 else "ˌ" if syll.stress == 2 else ""
            print(f"    Syllable {i+1}: onset={syll.onset or '∅'}, nucleus={stress_marker}{syll.nucleus}, coda={syll.coda or '∅'}")


if __name__ == '__main__':
    demo()
