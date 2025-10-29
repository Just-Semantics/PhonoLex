#!/usr/bin/env python3
"""
Quick API verification script

Tests all major endpoints to verify backend is working correctly.
Run after starting the API server (python main_new.py)
"""

import requests
import json
import sys
from typing import Dict, Any

API_BASE = "http://localhost:8000"

def test_endpoint(name: str, method: str, url: str, data: Dict[str, Any] = None, params: Dict[str, Any] = None) -> bool:
    """Test a single endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    try:
        if method == "GET":
            response = requests.get(url, params=params, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            print(f"❌ Unknown method: {method}")
            return False

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            print(f"Response preview: {json.dumps(result, indent=2)[:500]}...")
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"❌ Connection failed - is the API running on {API_BASE}?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║          PhonoLex API Verification Test Suite            ║
╚═══════════════════════════════════════════════════════════╝
    """)

    results = []

    # Test 1: Health Check
    results.append(test_endpoint(
        "Health Check",
        "GET",
        f"{API_BASE}/health"
    ))

    # Test 2: Root Endpoint
    results.append(test_endpoint(
        "Root Endpoint",
        "GET",
        f"{API_BASE}/"
    ))

    # Test 3: Get Phoneme by IPA
    results.append(test_endpoint(
        "Get Phoneme /t/",
        "GET",
        f"{API_BASE}/api/phonemes/t"
    ))

    # Test 4: Search Phonemes by Features
    results.append(test_endpoint(
        "Search Phonemes (voiced stops)",
        "POST",
        f"{API_BASE}/api/phonemes/search",
        data={
            "features": {
                "consonantal": "+",
                "periodicGlottalSource": "+",
                "continuant": "-"
            }
        }
    ))

    # Test 5: Compare Phonemes
    results.append(test_endpoint(
        "Compare Phonemes /t/ vs /d/",
        "POST",
        f"{API_BASE}/api/phonemes/compare",
        data={
            "ipa1": "t",
            "ipa2": "d"
        }
    ))

    # Test 6: Get Word
    results.append(test_endpoint(
        "Get Word 'cat'",
        "GET",
        f"{API_BASE}/api/words/cat"
    ))

    # Test 7: Find Similar Words
    results.append(test_endpoint(
        "Find Similar Words to 'cat'",
        "GET",
        f"{API_BASE}/api/similarity/word/cat",
        params={"threshold": 0.85, "limit": 20}
    ))

    # Test 8: Builder (Pattern Matching)
    results.append(test_endpoint(
        "Builder: Words starting with /b/, ending with /t/",
        "POST",
        f"{API_BASE}/api/builder/generate",
        data={
            "patterns": [
                {"type": "STARTS_WITH", "phoneme": "b"},
                {"type": "ENDS_WITH", "phoneme": "t"}
            ],
            "properties": {
                "syllable_count": [1, 2],
                "wcm_score": [0, 5]
            },
            "limit": 50
        }
    ))

    # Test 9: Minimal Pairs
    results.append(test_endpoint(
        "Quick Tools: Minimal Pairs /t/ vs /d/",
        "POST",
        f"{API_BASE}/api/quick-tools/minimal-pairs",
        data={
            "phoneme1": "t",
            "phoneme2": "d",
            "word_length": "short",
            "complexity": "low",
            "limit": 30
        }
    ))

    # Test 10: Rhyme Set
    results.append(test_endpoint(
        "Quick Tools: Rhyme Set for 'cat'",
        "POST",
        f"{API_BASE}/api/quick-tools/rhyme-set",
        data={
            "target_word": "cat",
            "perfect_only": True,
            "limit": 50
        }
    ))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")

    if passed == total:
        print("\n✅ All tests passed! API is working correctly.")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
