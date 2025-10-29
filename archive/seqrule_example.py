from seqrule import AbstractObject, DSLRule, check_sequence

# Create a very simple sequence of objects
sequence = [
    AbstractObject(value=1, color="red"),
    AbstractObject(value=2, color="blue"),
    AbstractObject(value=3, color="red"),
]

# Define a simple rule function
def alternating_colors(seq):
    """Colors must alternate in the sequence"""
    if len(seq) <= 1:
        return True
    
    for i in range(len(seq)-1):
        if seq[i]["color"] == seq[i+1]["color"]:
            return False
    return True

# Create a DSL rule
rule = DSLRule(alternating_colors, "colors must alternate")

# Check if the sequence satisfies the rule
is_valid = check_sequence(sequence, rule)
print(f"Valid sequence? {is_valid}")  # Should be True

# Try an invalid sequence
invalid_sequence = [
    AbstractObject(value=1, color="red"),
    AbstractObject(value=2, color="red"),  # Same color as previous
    AbstractObject(value=3, color="blue"),
]

is_invalid = check_sequence(invalid_sequence, rule)
print(f"Valid sequence? {is_invalid}")  # Should be False 