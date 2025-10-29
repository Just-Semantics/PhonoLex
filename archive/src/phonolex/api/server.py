"""
FastAPI server for PhonoLex.

This module provides a web API for accessing phonological rules and data.
"""

from typing import Dict, List, Optional, Union, Literal
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from seqrule import AbstractObject, check_sequence

from phonolex.rules import (
    create_feature_match_rule,
    create_natural_class_rule,
    create_environment_rule,
    create_assimilation_rule,
    create_syllable_structure_rule,
    create_sonority_rule,
    NATURAL_CLASSES,
    FEATURE_VALUES
)

# Create FastAPI app
app = FastAPI(
    title="PhonoLex API",
    description="API for phonological rules and data processing",
    version="0.1.0"
)

# Define API models
class PhonemeFeatures(BaseModel):
    """Features of a phoneme."""
    ipa: str = Field(..., description="IPA representation of the phoneme")
    features: Dict[str, bool] = Field(..., description="Phonological features")

class RuleType(BaseModel):
    """Type of phonological rule to create."""
    rule_type: Literal[
        "feature_match", 
        "natural_class", 
        "environment", 
        "assimilation",
        "syllable_structure", 
        "sonority"
    ] = Field(..., description="Type of rule to create")
    parameters: Dict = Field(..., description="Parameters for rule creation")

class RuleCheckRequest(BaseModel):
    """Request to check if a sequence satisfies a rule."""
    phonemes: List[PhonemeFeatures] = Field(..., description="Phonemes to check")
    rule: RuleType = Field(..., description="Rule to check against")

class RuleCheckResponse(BaseModel):
    """Response with rule check result."""
    valid: bool = Field(..., description="Whether the sequence satisfies the rule")
    rule_description: str = Field(..., description="Description of the rule")

# API Routes
@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Welcome to PhonoLex API"}

@app.get("/features")
def get_features():
    """Get all possible feature values."""
    return {"features": FEATURE_VALUES}

@app.get("/natural-classes")
def get_natural_classes():
    """Get all defined natural classes."""
    return {"natural_classes": NATURAL_CLASSES}

@app.post("/check-rule", response_model=RuleCheckResponse)
def check_rule(request: RuleCheckRequest):
    """Check if a sequence satisfies a phonological rule."""
    # Convert phonemes to AbstractObjects
    phoneme_objects = []
    for p in request.phonemes:
        obj_dict = {"ipa": p.ipa}
        obj_dict.update(p.features)
        phoneme_objects.append(AbstractObject(**obj_dict))
    
    # Create rule based on type and parameters
    try:
        if request.rule.rule_type == "feature_match":
            features = request.rule.parameters.get("features", {})
            rule = create_feature_match_rule(features)
        
        elif request.rule.rule_type == "natural_class":
            class_name = request.rule.parameters.get("class_name", "")
            rule = create_natural_class_rule(class_name)
        
        elif request.rule.rule_type == "environment":
            target = request.rule.parameters.get("target_features", {})
            left = request.rule.parameters.get("left_context")
            right = request.rule.parameters.get("right_context")
            rule = create_environment_rule(target, left, right)
        
        elif request.rule.rule_type == "assimilation":
            target = request.rule.parameters.get("target_class", "")
            trigger = request.rule.parameters.get("assimilate_to", "")
            feature = request.rule.parameters.get("feature", "")
            direction = request.rule.parameters.get("direction", "progressive")
            rule = create_assimilation_rule(target, trigger, feature, direction)
        
        elif request.rule.rule_type == "syllable_structure":
            patterns = request.rule.parameters.get("allowed_patterns", [])
            rule = create_syllable_structure_rule(patterns)
        
        elif request.rule.rule_type == "sonority":
            scale = request.rule.parameters.get("sonority_scale", {})
            rule = create_sonority_rule(scale)
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown rule type: {request.rule.rule_type}")
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Check the sequence against the rule
    result = check_sequence(phoneme_objects, rule)
    
    return {
        "valid": result,
        "rule_description": rule.description
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("phonolex.api.server:app", host="0.0.0.0", port=8000, reload=True) 