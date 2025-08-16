#!/usr/bin/env python3
"""
Script to enhance code comments throughout the TensorFlow tutorial project.

This script identifies areas where more detailed tutorial-style comments
would be beneficial and suggests improvements.
"""

import os
import re
from pathlib import Path


def analyze_comment_density(file_path):
    """Analyze the comment density and educational value of a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    
    total_lines = len(lines)
    comment_lines = 0
    code_lines = 0
    tutorial_patterns = 0
    
    # Patterns that indicate tutorial-style comments
    tutorial_indicators = [
        r'STEP \d+:',
        r'# \d+\.',
        r'# Why:',
        r'# Note:',
        r'# Important:',
        r'# Example:',
        r'# This demonstrates',
        r'# Key concept:',
        r'# Mathematical',
        r'# In practice',
    ]
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            comment_lines += 1
            # Check for tutorial-style patterns
            for pattern in tutorial_indicators:
                if re.search(pattern, line, re.IGNORECASE):
                    tutorial_patterns += 1
                    break
        elif stripped and not stripped.startswith('"""') and not stripped.startswith("'''"):
            code_lines += 1
    
    if code_lines == 0:
        return None
    
    comment_ratio = comment_lines / code_lines
    tutorial_score = tutorial_patterns / comment_lines if comment_lines > 0 else 0
    
    return {
        'file': file_path,
        'total_lines': total_lines,
        'comment_lines': comment_lines,
        'code_lines': code_lines,
        'comment_ratio': comment_ratio,
        'tutorial_patterns': tutorial_patterns,
        'tutorial_score': tutorial_score
    }


def find_functions_needing_comments(file_path):
    """Find functions that could benefit from more detailed comments."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    # Find function definitions
    function_pattern = r'def\s+(\w+)\s*\([^)]*\):'
    functions = re.findall(function_pattern, content)
    
    needs_improvement = []
    
    for func_name in functions:
        # Look for the function and analyze its comments
        func_pattern = rf'def\s+{func_name}\s*\([^)]*\):(.*?)(?=\ndef|\nclass|\Z)'
        match = re.search(func_pattern, content, re.DOTALL)
        
        if match:
            func_body = match.group(1)
            
            # Count meaningful comments in function body
            comment_count = len(re.findall(r'^\s*#[^#]', func_body, re.MULTILINE))
            step_comments = len(re.findall(r'STEP \d+:', func_body))
            code_lines = len([line for line in func_body.split('\n') 
                             if line.strip() and not line.strip().startswith('#')])
            
            # Criteria for needing improvement
            if code_lines > 10 and comment_count < 5:
                needs_improvement.append({
                    'function': func_name,
                    'code_lines': code_lines,
                    'comments': comment_count,
                    'step_comments': step_comments,
                    'suggestion': 'Add more step-by-step comments'
                })
            elif step_comments == 0 and code_lines > 5:
                needs_improvement.append({
                    'function': func_name,
                    'code_lines': code_lines,
                    'comments': comment_count,
                    'step_comments': step_comments,
                    'suggestion': 'Add STEP-by-STEP tutorial structure'
                })
    
    return needs_improvement


def generate_comment_report():
    """Generate a comprehensive report on comment quality across the project."""
    src_dir = Path('src')
    if not src_dir.exists():
        print("Error: 'src' directory not found. Run this script from the project root.")
        return
    
    print("=== TensorFlow Tutorial Comment Analysis ===\n")
    
    # Analyze all Python files
    python_files = list(src_dir.rglob('*.py'))
    analyses = []
    
    for file_path in python_files:
        if file_path.name == '__init__.py':
            continue
        
        analysis = analyze_comment_density(file_path)
        if analysis:
            analyses.append(analysis)
    
    # Sort by comment ratio
    analyses.sort(key=lambda x: x['comment_ratio'])
    
    print("COMMENT DENSITY REPORT")
    print("=" * 50)
    print(f"{'File':<40} {'Comments':<10} {'Ratio':<8} {'Tutorial':<8}")
    print("-" * 70)
    
    for analysis in analyses:
        file_name = str(analysis['file']).replace('src/', '')
        ratio = f"{analysis['comment_ratio']:.2f}"
        tutorial = f"{analysis['tutorial_score']:.2f}"
        
        print(f"{file_name:<40} {analysis['comment_lines']:<10} {ratio:<8} {tutorial:<8}")
    
    # Find files needing improvement
    print(f"\nFILES NEEDING MORE TUTORIAL COMMENTS")
    print("=" * 50)
    
    low_comment_files = [a for a in analyses if a['comment_ratio'] < 0.3 or a['tutorial_score'] < 0.1]
    
    if not low_comment_files:
        print("All files have good comment coverage!")
    else:
        for analysis in low_comment_files:
            file_name = str(analysis['file']).replace('src/', '')
            print(f"{file_name}")
            print(f"   Current ratio: {analysis['comment_ratio']:.2f} (target: >0.3)")
            print(f"   Tutorial score: {analysis['tutorial_score']:.2f} (target: >0.1)")
            
            # Analyze specific functions
            improvements = find_functions_needing_comments(analysis['file'])
            if improvements:
                print("   Functions needing improvement:")
                for imp in improvements[:3]:  # Show top 3
                    print(f"   - {imp['function']}(): {imp['suggestion']}")
            print()
    
    # Overall statistics
    total_comments = sum(a['comment_lines'] for a in analyses)
    total_code = sum(a['code_lines'] for a in analyses)
    avg_ratio = total_comments / total_code if total_code > 0 else 0
    
    print(f"OVERALL STATISTICS")
    print("=" * 50)
    print(f"Total comment lines: {total_comments}")
    print(f"Total code lines: {total_code}")
    print(f"Overall comment ratio: {avg_ratio:.2f}")
    
    if avg_ratio >= 0.4:
        print("Excellent tutorial comment coverage!")
    elif avg_ratio >= 0.3:
        print("Good tutorial comment coverage!")
    elif avg_ratio >= 0.2:
        print("Adequate comment coverage, could be improved")
    else:
        print("Needs significant comment improvement for tutorial purposes")


def suggest_comment_improvements():
    """Suggest specific comment improvements for tutorial purposes."""
    print(f"\nTUTORIAL COMMENT IMPROVEMENT SUGGESTIONS")
    print("=" * 60)
    
    suggestions = [
        {
            'area': 'Function Headers',
            'current': '"""Brief description."""',
            'improved': '''"""
            Detailed explanation of what this function teaches.
            
            Include why this concept is important, what the user will learn,
            and how it relates to broader TensorFlow/ML concepts.
            """'''
        },
        {
            'area': 'Code Blocks',
            'current': '# Create model',
            'improved': '''# STEP 1: Create the model architecture
            # Sequential models stack layers linearly - data flows input → output
            # This is the simplest way to build neural networks in Keras'''
        },
        {
            'area': 'Parameter Explanations',
            'current': 'layers.Dense(64, activation="relu")',
            'improved': '''layers.Dense(
                64,                    # 64 neurons in this layer
                activation="relu"      # ReLU activation: f(x) = max(0, x)
                                      # Prevents vanishing gradients
            )'''
        },
        {
            'area': 'Mathematical Context',
            'current': 'gradient = tape.gradient(loss, weights)',
            'improved': '''# Compute gradients: ∂loss/∂weights
            # This tells us how to change weights to reduce loss
            # Gradient descent: weights = weights - learning_rate * gradients
            gradient = tape.gradient(loss, weights)'''
        }
    ]
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion['area']}")
        print("   Current style:")
        print(f"   {suggestion['current']}")
        print("   Tutorial style:")
        for line in suggestion['improved'].strip().split('\n'):
            print(f"   {line}")
        print()


if __name__ == "__main__":
    generate_comment_report()
    suggest_comment_improvements()