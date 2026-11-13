#!/usr/bin/env python3
"""
Quick test script to verify the implementation works.
Run this to test with demo data.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import AspectSentimentPipeline


def test_demo_pipeline():
    """Test the pipeline with demo data."""
    print("Testing Aspect-Based Topic Modeling Pipeline...")
    print("Using demo dataset (100 synthetic reviews)\n")
    
    pipeline = AspectSentimentPipeline(output_dir="results_demo")
    
    try:
        pipeline.run_full_pipeline(
            use_demo=True,
            num_topics=5,
            optimize_lda=False
        )
        print("\n✓ Pipeline test PASSED!")
        return True
    except Exception as e:
        print(f"\n✗ Pipeline test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_demo_pipeline()
    sys.exit(0 if success else 1)
