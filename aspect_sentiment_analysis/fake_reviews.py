from pipeline import AspectSentimentPipeline

# Create pipeline
pipeline = AspectSentimentPipeline(output_dir="results_demo")

# Run full analysis with demo data
# This generates 100 synthetic reviews about electronics
pipeline.run_full_pipeline(
    use_demo=True,           # Use synthetic data
    num_topics=5,            # 5 topics
    optimize_lda=False       # Skip optimization for speed
)
