from pipeline import AspectSentimentPipeline
import pandas as pd

# Assuming you have a CSV with columns: 'text', 'rating'
# Download from: https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset

pipeline = AspectSentimentPipeline(output_dir="results_electronics")

# Run with real data
pipeline.run_full_pipeline(
    filepath="amazon_electronics_reviews.csv",
    num_topics=15,                    # Identify 15 aspects
    nrows=10000,                      # Use first 10k reviews
    sample_per_rating=1000,           # Balance by rating
    optimize_lda=True                 # Find optimal topic count
)

# Load and explore results
aspect_sentiment_lda = pd.read_csv("results_electronics/lda_aspect_sentiment.csv")
negative_drivers_lda = pd.read_csv("results_electronics/lda_negative_drivers.csv")
negative_drivers_bert = pd.read_csv("results_electronics/bertopic_negative_drivers.csv")

print("=" * 80)
print("ASPECT SENTIMENT ANALYSIS - LDA")
print("=" * 80)
print(aspect_sentiment_lda[['topic_id', 'avg_rating', 'negative_sentiment_score', 'count_docs']])

print("\n" + "=" * 80)
print("TOP 10 NEGATIVE DRIVERS - LDA")
print("=" * 80)
print(negative_drivers_lda.head(10)[['topic_id', 'keywords', 'impact_score', 'avg_rating']])

print("\n" + "=" * 80)
print("TOP 10 NEGATIVE DRIVERS - BERTOPIC")
print("=" * 80)
print(negative_drivers_bert.head(10)[['topic_id', 'keywords', 'impact_score', 'avg_rating']])
