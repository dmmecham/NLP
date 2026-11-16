from pipeline import AspectSentimentPipeline
import kagglehub
import pandas as pd

print("Fetching Amazon Electronics reviews from Kaggle...")
reviews_path = kagglehub.dataset_download(
    "cynthiarempel/amazon-us-customer-reviews-dataset",
    path="amazon_reviews_us_Electronics_v1_00.tsv",
)
print(f"Review data: {reviews_path}")

pipeline = AspectSentimentPipeline(output_dir="results_electronics")

# Run with real data
pipeline.run_full_pipeline(
    filepath=reviews_path,
    use_demo=False,                   # Require the real data, never demo data
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
