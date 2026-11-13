#!/usr/bin/env python3
"""
EXAMPLE USAGE & EXPECTED OUTPUTS
Complete walkthrough of the Aspect-Based Topic Modeling system
"""

# ============================================================================
# EXAMPLE 1: QUICK START WITH DEMO DATA (No external data needed)
# ============================================================================

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

# EXPECTED OUTPUT FLOW:
# ================
# ================================================================================
# STEP 1: LOADING DATA
# ================================================================================
# Creating demo dataset...
# Loaded 100 reviews
# Rating distribution:
#  5    25
#  4    20
#  3    15
#  2    20
#  1    20
#
# Dataset Statistics:
#   total_reviews: 100
#   avg_rating: 3.0
#   average_text_length: 45.2
#   average_word_count: 8.5
#   rating_distribution: {5: 25, 4: 20, ...}
#
# ================================================================================
# STEP 2: PREPROCESSING DATA
# ================================================================================
# Preprocessing reviews...
# Sample preprocessed review:
#   Original: This phone is amazing! Fast processor, beautiful display, and excellent battery life.
#   Tokens: ['phone', 'amazing', 'fast', 'processor', 'beautiful', 'display', 'excellent', 'battery', 'life']
#   Cleaned: This phone is amazing! Fast processor, beautiful display, and excellent battery life.
#
# ================================================================================
# STEP 3: TRAINING LDA MODEL (Classical Stream)
# ================================================================================
# Dictionary created: 85 terms
# Corpus created: 100 documents
# Training LDA with 5 topics...
# LDA training complete!
# Coherence Score: 0.4523
#
# Top 3 topics from LDA:
#
#   Topic 0:
#     battery: 0.0156
#     drain: 0.0142
#     life: 0.0138
#     ...
#
#   Topic 1:
#     screen: 0.0189
#     display: 0.0167
#     pixel: 0.0145
#     ...
#
#   Topic 2:
#     processor: 0.0201
#     fast: 0.0168
#     slow: 0.0134
#     ...
#
# ================================================================================
# STEP 4: TRAINING BERTOPIC MODEL (Contextual Stream)
# ================================================================================
# Generating embeddings using all-MiniLM-L6-v2...
# Generated 100 embeddings of dimension 384
#
# Training BERTopic with 5 topics...
# BERTopic training complete!
#
# Top 3 topics from BERTopic:
#
#   Topic 0:
#     screen: 0.3422
#     display: 0.2918
#     glass: 0.1845
#     bright: 0.1612
#     ...
#
#   Topic 1:
#     battery: 0.4156
#     charge: 0.3234
#     drain: 0.2891
#     life: 0.2145
#     ...
#
#   Topic 2:
#     processor: 0.3845
#     fast: 0.3121
#     slow: 0.2634
#     lag: 0.1956
#     ...
#
# ================================================================================
# STEP 5: SENTIMENT ANALYSIS - LDA
# ================================================================================
# Generating topic distributions for all reviews...
#
# Top 5 negative drivers (LDA):
#    topic_id  keywords           impact_score  avg_rating
#    0         battery, drain     12.456        2.3
#    2         screen, cracked    10.234        2.5
#    1         processor, slow    8.123         2.7
#    3         build, quality     6.890         2.9
#    4         camera, photo      5.123         3.2
#
# ================================================================================
# STEP 6: SENTIMENT ANALYSIS - BERTOPIC
# ================================================================================
# Generating topic distributions for all reviews...
#
# Top 5 negative drivers (BERTopic):
#    topic_id  keywords                impact_score  avg_rating
#    0         battery, charge, drain  13.567        2.2
#    1         screen, display, glass  11.234        2.4
#    2         processor, fast, lag    9.456         2.6
#    3         build, quality, durable 7.234         2.8
#    4         camera, photo, video    6.123         3.1
#
# ================================================================================
# STEP 7: EVALUATION AND COMPARISON
# ================================================================================
# ================================================================================
# ASPECT-BASED TOPIC MODELING AND SENTIMENT ANALYSIS - COMPARISON REPORT
# ================================================================================
#
# DATASET STATISTICS
# ...
# [Full report text]
# ...
#
# ================================================================================
# STEP 8: GENERATING VISUALIZATIONS
# ================================================================================
# Saved comparison visualization
# Saved negative drivers visualizations
#
# ================================================================================
# PIPELINE COMPLETE!
# Results saved to: results_demo
# ================================================================================


# ============================================================================
# EXAMPLE 2: ANALYZE REAL AMAZON REVIEWS
# ============================================================================

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

# EXPECTED RESULTS FOR ELECTRONICS REVIEWS:
# ======================================
# Common negative drivers found:
# - Battery issues (quick drain, capacity)
# - Screen/Display problems (cracks, dead pixels, brightness)
# - Performance issues (lag, slow processor)
# - Build quality (durability, materials)
# - Heating problems (overheating during use)
#
# BERTopic typically shows:
# - Better semantic grouping of synonyms
# - Clearer aspect identification
# - More coherent keyword clusters
#
# LDA shows:
# - Detailed word frequency patterns
# - Potentially more granular topic separation
# - Faster processing


# ============================================================================
# EXAMPLE 3: STEP-BY-STEP ANALYSIS WITH CUSTOM PARAMETERS
# ============================================================================

from pipeline import AspectSentimentPipeline
from preprocessing import TextPreprocessor
from lda_model import LDATopicModel
from sentiment_analysis import AspectSentimentAnalyzer

pipeline = AspectSentimentPipeline(output_dir="results_custom")

# Step 1: Load data
print("Step 1: Loading reviews...")
pipeline.load_data(
    filepath="amazon_reviews.csv",
    nrows=5000,
    sample_per_rating=500
)

# Step 2: Preprocess
print("Step 2: Preprocessing...")
pipeline.preprocess_data()

# Step 3: Train LDA with optimization
print("Step 3: Training LDA with topic optimization...")
coherence_scores = pipeline.train_lda(
    num_topics=10,
    optimize_topics=True  # Find optimal count from 5-20
)
# This will print something like:
# Optimal topic count: 12 (score: 0.4523)

# Step 4: Train BERTopic
print("Step 4: Training BERTopic...")
pipeline.train_bertopic(num_topics=12)  # Match LDA's optimal count

# Step 5: Analyze sentiment
print("Step 5: Analyzing sentiment...")
lda_sentiment = pipeline.analyze_sentiment_lda()
bert_sentiment = pipeline.analyze_sentiment_bertopic()

# Step 6: Get detailed comparison
results = pipeline.evaluate_and_compare()

# EXPECTED OUTPUT STRUCTURE:
# ========================
# results = {
#     'lda': {
#         'aspect_sentiment': DataFrame with columns [topic_id, avg_rating, ...],
#         'negative_drivers': DataFrame with columns [topic_id, keywords, impact_score, ...],
#         'dataset_stats': {total_reviews, avg_rating, ...},
#         'optimal_topics': 12
#     },
#     'bertopic': {
#         'aspect_sentiment': DataFrame with columns [topic_id, avg_rating, ...],
#         'negative_drivers': DataFrame with columns [topic_id, keywords, impact_score, ...],
#         'num_topics': 12
#     }
# }

# Step 7: Manual analysis
print("\nManual Analysis Results:")
print("-" * 80)

# Compare average ratings per topic
print("\nTopic Rating Comparison:")
print("LDA Topics:")
print(results['lda']['aspect_sentiment'][['topic_id', 'avg_rating']].head())
print("\nBERTopic Topics:")
print(results['bertopic']['aspect_sentiment'][['topic_id', 'avg_rating']].head())

# Identify biggest negative drivers
print("\nBiggest Negative Driver (LDA):")
top_driver = results['lda']['negative_drivers'].iloc[0]
print(f"  Topic: {top_driver['topic_id']}")
print(f"  Keywords: {top_driver['keywords']}")
print(f"  Impact Score: {top_driver['impact_score']:.4f}")
print(f"  Avg Rating: {top_driver['avg_rating']:.2f}")

print("\nBiggest Negative Driver (BERTopic):")
top_driver = results['bertopic']['negative_drivers'].iloc[0]
print(f"  Topic: {top_driver['topic_id']}")
print(f"  Keywords: {top_driver['keywords']}")
print(f"  Impact Score: {top_driver['impact_score']:.4f}")
print(f"  Avg Rating: {top_driver['avg_rating']:.2f}")

# Step 8: Visualizations
print("Step 8: Generating visualizations...")
pipeline.generate_visualizations()

print("\n" + "=" * 80)
print("Analysis Complete!")
print(f"Results saved to: results_custom/")
print("Key files:")
print("  - lda_negative_drivers.csv")
print("  - bertopic_negative_drivers.csv")
print("  - comparison_sentiment.png")
print("  - COMPARISON_REPORT.txt")
print("=" * 80)


# ============================================================================
# EXAMPLE 4: EXTRACT INSIGHTS FROM SPECIFIC ASPECTS
# ============================================================================

import pandas as pd
from data_loader import AmazonReviewsDataLoader

# Load the comparison results
lda_drivers = pd.read_csv("results_electronics/lda_negative_drivers.csv")
lda_dist = pd.read_csv("results_electronics/lda_topic_distribution.csv")

# Find reviews dominated by a specific negative topic
top_negative_topic = lda_drivers.iloc[0]['topic_id']
print(f"Analyzing Topic {top_negative_topic}: {lda_drivers.iloc[0]['keywords']}")

# Get reviews where this topic is dominant
topic_col = f'topic_{int(top_negative_topic)}'
critical_reviews = lda_dist[lda_dist[topic_col] > 0.3]
print(f"Found {len(critical_reviews)} reviews dominated by this topic")

# Filter to 1-star reviews
one_star_critical = critical_reviews[critical_reviews['rating'] == 1]
print(f"  Of which {len(one_star_critical)} are 1-star reviews")

# EXAMPLE INSIGHTS:
# ================
# Analyzing Topic 2: battery, drain, power, life, charge
# Found 245 reviews dominated by this topic
#   Of which 187 are 1-star reviews
#
# This means 76% of critical reviews mentioning battery issues give 1-star ratings
# --> Battery is a critical factor in customer satisfaction for this product


# ============================================================================
# EXAMPLE 5: CUSTOM PREPROCESSING FOR DOMAIN-SPECIFIC ANALYSIS
# ============================================================================

from preprocessing import TextPreprocessor
import pandas as pd

preprocessor = TextPreprocessor()

# Add electronics-specific stopwords
preprocessor.stop_words.update([
    'device', 'product', 'phone', 'tablet',  # Generic device words
    'year', 'month', 'day', 'time'           # Time expressions
])

# Load and preprocess reviews
df = pd.read_csv("amazon_reviews.csv")

# Custom preprocessing
df['tokens_lda'] = df['text'].apply(preprocessor.preprocess_lda)
df['text_bert'] = df['text'].apply(preprocessor.preprocess_bertopic)

print("Sample preprocessing:")
print(f"Original: {df['text'].iloc[0]}")
print(f"Tokens: {df['tokens_lda'].iloc[0]}")
print(f"BERTopic: {df['text_bert'].iloc[0]}")

# Now use with models
lda_model = LDATopicModel()
lda_model.train(df['tokens_lda'].tolist(), num_topics=10)


# ============================================================================
# EXPECTED OUTPUT FILES STRUCTURE
# ============================================================================

"""
results_demo/
├── lda_coherence_scores.csv                  # Coherence for topic counts 5,7,9,...19
├── lda_topic_distribution.csv                # Shape: (100, 6) [5 topics + review_id + rating]
├── lda_aspect_sentiment.csv                  # Shape: (5, 9) [topic stats]
├── lda_negative_drivers.csv                  # Ranked by impact_score
│
├── bertopic_topic_distribution.csv           # Shape: (100, 6)
├── bertopic_aspect_sentiment.csv             # Shape: (5, 9)
├── bertopic_negative_drivers.csv             # Ranked by impact_score
│
├── comparison_sentiment.png                  # Side-by-side bar charts
├── lda_negative_drivers.png                  # Horizontal bar chart
├── bertopic_negative_drivers.png             # Horizontal bar chart
│
└── COMPARISON_REPORT.txt                     # Full text report

CSV FILE FORMATS:
=================

lda_topic_distribution.csv:
  topic_0, topic_1, topic_2, ..., review_id, rating
  0.234   0.156   0.123         0          5
  0.145   0.287   0.098         1          4
  ...

lda_aspect_sentiment.csv:
  topic_id, avg_rating, median_rating, std_rating, count_docs, count_1star, count_5star, pct_1star, pct_5star, negative_sentiment_score
  0         2.3        2.0             1.2        24        12         2           50.0     8.3       0.54
  1         3.8        4.0             0.8        22        3          8           13.6     36.4      0.24
  ...

lda_negative_drivers.csv:
  topic_id, keywords, avg_prominence_in_negative, pct_negative_reviews, avg_rating, count_negative_mentions, impact_score
  0         battery,...  0.342                     45.2                 2.3        98                 15.47
  2         screen,...   0.298                     42.1                 2.5        91                 12.55
  ...

TEXT REPORT (COMPARISON_REPORT.txt):
====================================
Includes:
  - Dataset statistics
  - LDA results (optimal topics, coherence)
  - BERTopic results
  - Sentiment analysis comparison
  - Key findings
  - Recommendations

PNG VISUALIZATIONS:
===================
- comparison_sentiment.png: 2-panel bar chart showing avg rating per topic
- lda_negative_drivers.png: Top N topics ranked by impact score
- bertopic_negative_drivers.png: Top N topics ranked by impact score
"""

print(__doc__)
