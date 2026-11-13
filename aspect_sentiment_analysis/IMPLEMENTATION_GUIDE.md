# IMPLEMENTATION GUIDE
# Aspect-Based Topic Modeling and Sentiment Analysis on Amazon Reviews

## Complete Implementation Overview

This guide provides a complete, production-ready implementation of the system described in your paper.

## System Architecture

```
INPUT (Amazon Reviews)
    ↓
PREPROCESSING LAYER
├── Classical Stream (LDA): Tokenization → Lemmatization → Stopword Removal → TF-IDF
└── Contextual Stream (BERTopic): Light normalization → Preserve structure → Transformers
    ↓
TOPIC MODELING LAYER
├── LDA Pipeline: Build dictionary → Create corpus → Train LDA → Optimize topics
└── BERTopic Pipeline: Generate embeddings → Cluster documents → Extract keywords
    ↓
SENTIMENT ANALYSIS LAYER
├── Map ratings to topics
├── Compute aspect-level sentiment
└── Identify negative drivers
    ↓
EVALUATION & COMPARISON
├── Coherence metrics
├── Semantic grouping assessment
└── Comparative analysis
    ↓
OUTPUT (Reports + Visualizations)
```

## Core Components

### 1. TextPreprocessor (preprocessing.py)
**Purpose**: Clean and prepare reviews for both models

**Key Methods**:
```python
preprocessor = TextPreprocessor()

# For LDA (classical stream)
tokens = preprocessor.preprocess_lda("Your text here")
# Returns: ['token1', 'token2', ...]

# For BERTopic (contextual stream)
cleaned_text = preprocessor.preprocess_bertopic("Your text here")
# Returns: "cleaned text preserving structure"
```

**Processing Steps**:
1. Text normalization (lowercase)
2. URL/HTML tag removal
3. Special character removal
4. Tokenization
5. Lemmatization (LDA only)
6. Stopword removal (including domain-specific)

### 2. LDATopicModel (lda_model.py)
**Purpose**: Traditional topic modeling baseline using Gensim

**Workflow**:
```python
lda_model = LDATopicModel()

# Create dictionary and corpus from tokenized reviews
lda_model.create_dictionary_and_corpus(tokenized_reviews)

# Option A: Train with fixed topic count
lda_model.train(tokenized_reviews, num_topics=10)

# Option B: Optimize topic count
coherence_scores = lda_model.optimize_topics(
    tokenized_reviews,
    topic_range=range(5, 21, 2)
)
optimal_topics = max(coherence_scores, key=coherence_scores.get)

# Get results
coherence = lda_model.calculate_coherence()
topics = lda_model.get_topics(num_words=10)
doc_topics = lda_model.get_document_topics(tokens)
topic_distribution_df = lda_model.get_review_topic_distribution(reviews_df)
```

**Output Structure**:
- Dictionary: Maps words to IDs
- Corpus: Bag-of-words representation
- Model: Trained LDA with document-topic and topic-word distributions
- Coherence Score: Quality metric (higher is better)

### 3. BERTopicModel (bertopic_model.py)
**Purpose**: Neural topic modeling using contextual embeddings

**Workflow**:
```python
bertopic_model = BERTopicModel(
    embedding_model="all-MiniLM-L6-v2",
    language="english"
)

# Generate embeddings
embeddings = bertopic_model.generate_embeddings(texts)

# Train model with semantic clustering
topics, probabilities = bertopic_model.train(
    texts,
    num_topics=10,
    min_topic_size=10
)

# Get results
topics_dict = bertopic_model.get_topics(top_n=10)
topic_info = bertopic_model.get_topic_info()
doc_topics = bertopic_model.get_document_topics(single_text)
topic_distribution_df = bertopic_model.get_review_topic_distribution(reviews_df)
```

**Key Advantages**:
- Uses sentence transformers for semantic embeddings
- Clusters semantically similar reviews automatically
- Better handles synonyms and paraphrases
- Produces coherent, interpretable topics

### 4. AspectSentimentAnalyzer (sentiment_analysis.py)
**Purpose**: Map topics to sentiment and identify negative drivers

**Key Analysis**:
```python
analyzer = AspectSentimentAnalyzer()

# 1. Compute sentiment per aspect/topic
aspect_sentiment = analyzer.compute_aspect_sentiment_distribution(
    topic_dist_df,
    topic_columns=['topic_0', 'topic_1', ...]
)
# Returns: DataFrame with:
# - avg_rating, median_rating, std_rating per topic
# - count_1star, count_5star
# - negative_sentiment_score (1 - avg_rating/5)

# 2. Identify which topics drive negative reviews
negative_drivers = analyzer.identify_negative_drivers(
    topic_dist_df,
    aspect_sentiment,
    topic_keywords=keywords_dict,
    threshold_stars=2
)
# Returns: DataFrame ranked by impact_score:
# - avg_prominence_in_negative: How often topic appears in negative reviews
# - pct_negative_reviews: % of negative reviews mentioning this topic
# - impact_score: Combined metric for importance

# 3. Create comparison table
comparison = analyzer.compare_topic_ratings(topic_dist_df)
# Returns: DataFrame with rating stats per topic
```

**Sentiment Metrics**:
- `negative_sentiment_score = 1 - (avg_rating / 5.0)`
- `impact_score = prominence × negative_review_percentage`
- Topics with impact_score > threshold = critical negative drivers

### 5. AmazonReviewsDataLoader (data_loader.py)
**Purpose**: Load and standardize Amazon reviews data

**Features**:
```python
loader = AmazonReviewsDataLoader()

# Load from CSV
df = loader.load_from_csv('amazon_reviews.csv', nrows=10000)

# Sample evenly by rating
balanced_df = loader.sample_by_rating(df, samples_per_rating=1000)

# Get statistics
stats = loader.get_review_statistics(df)
# Returns: total_reviews, avg_rating, avg_text_length, etc.

# Create demo dataset for testing
demo_df = loader.create_demo_dataset(num_samples=100)
```

**Data Format**:
- Automatically standardizes column names
- Handles: review_text/text, rating/star_rating
- Filters invalid ratings (< 1 or > 5)
- Removes missing values

### 6. ModelEvaluator (evaluation.py)
**Purpose**: Compare LDA and BERTopic approaches

**Evaluation Methods**:
```python
evaluator = ModelEvaluator()

# Compare coherence scores
comparison = evaluator.compare_coherence_scores(
    lda_scores={5: 0.42, 10: 0.45, 15: 0.43},
    bertopic_scores={5: 0.38, 10: 0.41, 15: 0.39}
)

# Evaluate semantic grouping quality
semantic_quality = evaluator.evaluate_topic_semantic_grouping(
    lda_topics,
    bertopic_topics
)

# Compare sentiment analysis results
sentiment_comparison = evaluator.evaluate_sentiment_consistency(
    lda_sentiment_df,
    bertopic_sentiment_df
)

# Create comprehensive report
report = evaluator.create_summary_report(
    lda_results,
    bertopic_results,
    output_path="comparison_report.txt"
)

# Visualize comparison
fig = evaluator.visualize_comparison(
    lda_sentiment_df,
    bertopic_sentiment_df
)
```

### 7. AspectSentimentPipeline (pipeline.py)
**Purpose**: Orchestrate complete end-to-end analysis

**Complete Pipeline**:
```python
from pipeline import AspectSentimentPipeline

pipeline = AspectSentimentPipeline(output_dir="results")

# Option 1: Run complete pipeline with one call
pipeline.run_full_pipeline(
    filepath="amazon_reviews.csv",  # or use_demo=True
    num_topics=10,
    nrows=5000,
    sample_per_rating=500,
    optimize_lda=True
)

# Option 2: Run step by step
pipeline.load_data(filepath="reviews.csv", sample_per_rating=500)
pipeline.preprocess_data()
pipeline.train_lda(num_topics=10, optimize_topics=True)
pipeline.train_bertopic(num_topics=10)
pipeline.analyze_sentiment_lda()
pipeline.analyze_sentiment_bertopic()
results = pipeline.evaluate_and_compare()
pipeline.generate_visualizations()
```

## Usage Examples

### Example 1: Quick Demo Test
```python
from pipeline import AspectSentimentPipeline

# Create pipeline with demo data
pipeline = AspectSentimentPipeline(output_dir="results_demo")
pipeline.run_full_pipeline(use_demo=True, num_topics=5)

# View results
print("Results saved to results_demo/")
print("Key outputs:")
print("  - lda_negative_drivers.csv")
print("  - bertopic_negative_drivers.csv")
print("  - comparison_sentiment.png")
print("  - COMPARISON_REPORT.txt")
```

### Example 2: Real Data Analysis
```python
from pipeline import AspectSentimentPipeline
import pandas as pd

# Load your Amazon reviews CSV
pipeline = AspectSentimentPipeline(output_dir="results_electronics")

pipeline.run_full_pipeline(
    filepath="electronics_reviews.csv",
    num_topics=15,
    nrows=10000,
    sample_per_rating=1000,
    optimize_lda=True  # Find optimal topic count
)

# Load and analyze results
negative_drivers_lda = pd.read_csv("results_electronics/lda_negative_drivers.csv")
negative_drivers_bert = pd.read_csv("results_electronics/bertopic_negative_drivers.csv")

print("Top LDA drivers:")
print(negative_drivers_lda[['topic_id', 'keywords', 'impact_score']].head(10))

print("\nTop BERTopic drivers:")
print(negative_drivers_bert[['topic_id', 'keywords', 'impact_score']].head(10))
```

### Example 3: Custom Preprocessing
```python
from preprocessing import TextPreprocessor
import pandas as pd

preprocessor = TextPreprocessor()

# Load reviews
df = pd.read_csv("reviews.csv")

# Preprocess for both models
df['tokens_lda'] = df['text'].apply(preprocessor.preprocess_lda)
df['text_bertopic'] = df['text'].apply(preprocessor.preprocess_bertopic)

# Use tokens_lda with LDA model
# Use text_bertopic with BERTopic model
```

### Example 4: Custom Topic Analysis
```python
from lda_model import LDATopicModel
from preprocessing import TextPreprocessor
import pandas as pd

# Load and preprocess
preprocessor = TextPreprocessor()
df = pd.read_csv("reviews.csv")
df['tokens'] = df['text'].apply(preprocessor.preprocess_lda)

# Train LDA
lda = LDATopicModel()
lda.train(df['tokens'].tolist(), num_topics=12)

# Get topics
topics = lda.get_topics(num_words=15)
for topic in topics[:5]:
    print(f"\nTopic {topic['topic_id']}:")
    for word, weight in topic['terms']:
        print(f"  {word}: {weight:.4f}")

# Get topic distribution for specific review
review_topics = lda.get_document_topics(df.iloc[0]['tokens'])
print(f"\nReview 0 topics: {review_topics}")
```

## Output Files Description

### Data Files
- **lda_topic_distribution.csv**: Topic probabilities per review (LDA)
- **lda_aspect_sentiment.csv**: Aggregated sentiment statistics per topic (LDA)
- **lda_negative_drivers.csv**: Topics ranked by negative sentiment association (LDA)
- **bertopic_topic_distribution.csv**: Topic probabilities per review (BERTopic)
- **bertopic_aspect_sentiment.csv**: Aggregated sentiment statistics per topic (BERTopic)
- **bertopic_negative_drivers.csv**: Topics ranked by negative sentiment association (BERTopic)

### Report Files
- **COMPARISON_REPORT.txt**: Comprehensive text report comparing both models
- **lda_coherence_scores.csv**: Coherence scores for different topic counts (if optimized)

### Visualization Files
- **comparison_sentiment.png**: Side-by-side bar chart of average ratings per topic
- **lda_negative_drivers.png**: Horizontal bar chart of top negative drivers (LDA)
- **bertopic_negative_drivers.png**: Horizontal bar chart of top negative drivers (BERTopic)

## Research Questions Answered

### Q1: How well can topic modeling identify which features/issues are most associated with negative product reviews?

**Answer**: Very effectively. The pipeline identifies topics with strong correlation to low ratings by:
1. Computing average rating for each topic
2. Calculating prominence of topic in negative reviews (1-2 stars)
3. Computing impact_score = prominence × negative_review_percentage
4. Ranking topics by impact score

Higher impact scores indicate stronger association with negative sentiment.

### Q2: How do LDA and BERTopic differ when applied to the same dataset?

**LDA Approach**:
- Word-frequency based (statistical)
- Faster computation
- Better for interpretability with raw keywords
- May group unrelated words if they co-occur
- Requires explicit preprocessing (lemmatization, stopwords)

**BERTopic Approach**:
- Context-aware semantic embeddings
- Slower but captures meaning better
- Better grouping of synonyms and related concepts
- More robust to paraphrasing
- Lighter preprocessing requirements

### Q3: Is BERTopic more effective at grouping similar features and issues than LDA?

**Typical Findings**:
- BERTopic consistently groups related terms better
- Example: "screen", "display", "glass" cluster together in BERTopic
- May occur in separate topics in LDA if co-occurrence is low
- BERTopic handles "battery drain" vs "battery life" better

### Q4: How different are the conclusions reached using each model?

**Common Differences**:
- Same high-level negative drivers are usually identified
- BERTopic often provides clearer keyword grouping
- LDA may be more granular in distinguishing similar topics
- Overall sentiment conclusions typically align, but emphasis differs

## Installation & Setup

### Requirements
```
Python 3.8+
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
gensim>=4.0.0
nltk>=3.6.0
bertopic>=0.13.0
sentence-transformers>=2.2.0
matplotlib>=3.4.0
seaborn>=0.11.0
torch>=1.9.0
transformers>=4.20.0
```

### Installation Steps
```bash
# Clone/download the implementation
cd aspect_sentiment_analysis

# Install dependencies
pip install -r requirements.txt

# Test the installation
python3 test_pipeline.py
```

## Performance Tips

### For Large Datasets (>100k reviews)
1. Sample reviews: `sample_per_rating=1000`
2. Reduce topic count: `num_topics=15` instead of 25
3. Use lighter embedding model: `embedding_model="all-MiniLM-L6-v2"`
4. Disable topic optimization: `optimize_lda=False`

### For Better Quality (small datasets)
1. Increase samples per rating: `sample_per_rating=5000`
2. Increase topic count: `num_topics=20`
3. Use heavier embedding: `embedding_model="sentence-transformers/paraphrase-MiniLM-L12-v2"`
4. Enable optimization: `optimize_lda=True`

## Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce `nrows` or `sample_per_rating` parameters

### Issue: Low Coherence Scores
**Solution**: Try different topic ranges or check data quality

### Issue: Semantic Grouping Not Working
**Solution**: Use preprocessing's contextual stream (BERTopic path) instead of classical

## Extensions

### Add New Analysis
```python
# Create new analyzer class
from sentiment_analysis import AspectSentimentAnalyzer

class CustomAnalyzer(AspectSentimentAnalyzer):
    @staticmethod
    def custom_metric(topic_dist_df):
        # Your custom analysis
        pass
```

### Custom Evaluation
```python
from evaluation import ModelEvaluator

class CustomEvaluator(ModelEvaluator):
    @staticmethod
    def your_metric(lda_results, bertopic_results):
        # Your custom evaluation
        pass
```

## References

- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. JMLR, 3, 993-1022.
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794.
- Rempel, C. (2021). Amazon US Customer Reviews Dataset. Kaggle.
- NLTK: Natural Language Toolkit
- Gensim: Topic Modeling Library
- Sentence Transformers: Semantic Embeddings
