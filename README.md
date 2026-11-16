# Aspect-Based Topic Modeling and Sentiment Analysis

Implementation of the system described in "Aspect-Based Topic Modeling and Sentiment Analysis on Amazon Reviews" by Matt Robinson and Drew Mecham.

## Overview

This project compares two approaches for identifying product aspects/features in customer reviews and determining their association with product ratings:

1. **LDA (Latent Dirichlet Allocation)** - Traditional topic modeling based on word frequency
2. **BERTopic** - Modern approach using semantic embeddings and deep learning

## Project Structure

```
aspect_sentiment_analysis/
├── preprocessing.py          # Text preprocessing pipeline
├── lda_model.py             # LDA implementation and training
├── bertopic_model.py        # BERTopic implementation
├── sentiment_analysis.py    # Aspect-level sentiment analysis
├── data_loader.py           # Data loading and utilities
├── evaluation.py            # Model comparison and evaluation
├── pipeline.py              # End-to-end pipeline orchestrator
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Key Features

### 1. Preprocessing Pipeline
- **Classical Stream (for LDA)**: Tokenization, lemmatization, stopword removal, TF-IDF vectorization
- **Contextual Stream (for BERTopic)**: Light normalization preserving sentence structure for transformers

### 2. Topic Modeling
- **LDA**: Train on term frequencies, optimize topic count using coherence metrics
- **BERTopic**: Generate sentence embeddings, cluster semantically similar reviews, extract keywords
- BERTopic uses KMeans: each review receives one topic, not a probability distribution.
  Its output topic columns are one-hot assignments (1 for the assigned topic, 0 otherwise).
  `train()` returns `None` for probabilities, and `min_topic_size` does not apply to KMeans.

### 3. Sentiment & Aspect Synthesis
- Map review star ratings to extracted topics
- Compute aspect-level sentiment distribution
- Identify features tied to low ratings (1-2 stars)
- Calculate impact scores for negative drivers

### 4. Evaluation
- Coherence scores for topic quality
- Semantic grouping assessment
- Comparative analysis of both approaches
- Comprehensive reports and visualizations

## Installation

```bash
pip install -r requirements.txt
```

NLTK data is separate from the Python packages. On first use, the
preprocessor downloads missing resources automatically. If that fails,
install them manually with:

```bash
python -m nltk.downloader stopwords wordnet omw-1.4 punkt
```

NLTK can use these resources either extracted or in their downloaded ZIP
archives.

## Quick Start

### Using Demo Data

```python
from pipeline import AspectSentimentPipeline

pipeline = AspectSentimentPipeline(output_dir="results")
pipeline.run_full_pipeline(use_demo=True, num_topics=5)
```

### Using Real Data

```python
pipeline = AspectSentimentPipeline(output_dir="results")
pipeline.run_full_pipeline(
    filepath="path/to/amazon_reviews.csv",
    num_topics=10,
    nrows=5000,
    sample_per_rating=500,
    optimize_lda=True
)
```

### Step-by-Step Usage

```python
from pipeline import AspectSentimentPipeline

pipeline = AspectSentimentPipeline(output_dir="results")

# 1. Load data
pipeline.load_data(use_demo=True)

# 2. Preprocess
pipeline.preprocess_data()

# 3. Train both models
pipeline.train_lda(num_topics=10, optimize_topics=True)
pipeline.train_bertopic(num_topics=10)

# 4. Analyze sentiment
pipeline.analyze_sentiment_lda()
pipeline.analyze_sentiment_bertopic()

# 5. Compare and evaluate
results = pipeline.evaluate_and_compare()

# 6. Generate visualizations
pipeline.generate_visualizations()
```

## Input Data Format

CSV file with columns:
- `text` (or `review_text`): Review text
- `rating` (or `star_rating`): Rating from 1-5

Example:
```csv
text,rating
"This product is amazing!",5
"Battery drains too quickly",1
```

## Output Files

- `lda_coherence_scores.csv` - Coherence scores for different topic counts
- `lda_topic_distribution.csv` - Topic probabilities for each review (LDA)
- `lda_aspect_sentiment.csv` - Average ratings per topic (LDA)
- `lda_negative_drivers.csv` - Topics ranked by association with negative reviews
- `bertopic_topic_distribution.csv` - One-hot topic assignments for each review (BERTopic)
- `bertopic_aspect_sentiment.csv` - Average ratings per topic (BERTopic)
- `bertopic_negative_drivers.csv` - Topics ranked by association with negative reviews
- `COMPARISON_REPORT.txt` - Detailed comparison of both approaches
- `comparison_sentiment.png` - Visual comparison of topic sentiment
- `lda_negative_drivers.png` - LDA's top negative drivers
- `bertopic_negative_drivers.png` - BERTopic's top negative drivers

## Key Classes

### TextPreprocessor
Handles all text preprocessing:
- `preprocess_lda()` - Prepare text for LDA
- `preprocess_bertopic()` - Prepare text for BERTopic
- `process_reviews_df()` - Batch processing

### LDATopicModel
Traditional topic modeling:
- `train()` - Train LDA model
- `optimize_topics()` - Find optimal number of topics
- `calculate_coherence()` - Get coherence score
- `get_document_topics()` - Get topic distribution for a review

### BERTopicModel
Neural topic modeling:
- `generate_embeddings()` - Create sentence embeddings
- `train()` - Train BERTopic model
- `get_topics()` - Get topics with keywords
- `reduce_topics()` - Reduce number of topics

### AspectSentimentAnalyzer
Sentiment analysis:
- `compute_aspect_sentiment_distribution()` - Calculate ratings per topic
- `identify_negative_drivers()` - Find topics associated with negative reviews
- `visualize_topic_sentiment()` - Create visualizations

### ModelEvaluator
Comparison tools:
- `compare_coherence_scores()` - Compare model quality metrics
- `evaluate_topic_semantic_grouping()` - Assess semantic quality
- `create_summary_report()` - Generate comparison report
- `visualize_comparison()` - Create comparison visualizations

## Research Questions Addressed

1. **How well can topic modeling identify which features/issues are most associated with negative product reviews?**
   - Both models identify topics correlated with low ratings
   - Impact scores quantify the association strength

2. **How do LDA and BERTopic differ when applied to the same dataset?**
   - LDA: Word-frequency based, faster, interpretable
   - BERTopic: Context-aware, semantic understanding, better grouping of similar aspects

3. **Is BERTopic more effective at grouping similar features and issues than LDA?**
   - BERTopic typically groups semantically similar terms better
   - Better handles synonyms (e.g., "screen", "display", "glass")

4. **How different are the conclusions reached using each model?**
   - Comparison report shows differences in identified aspects and their ratings

## References

- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. *JMLR*, 3, 993-1022.
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv preprint arXiv:2203.05794*.
- Rempel, C. (2021). Amazon US Customer Reviews Dataset. Kaggle.

## License

This implementation is for research purposes.
