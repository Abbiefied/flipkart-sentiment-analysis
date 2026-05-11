# Flipkart Product Reviews  Sentiment Analysis

## Overview

This project performs **multi-class sentiment analysis** on 205,000+ Flipkart customer product reviews. Using a combination of a rule-based NLP model (VADER) and a supervised machine learning model (TF-IDF + Logistic Regression), we classify each review as **positive**, **neutral**, or **negative**, and extract insights about what drives customer satisfaction.

---

## Dataset

| Detail | Value |
|---|---|
| Source | [Kaggle - Niralii Vaghani](https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset) |
| Rows | 205,052 |
| Columns | 6 |
| Task type | Supervised multi-class classification |

**Columns:**

| Column | Description |
|---|---|
| `product_name` | Name of the Flipkart product |
| `product_price` | Listed price in INR |
| `Rate` | Customer star rating (1-5) |
| `Review` | Full-text review |
| `Summary` | Short review summary |
| `Sentiment` | Pre-labelled target: `positive`, `neutral`, `negative` |

**Class distribution:**

| Sentiment | Count | Share |
|---|---|---|
| Positive | 166,573 | ~81% |
| Negative | 28,231 | ~14% |
| Neutral | 10,234 | ~5% |

> The dataset is **imbalanced** - the Logistic Regression model uses `class_weight='balanced'` to correct for this.

---

## Methodology

### 1. Data Cleaning
- Dropped rows with non-numeric `Rate` values (corrupt entries).
- Filled missing `Review` text with the `Summary` field.
- Combined `Summary` + `Review` into a single `text` field for richer features.

### 2. Text Preprocessing
- Lowercased all text.
- Removed non-alphabetic characters (punctuation, digits, symbols).
- Removed stopwords using a curated English stopword list.
- Tokenised by whitespace splitting.

### 3. Modelling

Two approaches were built and compared:

#### Rule-Based: VADER
VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon-based tool tuned for social and consumer text. It assigns a **compound score** between −1 and +1, mapped to classes using thresholds:

```
compound ≥  0.05  =  positive
compound ≤ −0.05  =  negative
otherwise         =  neutral
```

#### Supervised ML: TF-IDF + Logistic Regression
- **TF-IDF Vectoriser:** 10,000 features, unigrams + bigrams (`ngram_range=(1,2)`), log-normalised term frequency.
- **Logistic Regression:** `class_weight='balanced'`, `solver='lbfgs'`, `max_iter=1000`.
- **Split:** 80% train / 20% test, stratified by class.

---

## Results

| Model | Accuracy | Notes |
|---|---|---|
| VADER (rule-based) | **90.98%** | Evaluated on 5,000-row sample |
| Logistic Regression (TF-IDF) | **86.39%** | Evaluated on full 41,008-row test set |

### Classification Report - Logistic Regression

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Negative | 0.79 | 0.84 | 0.82 |
| Neutral | 0.27 | 0.70 | 0.39 |
| Positive | 0.98 | 0.88 | 0.93 |
| **Weighted avg** | **0.92** | **0.86** | **0.89** |

> The **neutral class** has the lowest F1 score across both models - neutral language is inherently ambiguous and shares vocabulary with both positive and negative reviews.

---

## Key Findings

- Reviews rated 4–5 stars are almost exclusively labelled **positive**; 1-star reviews are mostly **negative** but contain a notable share of neutral-labelled entries (factual complaints with no emotional language).
- Most predictive **positive** words: `best`, `excellent`, `love`, `quality`, `fast delivery`.
- Most predictive **negative** words: `waste`, `broken`, `return`, `refund`, `fake`, `worst`.
- VADER outperforms the trained classifier on this dataset because the reviews are short and written in everyday conversational English which is precisely the register VADER was designed for.

---

## Setup and Usage

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn wordcloud scikit-learn vaderSentiment
```

### Run the notebook

```bash
git clone https://github.com/Abbiefied/flipkart-sentiment-analysis.git
cd flipkart-sentiment-analysis
jupyter notebook flipkart_sentiment_analysis.ipynb
```

> Make sure `flipkart-reviews.csv` is in the same directory as the notebook before running.

### Predict sentiment on a new review

```python
result = predict_sentiment("Absolutely love this product! Works perfectly and arrived fast.")
print(result)
# {
#   'review': 'Absolutely love this product! Works perfectly and arrived fast.',
#   'VADER': 'positive',
#   'compound': 0.8316,
#   'LR_predicted': 'positive',
#   'LR_negative': '0.3%',
#   'LR_neutral': '0.8%',
#   'LR_positive': '98.9%'
# }
```

---

## Potential Next Steps

- **Better neutral class handling** - oversample neutral reviews or apply probability threshold tuning.
- **Tree-based ensembles** - try LightGBM or XGBoost on the same TF-IDF features.
- **Transformer fine-tuning** - fine-tune `distilBERT` or `afro-xlmr` for state-of-the-art results.
- **Deployment** - wrap the pipeline in a FastAPI endpoint for real-time review scoring.

---

## Acknowledgements

- Dataset by [Niralii Vaghani on Kaggle](https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset).
