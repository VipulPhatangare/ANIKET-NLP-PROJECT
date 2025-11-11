# NLP Mini Project: Product Review Analysis

## 📌 Project Overview
This project performs comprehensive Natural Language Processing (NLP) analysis on product reviews for the **BeMinimalist Salicylic + LHA 2% Cleanser**. The analysis includes web scraping, text preprocessing, sentiment analysis, topic modeling, and question-answering capabilities.

## 🎯 Objectives
- Extract customer reviews from product website
- Perform multilingual text preprocessing and translation
- Analyze sentiment and identify key themes
- Extract semantic relationships and product features
- Generate automated Q&A responses based on review insights

## 📂 Project Structure
```
NLP Mini Project/
├── NLP PROJECT.ipynb                                    # Main analysis notebook
├── salicylic_lha_cleanser_reviews.csv                  # Raw scraped reviews
├── salicylic_lha_cleanser_reviews_translated.csv       # Translated reviews
├── salicylic_lha_cleanser_reviews_cleaned.csv          # Cleaned and normalized reviews
├── salicylic_lha_cleanser_reviews_ner_tfidf.csv        # NER and TF-IDF features
├── salicylic_lha_cleanser_spacy_sentiment.csv          # Sentiment analysis results
├── lsa_topics_keywords.csv                              # LSA topic modeling output
├── lsa_document_topics.csv                              # Document-topic distributions
├── vector_semantics_similarity.csv                      # Word similarity analysis
├── review_summary_similarity_index.csv                  # Clustered review summaries
└── salicylic_cleanser_QA_summary.csv                   # Generated Q&A pairs
```

## 🔧 Technologies & Libraries Used

### Core Libraries
- **Selenium & BeautifulSoup4**: Web scraping and HTML parsing
- **spaCy**: NLP pipeline (POS tagging, NER, lemmatization)
- **pandas**: Data manipulation and analysis
- **scikit-learn**: TF-IDF, clustering, dimensionality reduction

### Additional Tools
- **langdetect**: Language identification
- **deep-translator (LibreTranslate)**: Multilingual translation
- **gensim**: Word2Vec embeddings and semantic modeling
- **numpy**: Numerical computations

## 📊 Analysis Pipeline

### Phase 1: Data Acquisition & Preprocessing

#### 1. Review Extraction
- **Source**: BeMinimalist product page (Yotpo review widget)
- **Method**: Selenium-based dynamic web scraping with pagination
- **Data Collected**: 
  - Customer name and date
  - Star rating (1-5)
  - Review title and text
- **Output**: `salicylic_lha_cleanser_reviews.csv`

#### 2. Language Processing
- **Language Detection**: Automatic identification using langdetect
- **Translation**: Non-English reviews translated to English via LibreTranslate API
- **Supported Languages**: Hindi, and other regional languages
- **Output**: `salicylic_lha_cleanser_reviews_translated.csv`

#### 3. Data Cleaning & Normalization
- **Encoding Fixes**: UTF-8 normalization
- **Noise Removal**: HTML tags, URLs, special characters
- **Text Processing**:
  - Case normalization (lowercase)
  - Tokenization (word and sentence level)
  - Stop word removal
  - Lemmatization using spaCy
- **Duplicate Removal**: Exact match detection
- **Output**: `salicylic_lha_cleanser_reviews_cleaned.csv`

### Phase 2: Syntactic & Semantic Analysis

#### 4. Part-of-Speech (POS) Tagging
- **Model**: spaCy en_core_web_sm
- **Insights**:
  - Overall POS distribution analysis
  - Top 20 most common adjectives (product descriptors)
  - Key descriptive patterns

#### 5. Named Entity Recognition (NER)
- **Entities Extracted**: Organizations, persons, dates, product components
- **Entity Frequency Analysis**: Most mentioned entities across reviews
- **Output**: `salicylic_lha_cleanser_reviews_ner_tfidf.csv`

#### 6. Vector Representations
- **Bag-of-Words (BoW)**: Basic frequency representation
- **TF-IDF**: Term importance weighting (5000 features)
- **Word2Vec**: 
  - Vector size: 100
  - Window: 5
  - Skip-gram model
  - Semantic similarity for key terms (skin, acne, cleanser, gentle, oil)

#### 7. Sentiment Analysis
- **Method**: Lexicon-based with spaCy
- **Lexicons**: 
  - Positive words (good, great, amazing, effective, etc.)
  - Negative words (bad, irritating, breakout, expensive, etc.)
- **Features**:
  - Sentiment score (continuous)
  - Sentiment label (positive/neutral/negative)
  - POS-weighted scoring (adjectives, adverbs, verbs)
- **Insights**:
  - Sentiment distribution
  - Common positive/negative adjectives
  - Key sentiment-driving phrases
- **Output**: `salicylic_lha_cleanser_spacy_sentiment.csv`

#### 8. Topic Modeling
- **Algorithm**: Latent Semantic Analysis (LSA) with TruncatedSVD
- **Topics**: 3-5 automatically determined themes
- **Features**:
  - Top 10 keywords per topic
  - Top 3 representative documents per topic
  - Document-topic distributions
- **Output**: 
  - `lsa_topics_keywords.csv`
  - `lsa_document_topics.csv`

### Phase 3: Advanced Analysis & Application

#### 9. Vector Semantics & Similarity
- **Word2Vec Similarities**: Semantic relationships for product features
- **TF-IDF Cosine Similarity**: Document-level similarity matrix
- **Feature Analysis**: skin, acne, cleanser, gentle, oil
- **Output**: `vector_semantics_similarity.csv`

#### 10. Review Summarization
- **Method**: Clustering + Similarity-based selection
- **Algorithm**: KMeans clustering (2-5 clusters)
- **Selection**: Top 3 most representative reviews per cluster
- **Metric**: Cosine similarity to cluster centroid
- **Output**: `review_summary_similarity_index.csv`

#### 11. Question-Answering System
- **Approach**: Feature-based + Sentiment-driven
- **Questions Answered**:
  1. Does the cleanser effectively reduce acne?
  2. Is it gentle or does it cause dryness/irritation?
  3. Does it help control oil and sebum?
  4. How is the texture and fragrance?
  5. Is it worth the price?
- **Answer Generation**: Statistical analysis + sentiment percentages + feature frequency
- **Output**: `salicylic_cleanser_QA_summary.csv`

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Chrome browser (for web scraping)

### Install Dependencies
```bash
pip install selenium beautifulsoup4 pandas webdriver-manager
pip install spacy langdetect deep-translator
pip install scikit-learn gensim numpy

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### Run the Notebook
```bash
jupyter notebook "NLP PROJECT.ipynb"
```

## 📈 Key Findings

### Sentiment Distribution
- **Positive**: ~70-80% of reviews
- **Neutral**: ~10-15%
- **Negative**: ~5-15%

### Most Common Positive Adjectives
- good, great, amazing, effective, gentle, best, clean, soft

### Most Common Negative Adjectives
- dry, expensive, irritating, hard, rough

### Top Product Features Mentioned
- Acne control and reduction
- Gentle formulation
- Oil control
- Texture and consistency
- Price and value

### Identified Topics
1. **Skin improvement & acne treatment**
2. **Product texture & application**
3. **Oil control & cleansing**
4. **Value for money & purchase decision**
5. **Side effects & sensitivity**

## 📊 Results & Outputs

All analysis results are saved as CSV files:
- **Raw Data**: Original and translated reviews
- **Processed Data**: Cleaned, tokenized, and normalized text
- **Analysis Results**: Sentiment scores, topic distributions, entity mentions
- **Summaries**: Representative reviews and Q&A pairs

## 🔍 Use Cases

1. **Product Development**: Identify features to improve
2. **Marketing Insights**: Understand customer pain points and satisfaction drivers
3. **Customer Support**: Auto-generate FAQ responses
4. **Competitive Analysis**: Compare sentiment across products
5. **Quality Assurance**: Track sentiment trends over time

## 📝 Future Enhancements

- [ ] Integrate transformer models (BERT, RoBERTa) for better sentiment analysis
- [ ] Implement aspect-based sentiment analysis
- [ ] Add time-series sentiment tracking
- [ ] Create interactive dashboard for visualization
- [ ] Expand to multi-product comparison
- [ ] Implement neural topic modeling (BERTopic)

## 👥 Contributors
- **Aniket** - Project Lead & Analysis

## 📄 License
This project is for educational purposes.

## 🙏 Acknowledgments
- BeMinimalist for product data
- spaCy and scikit-learn communities
- LibreTranslate for translation services

---

**Note**: This project demonstrates end-to-end NLP pipeline implementation without relying on pre-trained transformer models, focusing on classical NLP techniques and libraries like spaCy, scikit-learn, and gensim.
