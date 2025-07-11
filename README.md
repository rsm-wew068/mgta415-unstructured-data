# MGTA415 - Unstructured Data Analysis

This repository contains the complete coursework for MGTA415, covering natural language processing (NLP), machine learning, and deep learning techniques for analyzing unstructured data.

## 📚 Course Overview

This course explores various techniques for processing and analyzing unstructured data, including text classification, word embeddings, language models, and prototype-based learning. The coursework progresses from traditional NLP methods to advanced deep learning approaches.

## 🗂️ Project Structure

```
mgta415-unstructured-data/
├── homework 1.ipynb          # Text Classification with Bag-of-Words
├── homework 2.ipynb          # Word Embeddings and Language Models
├── homework 3.ipynb          # Phrase Mining and Advanced NLP
├── baseline.ipynb            # Restaurant Classification Baseline
├── data challenge.ipynb      # Advanced Restaurant Classification
├── final_project/
│   ├── mnist.ipynb          # MNIST Digit Classification
│   ├── emnist.ipynb         # EMNIST Letter Classification
│   └── kmnist.ipynb         # KMNIST Japanese Character Classification
└── README.md
```

## 📖 Homework Assignments

### Homework 1: Text Classification with Bag-of-Words
**Dataset**: Financial Phrase Bank (FPB) - Sentiment Analysis

**Key Concepts**:
- Text preprocessing (tokenization, stemming, stop word removal)
- Document representation techniques:
  - Binary vectors (presence/absence)
  - Frequency vectors (word counts)
  - TF-IDF vectors (term frequency × inverse document frequency)
- Logistic regression classification
- Evaluation metrics: AUROC, Macro-F1, Micro-F1

**Results**: Achieved ~80% accuracy across different representations

### Homework 2: Word Embeddings and Language Models
**Dataset**: Financial Phrase Bank (FPB) - Sentiment Analysis

**Key Concepts**:
- **Word Embeddings**:
  - Word2Vec vs GloVe comparison
  - Sentence similarity using averaged word vectors
  - Classification with word embeddings
- **Language Models**:
  - N-gram models (unigram, bigram)
  - Zero probability problem and smoothing techniques
  - Probability calculations for unseen sequences

**Results**: Word2Vec achieved ~89% AUROC, GloVe achieved ~85% AUROC

### Homework 3: Phrase Mining and Advanced NLP
**Dataset**: Computer Science research papers

**Key Concepts**:
- **Phrase Mining**:
  - AutoPhrase algorithm for phrase extraction
  - Phrase scoring and anomaly detection
  - Text segmentation techniques
- **Advanced NLP**:
  - Word2Vec training on phrase-segmented text
  - Domain-specific similarity analysis
  - Computer science terminology understanding

**Results**: Identified meaningful technical phrases and domain relationships

## 🏆 Data Challenge: Restaurant Classification

**Problem**: Multi-class classification of restaurant types based on customer reviews

### Baseline Implementation (`baseline.ipynb`)
- Traditional NLP pipeline
- Word2Vec embeddings
- Simple logistic regression
- Basic text preprocessing

### Advanced Implementation (`data challenge.ipynb`)
- **Enhanced Features**:
  - TF-IDF with n-grams (1-3 word combinations)
  - 40,000+ features
  - Domain-specific preprocessing
- **Advanced Techniques**:
  - Hyperparameter optimization (Grid Search)
  - Cross-validation (5-fold stratified)
  - Class balancing for imbalanced data
  - Ensemble methods

**Results**: Achieved ~81% F1 score with advanced techniques

## 🔬 Final Project: Prototype-Based Classification Research

**Research Question**: "How does the number of training prototypes affect the performance of nearest neighbor classification on different handwritten character datasets?"

### Datasets Analyzed

#### 1. MNIST (`mnist.ipynb`)
- **Content**: Handwritten digits (0-9)
- **Size**: 60,000 training, 10,000 test samples
- **Classes**: 10
- **Performance**: ~95% accuracy with 10,000 prototypes

#### 2. EMNIST (`emnist.ipynb`)
- **Content**: Handwritten letters (A-Z)
- **Size**: 124,800 training, 20,800 test samples
- **Classes**: 26
- **Performance**: ~85% accuracy with 75,000 prototypes

#### 3. KMNIST (`kmnist.ipynb`)
- **Content**: Japanese Hiragana characters
- **Size**: 232,365 training, 38,547 test samples
- **Classes**: 49
- **Performance**: ~83% accuracy with 125,000 prototypes

### Key Findings
- **Dataset Complexity**: Performance decreases with increasing class complexity
- **Prototype Scaling**: More prototypes improve performance with diminishing returns
- **Optimal Trade-offs**: Balance between accuracy and computational cost
- **K-NN Analysis**: Optimal k varies by dataset characteristics

## 🛠️ Technologies Used

- **Python**: Primary programming language
- **PyTorch**: Deep learning framework
- **scikit-learn**: Machine learning algorithms
- **NLTK**: Natural language processing
- **Gensim**: Word embeddings (Word2Vec, GloVe)
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization

## 📊 Key Results Summary

| Project | Dataset | Best Performance | Key Technique |
|---------|---------|------------------|---------------|
| HW1 | FPB Sentiment | ~80% F1 | TF-IDF + Logistic Regression |
| HW2 | FPB Sentiment | ~89% AUROC | Word2Vec + Averaged Embeddings |
| HW3 | CS Papers | Domain Analysis | AutoPhrase + Phrase Embeddings |
| Data Challenge | Restaurant Reviews | ~81% F1 | TF-IDF + N-grams + Optimization |
| Final Project | MNIST | ~95% Accuracy | Prototype-based K-NN |
| Final Project | EMNIST | ~85% Accuracy | Prototype-based K-NN |
| Final Project | KMNIST | ~83% Accuracy | Prototype-based K-NN |

## 🎯 Learning Outcomes

### Technical Skills
- **NLP Fundamentals**: Text preprocessing, tokenization, stemming
- **Document Representation**: Bag-of-words, TF-IDF, word embeddings
- **Machine Learning**: Classification, clustering, hyperparameter tuning
- **Deep Learning**: Neural networks, PyTorch implementation
- **Research Methods**: Experimental design, statistical analysis

### Domain Knowledge
- **Text Classification**: Sentiment analysis, document categorization
- **Character Recognition**: Handwritten digit/letter/character classification
- **Cross-cultural Applications**: Different writing systems and languages
- **Prototype-based Learning**: K-nearest neighbors, clustering methods

## 📝 Usage Instructions

1. **Environment Setup**: Install required packages from `requirements.txt`
2. **Data Preparation**: Download datasets (MNIST, EMNIST, KMNIST, FPB)
3. **Execution**: Run notebooks in order for progressive learning
4. **Experimentation**: Modify parameters and analyze results

## 🤝 Contributing

This is a course project repository. For questions or improvements, please contact the course instructor.

## 📄 License

This project is for educational purposes as part of MGTA415 coursework.

---

**Course**: MGTA415 - Unstructured Data Analysis  
**Institution**: [University Name]  
**Semester**: [Semester/Year]