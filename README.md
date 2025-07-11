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

### 📓 [Homework 1: Text Classification with Bag-of-Words](./homework 1.ipynb)
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

### 📓 [Homework 2: Word Embeddings and Language Models](./homework 2.ipynb)
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

### 📓 [Homework 3: Phrase Mining and Advanced NLP](./homework 3.ipynb)
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

### 📓 [Baseline Implementation](./baseline.ipynb)
- Traditional NLP pipeline
- Word2Vec embeddings
- Simple logistic regression
- Basic text preprocessing

### 📓 [Advanced Implementation](./data challenge.ipynb)
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

## 🔬 Final Project: Image Classification via Prototyping

**Research Question**: "Can we reduce the size of image datasets for training without sacrificing model accuracy?"

### 🎯 Project Objective
This project explores image classification through the lens of data prototyping, aiming to reduce the complexity of training datasets while maintaining accuracy. By leveraging techniques like random sampling and K-Means clustering for prototype selection, we evaluate how well smaller, representative subsets can substitute full datasets in model training.

The project addresses a fundamental challenge in machine learning: the trade-off between computational efficiency and model performance. Traditional approaches require large datasets for training, but this research investigates whether intelligent data selection can achieve comparable results with significantly reduced computational requirements.

### 📊 Datasets Analyzed

| Dataset | Description | Classes | Training Samples | Test Samples | Performance |
|---------|-------------|---------|------------------|--------------|-------------|
| **MNIST** | Handwritten digits (0-9) | 10 | 60,000 | 10,000 | ~95% accuracy |
| **EMNIST** | Extended handwritten characters | 62 | 124,800 | 20,800 | ~85% accuracy |
| **KMNIST** | Japanese Hiragana characters | 49 | 232,365 | 38,547 | ~83% accuracy |

**Dataset Characteristics**:
- **MNIST**: Classic benchmark dataset with simple digit recognition
- **EMNIST**: Extended version with both letters and digits, higher complexity
- **KMNIST**: Japanese Hiragana characters, most complex due to cultural and linguistic differences

### ⚙️ Prototyping Methods

#### 🔹 Random Sampling
- Selects a random subset of the training data
- Fast and easy implementation with minimal computational overhead
- May not preserve class balance or underlying data structure
- Serves as baseline comparison for more sophisticated methods
- Particularly effective for simpler datasets with clear class boundaries

#### 🔹 K-Means Clustering
- Clusters images and selects centroids as prototypes
- Provides more structured and representative subsets
- Preserves underlying data distribution better than random sampling
- Variants tested:
  - **Standard K-Means** with random initialization
  - **K-Means++** with improved initialization for better convergence

**Clustering Advantages**:
- Better representation of data manifold
- Improved class balance preservation
- More robust to outliers and noise
- Captures underlying data structure

### 🧪 Experimental Design

#### Prototype Selection Strategies
1. **Random Prototyping**: Random subset selection with varying sizes (1,000 to 75,000 samples)
2. **K-Means Clustering**: Centroid-based prototype selection with different cluster counts
3. **K-Means++**: Improved initialization for better clustering quality and convergence

#### Performance Analysis Framework
- **K-Nearest Neighbors**: Primary classification method for prototype-based learning
- **Varying K values**: Systematic testing from k=1 to k=31 for optimal performance
- **Statistical significance**: Multiple runs (10-30 per configuration) for robust results
- **Cross-dataset comparison**: Performance analysis across different complexity levels
- **Computational efficiency**: Training time and memory usage measurements

#### Evaluation Metrics
- **Accuracy**: Overall classification performance
- **Balanced Accuracy**: For datasets with class imbalance
- **Statistical Analysis**: Mean, standard deviation, and confidence intervals
- **Scalability Analysis**: Performance vs. computational cost trade-offs

### 📈 Key Findings

#### Dataset Complexity Impact
- **MNIST (10 classes)**: Easiest dataset, achieves highest accuracy (~95%)
  - Simple digit shapes with clear boundaries
  - Consistent writing styles across samples
  - Well-balanced class distribution
- **EMNIST (26 classes)**: Medium complexity, moderate accuracy (~85%)
  - Increased class count and character complexity
  - Mixed case letters with varying styles
  - More challenging feature space
- **KMNIST (49 classes)**: Most complex, lowest accuracy (~83%)
  - Japanese Hiragana characters with cultural variations
  - Complex stroke patterns and character similarities
  - Highest class count and feature complexity

#### Prototype Scaling Effects
- **More prototypes = Better performance** (up to a point)
  - Diminishing returns observed beyond certain thresholds
  - Optimal prototype count varies by dataset complexity
  - Performance plateaus indicate saturation points
- **Diminishing returns**: Large increases in prototypes yield smaller accuracy gains
  - MNIST: Saturation around 10,000 prototypes
  - EMNIST: Saturation around 50,000 prototypes
  - KMNIST: Saturation around 75,000 prototypes
- **Optimal trade-off**: Balance between accuracy and computational cost
  - 80% of maximum accuracy achievable with 20% of full dataset
  - Significant computational savings with minimal performance loss
- **Dataset-dependent scaling**: Different datasets require different prototype counts
  - Simpler datasets need fewer prototypes
  - Complex datasets benefit from larger prototype sets

#### Clustering vs Random Selection
- **K-Means++**: Generally outperforms standard K-Means
  - Better initialization leads to more representative centroids
  - Improved convergence and stability
  - More consistent performance across runs
- **Random sampling**: Effective baseline, especially for simpler datasets
  - Surprisingly competitive for MNIST and simple cases
  - Lower computational overhead
  - Good starting point for comparison
- **Class balance**: Clustering methods better preserve class distribution
  - Important for imbalanced datasets like KMNIST
  - Reduces bias towards majority classes
  - More robust performance across all classes

#### K-Nearest Neighbors Analysis
- **Optimal K values**: Vary significantly by dataset and prototype method
  - MNIST: k=1 often optimal due to clear class boundaries
  - EMNIST: k=3-5 optimal for noise reduction
  - KMNIST: k=5-7 optimal for complex character recognition
- **K sensitivity**: Higher K values can help with noisy data
  - Majority voting reduces impact of outliers
  - Trade-off between noise reduction and boundary precision
- **Method interaction**: Optimal K depends on prototype selection method
  - Clustering methods often work better with higher K
  - Random sampling may prefer lower K values

### 🎨 Visual Results and Analysis

The project includes comprehensive visualizations demonstrating:

#### Prototype Image Samples
- **Random prototypes**: Showcase typical training samples
- **K-Means centroids**: Reveal representative cluster centers
- **K-Means++ centroids**: Demonstrate improved cluster quality
- **Cross-dataset comparison**: Visual differences between datasets

#### Performance Curves
- **Accuracy vs. Prototype Count**: Shows scaling behavior
- **Accuracy vs. K value**: Reveals optimal K for each method
- **Method comparison**: Direct comparison of random vs. clustering
- **Statistical confidence**: Error bars and confidence intervals

#### Dataset-Specific Analysis
- **MNIST results**: Clean, high-accuracy performance curves
- **EMNIST results**: Moderate complexity with good scalability
- **KMNIST results**: Challenging dataset with lower but stable performance

### 🔬 Advanced Experiments

#### CNN Comparison with Lipschitz Regularization
- **Neural network baseline**: Traditional CNN performance
- **Lipschitz regularization**: Smoothness constraints for better generalization
- **Prototype efficiency**: Comparison with prototype-based methods
- **Computational trade-offs**: Training time vs. performance analysis

#### Statistical Significance Testing
- **Multiple runs**: 10-30 iterations per configuration
- **Confidence intervals**: 95% confidence bounds on results
- **Hypothesis testing**: Statistical significance of method differences
- **Robustness analysis**: Performance stability across runs

### 📁 Final Project Structure

The final project is organized into three main components:

#### Core Implementation Files
- **📓 [mnist.ipynb](./final_project/mnist.ipynb)**: Complete MNIST prototyping and classification analysis
- **📓 [emnist.ipynb](./final_project/emnist.ipynb)**: EMNIST dataset with extended character recognition
- **📓 [kmnist.ipynb](./final_project/kmnist.ipynb)**: Japanese Hiragana character classification

#### Documentation and Analysis
- **LaTeX Report**: Complete academic documentation with methodology
- **Visualization Directory**: Comprehensive figures and performance charts
- **Statistical Analysis**: Detailed results and significance testing

#### Data and Resources
- **Dataset Management**: Efficient loading and preprocessing
- **Prototype Storage**: Optimized data structures for large-scale experiments
- **Result Caching**: Performance optimization for repeated experiments

### 🎓 Academic Contributions

#### Research Methodology
- **Systematic experimentation**: Rigorous testing across multiple datasets
- **Statistical validation**: Robust analysis with confidence intervals
- **Cross-dataset comparison**: Generalizability across different domains
- **Computational efficiency**: Practical considerations for real-world applications

#### Novel Insights
- **Prototype scaling laws**: Quantitative relationships between dataset size and performance
- **Method effectiveness**: Comparative analysis of different prototype selection strategies
- **Dataset complexity impact**: Understanding of how dataset characteristics affect results
- **Practical guidelines**: Recommendations for prototype-based learning applications

#### Broader Implications
- **Efficient machine learning**: Reducing computational requirements
- **Scalable algorithms**: Methods that work with large datasets
- **Cross-cultural applications**: Understanding different writing systems
- **Educational value**: Demonstrating fundamental ML concepts

### 👤 My Contribution to Final Project

I contributed to the design and implementation of the prototype-based classification framework, including:
- **Data prototyping logic**: Development of random sampling and K-Means clustering strategies
- **Statistical evaluation design**: Implementation of robust evaluation metrics and confidence intervals
- **KNN classification implementation**: Optimization across all three datasets (MNIST, EMNIST, KMNIST)
- **Cross-dataset analysis**: Comparative performance evaluation and scalability testing
- **Visualization and reporting**: Creation of comprehensive performance charts and statistical analysis

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

## 📚 Academic Report

The complete academic report for the final project includes comprehensive methodology documentation, detailed experimental results, statistical analysis, and visualizations. The report demonstrates rigorous research methodology and provides insights into prototype-based learning for image classification tasks.
