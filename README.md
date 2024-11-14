# Film Genre Detection

This repository is part of a Natural Language Processing (NLP) course project aimed at creating an advanced film genre detection system for Persian-language data. This system can perform both document-level and token-level classification for identifying movie genres and named entities. The project implements various approaches, including traditional machine learning, transformers, and BERT-based models.

## Repository Structure

### Code

- **NLP_HW3_Group6_Basic_Doc_Classification.ipynb**  
  Implements basic document-level genre classification using traditional machine learning approaches like logistic regression, SVM, and Naive Bayes with TF-IDF feature extraction.

- **NLP_HW3_Group6_Advanced_Doc_Classification.ipynb**  
  Uses deep learning techniques, including a transformer model (e.g., CLIP and ParsBERT), for document-level genre classification on English and Persian movie descriptions.

- **NLP_HW3_Group6_Basic_Token_Classification.ipynb**  
  Contains code for basic token-level classification for genre-related terms within movie descriptions using traditional models.

- **NLP_HW3_Group6_Advanced_Token_Classification.ipynb**  
  Uses BERT-based models for advanced token-level classification (named entity recognition) on both English and Persian datasets (CoNLL-2003 and Arman).

### Dataset

- **Arman & Peyma Dataset**:  
  Contains labeled datasets for token classification tasks:
  - `arman-labels.txt`, `arman-tokens.txt`: Files for token classification in Persian using the Arman dataset.
  - `peyma-labels.txt`, `peyma-tokens.txt`: Files for token classification in Persian using the Peyma dataset.
  - `dataset.csv` and `dataset.xlsx`: CSV and Excel formats of the compiled dataset for document classification tasks.

### Documentation

- **NLP-HW3-Group6-Documentation.pdf**:  
  Comprehensive documentation that explains dataset details, preprocessing steps, model architectures, evaluation methods, and project goals.

## Overview

This project aims to classify film genres and named entities within movie descriptions written in Persian. It includes both document-level and token-level classification to detect genres and identify specific entities related to film names, locations, and organizations. This multi-part solution leverages both machine learning and deep learning techniques to analyze text data effectively.

### Key Features

- **Document-Level Classification**: Classifies entire movie descriptions into broad genre categories.
- **Token-Level Classification**: Identifies genre-relevant terms and named entities within movie descriptions.
- **Persian-Language Dataset**: Includes Persian datasets, enabling NLP research for an underrepresented language.
- **Evaluation**: Uses metrics like F1-score, accuracy, and confusion matrices to evaluate model performance.

## Installation

### Requirements

This project requires Python 3 and the following libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `torch`
- `transformers`
- `matplotlib`
- `hazm` (for Persian NLP processing)
- `datasets` (for loading and managing datasets)

To install all dependencies, use:

```bash
pip install -r requirements.txt
```

### Additional Setup

1. **Download Datasets**:
   - Ensure the `arman-labels.txt`, `arman-tokens.txt`, `peyma-labels.txt`, `peyma-tokens.txt`, `dataset.csv`, and `dataset.xlsx` files are in the `Dataset` folder.
   
2. **Pre-trained Models**:
   - This project uses CLIP and ParsBERT for advanced document and token classification. These models can be automatically loaded through the Hugging Face `transformers` library.

## Usage

### Document-Level Classification

1. **Basic Document Classification**:
   - Run the `NLP_HW3_Group6_Basic_Doc_Classification.ipynb` notebook or corresponding `.py` script to execute traditional ML-based document-level genre classification.
   
2. **Advanced Document Classification**:
   - Use `NLP_HW3_Group6_Advanced_Doc_Classification.ipynb` or `.py` for transformer-based classification. This script utilizes CLIP for English descriptions and ParsBERT for Persian descriptions.

### Token-Level Classification

1. **Basic Token Classification**:
   - Use `NLP_HW3_Group6_Basic_Token_Classification.ipynb` to perform basic token-level classification, identifying words or phrases indicative of specific genres.

2. **Advanced Token Classification**:
   - Run `NLP_HW3_Group6_Advanced_Token_Classification.ipynb` to utilize BERT models for token-level classification. It can be applied to both English (using CoNLL-2003 dataset) and Persian (using Arman dataset) for named entity recognition.

## Evaluation Metrics

- **Accuracy**: Overall correctness of predictions.
- **F1-Score**: Evaluates the balance between precision and recall.
- **Precision and Recall**: Measures for understanding true positives versus false positives and negatives.
- **Confusion Matrix**: Detailed visualization of model performance across different classes.

## Results

Detailed results of each classification model, including F1, accuracy, and confusion matrices, are provided within the notebooks. The advanced models typically outperform basic models, especially for token classification, due to the deeper contextual understanding offered by transformer-based architectures.

## Future Enhancements

- **Multi-Label Classification**: Modify the document classification to allow multi-genre tagging.
- **Multilingual Support**: Adapt models to work with multilingual datasets for broader application.
- **Real-Time Classification**: Develop an API or web application to enable real-time genre classification.

## License

This project is licensed under the MIT License.

## Acknowledgements

We thank the creators of the Arman and Peyma datasets, as well as the Hugging Face library for their open-source transformer models and tools.

## Contributing

Contributions are welcome. Please submit issues or pull requests to help improve this project.