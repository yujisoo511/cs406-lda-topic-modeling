# LDA Topic Modeling - Local Environment Setup

This repository contains a local conversion of the LDA legal document classifier originally developed for AWS SageMaker and Lambda. This converted version runs in a standard Python environment without AWS dependencies.

## Local Environment Setup

### Requirements

- Python 3.12.11

Install all required packages:

```bash
pip install numpy pandas matplotlib seaborn nltk gensim PyMuPDF --quiet
```

### Directory Structure

```
cs406-lda-topic-modeling/
├── train_lda_model.ipynb    # Converted SageMaker notebook
├── lambda_function.py       # Local version of Lambda function
├── lda_model/               # Pre-trained model files
├── input-pdfs/              # I placed sample PDFs here to test classification
└── classification-output/   # Classification results
```

## Running Locally

### Quick Start

1. Ensure you have the pre-trained model in the `lda_model/` directory
2. Create an `input-pdfs` directory if it doesn't exist
3. Place PDF files to classify in the `input-pdfs` directory
4. Run the classifier:

```bash
python lambda_function.py
```

5. Check results in the `classification-output/` directory

### Directory Configuration

To use different input or output directories, modify the constants in `lambda_function.py`:

```python
INPUT_DIR = "input-pdfs"     # Directory containing PDF files to classify
OUTPUT_DIR = "classification-output"  # Directory where results will be saved
```

## Results

It generates a JSON file for each document with classification details:

```json
{
  "doc_id": "document-name",
  "predicted_category": "final-approval",
  "confidence": 0.8562,
  "category_scores": {
    "final-approval": 0.8562,
    "preliminary-approval": 0.1125,
    "voluntary-dismissal": 0.0313
  },
  "top_words": ["settlement", "approval", "final", "fee", "award"]
}
```
