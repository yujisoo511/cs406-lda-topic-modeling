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

## Test sample files

Final Approval
- HOLDEN et al v. GUARDIAN ANALYTICS, INC. et al - Entry 62.pdf
- In re Broiler Chicken Antitrust Litigation - Entry 7311.pdf
- Lorenzo Rivera v. Marriott International, Inc. et al - Entry 103.pdf

Voluntary Dismissal
- Jar Capital, LLC v. Auburn Hills, City of - Entry 8.pdf
- Saylor v. Caribou Biosciences, Inc. et al - Entry 33.pdf
- Wilson v. Xerox Holdings Corporation et al - Entry 46.pdf

Preliminary Approval
─ Frasco v. Flo Health, Inc. - Entry 589.pdf
─ Peters v. Aetna Inc. et al - Entry 322.pdf
─ Smart et al v. NCAA - Entry 73.pdf


### Classification output analysis:

final-approval: 100.0% (3/3)
preliminary-approval: 66.7% (2/3)
voluntary-dismissal: 100.0% (3/3)

### Overall Accuracy: 88.9% (8/9)

| Document Name | True Category | Predicted Category | Confidence |
|---------------|---------------|-------------------|------------|
| **Final Approval Documents (3/3 correct)** |
| HOLDEN et al v. GUARDIAN ANALYTICS, INC. et al - Entry 62 | Final Approval | Final Approval | 67.2% |
| In re Broiler Chicken Antitrust Litigation - Entry 7311 | Final Approval | Final Approval | 100.0% |
| Lorenzo Rivera v. Marriott International, Inc. et al - Entry 103 | Final Approval | Final Approval | 66.0% |
| **Preliminary Approval Documents (2/3 correct)** |
| Frasco v. Flo Health, Inc. - Entry 589 | Preliminary Approval | Preliminary Approval | 76.9% |
| Peters v. Aetna Inc. et al - Entry 322 | Preliminary Approval | Final Approval | 80.0% |
| Smart et al v. NCAA - Entry 73 | Preliminary Approval | Preliminary Approval | 66.7% |
| **Voluntary Dismissal Documents (3/3 correct)** |
| Jar Capital, LLC v. Auburn Hills, City of - Entry 8 | Voluntary Dismissal | Voluntary Dismissal | 100.0% |
| Saylor v. Caribou Biosciences, Inc. et al - Entry 33 | Voluntary Dismissal | Voluntary Dismissal | 59.7% |
| Wilson v. Xerox Holdings Corporation et al - Entry 46 | Voluntary Dismissal | Voluntary Dismissal | 65.9% |