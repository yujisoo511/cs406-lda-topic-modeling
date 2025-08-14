import os
import json
import logging
from typing import Dict, Any, List, Tuple

import fitz  # PyMuPDF
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel

# Try to download necessary NLTK resources silently
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

# ---------- Configuration ----------
# Directories
INPUT_DIR = "input-pdfs"
OUTPUT_DIR = "classification-output"
MODEL_DIR = "lda_model"

# Model files
DICT_PATH = os.path.join(MODEL_DIR, "dictionary.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "lda_model")
TOPIC_MAP_PATH = os.path.join(MODEL_DIR, "topic_to_category.json")
PARAMS_PATH = os.path.join(MODEL_DIR, "model_params.json")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("simple-classifier")

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    log.info(f"Extracting text from {pdf_path}")
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        log.error(f"Error extracting text from {pdf_path}: {e}")
    return text

def get_topic_top_words(lda_model: LdaModel, topic_id: int, n: int = 10) -> List[str]:
    """Get the top words for a specific topic."""
    words = lda_model.show_topic(topic_id, n)
    return [word for word, _ in words]

def extract_document_header(text: str, max_chars: int = 3000) -> str:
    """Extract the beginning portion of the document."""
    return text[:max_chars]

def process_document_text(doc_text: str, dictionary: Dictionary, min_token_length: int) -> List:
    """Minimal processing of document text to produce bag-of-words for the model."""
    # Use only the document header (first part) for better focus
    header_text = extract_document_header(doc_text)
    
    # Simple tokenization and filtering
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(header_text.lower())
    
    # Minimal stopwords for common English words
    stopwords = set([
        "the", "and", "a", "to", "of", "in", "that", "is", "for", "on", "it", 
        "with", "as", "be", "this", "by", "are", "at", "or", "not", "an", "from", 
        "but", "which", "they", "their", "have", "has", "had", "will", "would"
    ])
    
    # Legal-specific stopwords
    legal_stopwords = set([
        "plaintiff", "plaintiffs", "defendant", "defendants", "v", "vs", "versus",
        "et", "al", "page", "court", "district", "united", "states", "judge",
        "case", "document", "exhibit", "section"
    ])
    
    stopwords.update(legal_stopwords)
    
    # Filter tokens
    tokens = [token for token in tokens 
              if token not in stopwords and len(token) >= min_token_length]
    
    # Basic lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Convert to bag-of-words using the dictionary
    bow = dictionary.doc2bow(tokens)
    
    return bow

def classify_document(pdf_path: str, dictionary: Dictionary, lda_model: LdaModel, 
                      topic_to_category: Dict, min_token_length: int) -> Dict[str, Any]:
    """Classify a document using the pre-trained LDA model with minimal processing."""
    # Extract document ID from filename
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Extract text from PDF
    doc_text = extract_text_from_pdf(pdf_path)
    text_preview = doc_text[:100] + "..." if len(doc_text) > 100 else doc_text
    log.info(f"Extracted {len(doc_text)} characters from {pdf_path}")
    log.debug(f"Text preview: {text_preview}")
    
    # Convert document to bag-of-words
    bow = process_document_text(doc_text, dictionary, min_token_length)
    
    # Get topic distribution
    topic_dist = lda_model.get_document_topics(bow)
    top_topics = sorted(topic_dist, key=lambda x: x[1], reverse=True)[:3]
    
    # Get top words for the main topic
    top_words = []
    if top_topics:
        top_topic_id = top_topics[0][0]
        top_words = get_topic_top_words(lda_model, top_topic_id)
    
    # Determine document category based on topics
    category_scores = {"final-approval": 0, "preliminary-approval": 0, "voluntary-dismissal": 0}
    total_prob = 0
    
    for topic_idx, prob in top_topics:
        if str(topic_idx) in topic_to_category:
            category = topic_to_category[str(topic_idx)]
            category_scores[category] += prob
            total_prob += prob
    
    # Get most likely category
    if total_prob > 0:
        for category in category_scores:
            category_scores[category] /= total_prob
            
    predicted_category = max(category_scores.items(), key=lambda x: x[1])[0]
    confidence = category_scores[predicted_category]
    
    # Create result object
    result = {
        "doc_id": doc_id,
        "predicted_category": predicted_category,
        "confidence": float(confidence),
        "category_scores": {k: float(v) for k, v in category_scores.items()},
        "top_words": top_words
    }
    
    return result

def load_models() -> Tuple[Dictionary, LdaModel, Dict, Dict]:
    """Load the dictionary, LDA model, topic map, and parameters."""
    log.info(f"Loading models from {MODEL_DIR}")
    
    # Load dictionary
    dictionary = Dictionary.load(DICT_PATH)
    
    # Load LDA model
    lda_model = LdaModel.load(MODEL_PATH)
    
    # Load topic-to-category mapping
    with open(TOPIC_MAP_PATH, 'r') as f:
        topic_to_category = json.load(f)
    
    # Load model parameters
    with open(PARAMS_PATH, 'r') as f:
        model_params = json.load(f)
    
    return dictionary, lda_model, topic_to_category, model_params

def main():
    """Main function to process all PDFs in the input directory."""
    # Use fixed directories without command-line options
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure input directory exists
    if not os.path.isdir(input_dir):
        log.error(f"Input directory {input_dir} does not exist or is not a directory")
        return
    
    # Load models and parameters
    try:
        dictionary, lda_model, topic_to_category, model_params = load_models()
        min_token_length = model_params.get('min_token_length', 3)
    except Exception as e:
        log.error(f"Failed to load models: {e}")
        return
    
    # Find all PDF files in the input directory and subdirectories
    pdf_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    if not pdf_files:
        log.warning(f"No PDF files found in {input_dir}")
        return
    
    log.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF file
    for pdf_path in pdf_files:
        try:
            # Classify the document
            result = classify_document(pdf_path, dictionary, lda_model, topic_to_category, min_token_length)
            
            # Write result to output directory
            doc_id = result["doc_id"]
            output_path = os.path.join(output_dir, f"{doc_id}_result.json")
            
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
                
            log.info(f"Classified {pdf_path} as {result['predicted_category']} (confidence: {result['confidence']:.4f})")
            log.info(f"Results written to {output_path}")
            
        except Exception as e:
            log.error(f"Error processing {pdf_path}: {e}")
    
    log.info("Document classification complete")

if __name__ == "__main__":
    main()
