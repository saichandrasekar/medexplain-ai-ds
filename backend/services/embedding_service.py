import os


from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_tokenizer = None
_model = None

def _load_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        # Load the BGE model and tokenizer
        model_name = "BAAI/bge-base-en-v1.5"
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModel.from_pretrained(model_name)
        _model.eval()

def get_embeddings(texts, batch_size=32):
    """
    Generate embeddings for a list of text strings.
    
    Args:
        texts: List of text strings (documents or queries)
        batch_size: Number of texts to process in each batch (default: 32)
    
    Returns:
        numpy array of embeddings, shape (len(texts), embedding_dim)
    """
    all_embeddings = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize the batch
        encoded_input = _tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt',
            max_length=512
        )
        
        # Generate embeddings
        with torch.no_grad():
            model_output = _model(**encoded_input)
            # Use CLS token embeddings (first token)
            embeddings = model_output[0][:, 0]
        
        # Normalize embeddings (recommended for BGE models)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Convert to numpy and add to results
        all_embeddings.append(embeddings.cpu().numpy())
    
    # Concatenate all batches
    return np.vstack(all_embeddings)