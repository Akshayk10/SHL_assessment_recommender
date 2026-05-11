"""
retrieval_rag.py - FAISS-based retrieval for SHL assessments
"""

import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class SHLRetrievalSystem:
    def __init__(self, catalog_path: str = "data/catalog.json"):
        self.catalog_path = Path(catalog_path)
        self.catalog = []
        self.index = None
        self.model = None
        self.model_name = "all-MiniLM-L6-v2"
        
    def load_catalog(self):
        """Load SHL assessments from catalog.json"""
        if not self.catalog_path.exists():
            raise FileNotFoundError(f"Catalog not found at {self.catalog_path}")
        
        with open(self.catalog_path, 'r', encoding='utf-8') as f:
            self.catalog = json.load(f)
        
        logger.info(f"Loaded {len(self.catalog)} assessments")
        return self.catalog
    
    def create_document_text(self, assessment: Dict) -> str:
        """Create searchable text for each assessment"""
        parts = [
            assessment.get('name', ''),
            assessment.get('test_type_label', ''),
            assessment.get('description', ''),
        ]
        
        if assessment.get('remote_testing'):
            parts.append("remote testing")
        if assessment.get('adaptive'):
            parts.append("adaptive")
        if assessment.get('duration'):
            parts.append(assessment.get('duration'))
        
        return " ".join(parts).lower()
    
    def build_index(self):
        """Build FAISS index"""
        if not self.catalog:
            self.load_catalog()
        
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        documents = [self.create_document_text(assess) for assess in self.catalog]
        logger.info(f"Creating embeddings for {len(documents)} documents...")
        
        embeddings = self.model.encode(documents, show_progress_bar=False)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
        
    def retrieve_relevant(self, query: str, k: int = 8) -> List[Dict]:
        """Retrieve top-k relevant assessments"""
        if self.index is None:
            self.build_index()
        
        query_embedding = self.model.encode([query.lower()])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.catalog):
                assessment = self.catalog[idx].copy()
                # Convert L2 distance to similarity score (0-1)
                assessment['relevance_score'] = float(1 / (1 + distances[0][i]))
                results.append(assessment)
        
        return results

# Global singleton
_retrieval_system = None

def get_retrieval_system() -> SHLRetrievalSystem:
    global _retrieval_system
    if _retrieval_system is None:
        _retrieval_system = SHLRetrievalSystem()
        _retrieval_system.load_catalog()
        _retrieval_system.build_index()
    return _retrieval_system

def retrieve_assessments(query: str, k: int = 8) -> List[Dict]:
    """Convenience function"""
    system = get_retrieval_system()
    return system.retrieve_relevant(query, k)