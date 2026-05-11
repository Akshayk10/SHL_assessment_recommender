"""
retrieval_rag.py - FAISS-based retrieval with fastembed (low memory)
"""

import json
import numpy as np
import faiss
from pathlib import Path
from fastembed import TextEmbedding
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class SHLRetrievalSystem:
    def __init__(self, catalog_path: str = "data/catalog.json"):
        self.catalog_path = Path(catalog_path)
        self.catalog = []
        self.index = None
        self.model = None
        self.model_name = "BAAI/bge-small-en-v1.5"
        
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
        """Build FAISS index using fastembed (low memory)"""
        if not self.catalog:
            self.load_catalog()
        
        logger.info(f"Initializing fastembed with model: {self.model_name}")
        self.model = TextEmbedding(model_name=self.model_name)
        
        documents = [self.create_document_text(assess) for assess in self.catalog]
        logger.info(f"Creating embeddings for {len(documents)} documents...")
        
        embeddings = []
        for i, doc in enumerate(documents):
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(documents)}")
            embedding = list(self.model.embed([doc]))[0]
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings, dtype='float32')
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        logger.info(f"FAISS index built with {self.index.ntotal} vectors, dimension {dimension}")
        
    def retrieve_relevant(self, query: str, k: int = 8) -> List[Dict]:
        """
        Retrieve top-k relevant assessments for a query.
        """
        if self.index is None:
            self.build_index()
        
        query_embedding = list(self.model.embed([query.lower()]))[0]
        query_embedding = np.array([query_embedding], dtype='float32')
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.catalog):
                assessment = self.catalog[idx].copy()
                similarity = 1 / (1 + distances[0][i])
                assessment['relevance_score'] = float(similarity)
                results.append(assessment)
        
        logger.info(f"Retrieved {len(results)} relevant assessments for query: {query[:50]}")
        return results
    
    def format_for_prompt(self, assessments: List[Dict]) -> str:
        """Format retrieved assessments for LLM prompt"""
        if not assessments:
            return "No relevant assessments found."
        
        lines = []
        for i, assess in enumerate(assessments[:5], 1):
            lines.append(f"{i}. {assess['name']}")
            lines.append(f"   URL: {assess['url']}")
            lines.append(f"   Type: {assess['test_type']} ({assess.get('test_type_label', '')})")
            if assess.get('description'):
                desc = assess['description'][:200]
                lines.append(f"   Description: {desc}")
            if assess.get('duration'):
                lines.append(f"   Duration: {assess['duration']}")
            lines.append("")
        
        return "\n".join(lines)

# Global singleton
_retrieval_system = None

def get_retrieval_system() -> SHLRetrievalSystem:
    """Singleton pattern for retrieval system"""
    global _retrieval_system
    if _retrieval_system is None:
        _retrieval_system = SHLRetrievalSystem()
        _retrieval_system.load_catalog()
    return _retrieval_system

def retrieve_assessments(query: str, k: int = 8) -> List[Dict]:
    """Convenience function to retrieve assessments"""
    system = get_retrieval_system()
    return system.retrieve_relevant(query, k)