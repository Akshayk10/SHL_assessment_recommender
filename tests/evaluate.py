"""
evaluate.py - Run comprehensive evaluation of the agent
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import requests

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation_metrics import (
    RetrievalMetrics, GroundednessMetrics, 
    EffectivenessMetrics, RecommendationRelevance
)
from retrieval_rag import retrieve_assessments
from retrieval import get_valid_urls

# Test queries with ground truth (expected relevant assessments)
TEST_QUERIES = [
    {
        "query": "Java developer with 4 years experience",
        "relevant": ["Java 8 (New)", ".NET Framework 4.5", "Spring Framework"],
        "role": "Java developer",
        "expected_refusal": False
    },
    {
        "query": "Hiring a project manager for IT projects",
        "relevant": ["Project Management", "Agile Software Development", "Leadership Assessment"],
        "role": "project manager",
        "expected_refusal": False
    },
    {
        "query": "What questions should I ask in an interview?",
        "relevant": [],
        "role": "",
        "expected_refusal": True
    },
    {
        "query": "Data analyst for financial services",
        "relevant": ["Verify Numerical Reasoning", "Verify Verbal Reasoning", "Data Analysis"],
        "role": "data analyst",
        "expected_refusal": False
    }
]

class Evaluator:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.valid_urls = get_valid_urls()
        self.valid_assessments = set()
        from retrieval import get_catalog
        for item in get_catalog():
            self.valid_assessments.add(item.name)
    
    def evaluate_retrieval_quality(self):
        """Evaluate FAISS retrieval quality"""
        print("\n" + "="*60)
        print("RETRIEVAL QUALITY EVALUATION")
        print("="*60)
        
        metrics = RetrievalMetrics()
        results = []
        
        for test in TEST_QUERIES:
            if not test['relevant']:
                continue
            
            retrieved = retrieve_assessments(test['query'], k=10)
            
            precision_5 = metrics.precision_at_k(retrieved, test['relevant'], k=5)
            recall_5 = metrics.recall_at_k(retrieved, test['relevant'], k=5)
            mrr = metrics.mean_reciprocal_rank(retrieved, test['relevant'])
            ndcg_5 = metrics.ndcg_at_k(retrieved, test['relevant'], k=5)
            
            results.append({
                'query': test['query'],
                'precision@5': precision_5,
                'recall@5': recall_5,
                'MRR': mrr,
                'NDCG@5': ndcg_5
            })
            
            print(f"\nQuery: {test['query']}")
            print(f"  Precision@5: {precision_5:.3f}")
            print(f"  Recall@5: {recall_5:.3f}")
            print(f"  MRR: {mrr:.3f}")
            print(f"  NDCG@5: {ndcg_5:.3f}")
        
        # Average metrics
        avg_precision = sum(r['precision@5'] for r in results) / len(results)
        avg_recall = sum(r['recall@5'] for r in results) / len(results)
        avg_mrr = sum(r['MRR'] for r in results) / len(results)
        
        print(f"\n{'='*40}")
        print(f"AVERAGE RETRIEVAL METRICS:")
        print(f"  Avg Precision@5: {avg_precision:.3f}")
        print(f"  Avg Recall@5: {avg_recall:.3f}")
        print(f"  Avg MRR: {avg_mrr:.3f}")
        
        return results
    
    def evaluate_groundedness(self):
        """Evaluate if responses are grounded in catalog"""
        print("\n" + "="*60)
        print("GROUNDEDNESS EVALUATION")
        print("="*60)
        
        metrics = GroundednessMetrics()
        
        # Test with actual API calls
        for test in TEST_QUERIES:
            response = requests.post(
                f"{self.api_url}/chat",
                json={"messages": [{"role": "user", "content": test['query']}]}
            ).json()
            
            recommendations = response.get('recommendations', [])
            
            url_validity = metrics.url_validity_rate(recommendations, self.valid_urls)
            hallucination = metrics.hallucination_rate(recommendations, self.valid_assessments)
            
            print(f"\nQuery: {test['query']}")
            print(f"  URL Validity Rate: {url_validity:.3f}")
            print(f"  Hallucination Rate: {hallucination:.3f}")
            print(f"  Recommendations: {len(recommendations)}")
    
    def evaluate_effectiveness(self):
        """Evaluate overall response effectiveness"""
        print("\n" + "="*60)
        print("EFFECTIVENESS EVALUATION")
        print("="*60)
        
        metrics = EffectivenessMetrics()
        results = []
        
        conversation_history = []
        
        for test in TEST_QUERIES:
            response = requests.post(
                f"{self.api_url}/chat",
                json={"messages": [{"role": "user", "content": test['query']}]}
            ).json()
            
            conversation_history.append({"role": "user", "content": test['query']})
            conversation_history.append({"role": "assistant", "content": response})
            
            completeness = metrics.response_completeness(
                response, 
                ['reply', 'recommendations', 'end_of_conversation']
            )
            refusal_correct = metrics.refusal_accuracy(response, test['expected_refusal'])
            
            results.append({
                'query': test['query'],
                'completeness': completeness,
                'refusal_correct': refusal_correct
            })
            
            print(f"\nQuery: {test['query']}")
            print(f"  Schema Completeness: {completeness:.3f}")
            print(f"  Refusal Correct: {refusal_correct}")
        
        efficiency = metrics.conversation_efficiency(conversation_history)
        print(f"\n{'='*40}")
        print(f"Average turns to recommendation: {efficiency}")
        
        return results
    
    def evaluate_relevance(self):
        """Evaluate recommendation relevance"""
        print("\n" + "="*60)
        print("RECOMMENDATION RELEVANCE")
        print("="*60)
        
        metrics = RecommendationRelevance()
        
        for test in TEST_QUERIES:
            if not test['role']:
                continue
            
            response = requests.post(
                f"{self.api_url}/chat",
                json={"messages": [{"role": "user", "content": test['query']}]}
            ).json()
            
            recommendations = response.get('recommendations', [])
            
            role_match = metrics.role_match_score(recommendations, test['role'])
            diversity = metrics.diversity_score(recommendations)
            
            print(f"\nQuery: {test['query']}")
            print(f"  Role Match Score: {role_match:.3f}")
            print(f"  Diversity Score: {diversity:.3f}")
            
            if recommendations:
                print(f"  Test Types: {[r.get('test_type') for r in recommendations]}")

def main():
    print("\n🚀 Starting Comprehensive Evaluation")
    print("Make sure the server is running at http://localhost:8000")
    
    evaluator = Evaluator()
    
    evaluator.evaluate_retrieval_quality()
    evaluator.evaluate_groundedness()
    evaluator.evaluate_effectiveness()
    evaluator.evaluate_relevance()
    
    print("\n" + "="*60)
    print("✅ Evaluation Complete!")
    print("="*60)

if __name__ == "__main__":
    main()