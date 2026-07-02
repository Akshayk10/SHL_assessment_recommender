"""
evaluation_metrics.py - Metrics for evaluating retrieval and recommendation quality
"""

import json
import numpy as np
from typing import List, Dict, Set
from collections import Counter

class RetrievalMetrics:
    """Metrics to measure retrieval quality"""
    
    @staticmethod
    def precision_at_k(retrieved: List[Dict], relevant: List[str], k: int = 5) -> float:
        """
        Precision@K: Fraction of retrieved items that are relevant.
        retrieved: List of retrieved assessment dicts
        relevant: List of relevant assessment names (ground truth)
        """
        retrieved_names = [r['name'] for r in retrieved[:k]]
        relevant_set = set(relevant)
        
        hits = sum(1 for name in retrieved_names if name in relevant_set)
        return hits / k if k > 0 else 0
    
    @staticmethod
    def recall_at_k(retrieved: List[Dict], relevant: List[str], k: int = 5) -> float:
        """
        Recall@K: Fraction of relevant items that are retrieved.
        """
        retrieved_names = set([r['name'] for r in retrieved[:k]])
        relevant_set = set(relevant)
        
        hits = len(retrieved_names & relevant_set)
        return hits / len(relevant_set) if relevant_set else 0
    
    @staticmethod
    def mean_reciprocal_rank(retrieved: List[Dict], relevant: List[str]) -> float:
        """
        MRR: Reciprocal rank of the first relevant item.
        """
        for i, r in enumerate(retrieved, 1):
            if r['name'] in relevant:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved: List[Dict], relevant: List[str], k: int = 5) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain.
        Accounts for position of relevant items.
        """
        relevance_scores = {}
        for i, name in enumerate(relevant):
            relevance_scores[name] = len(relevant) - i  # Higher rank = higher score
        
        dcg = 0.0
        for i, r in enumerate(retrieved[:k], 1):
            score = relevance_scores.get(r['name'], 0)
            dcg += score / np.log2(i + 1)
        
        # Calculate IDCG (Ideal DCG)
        ideal_relevant = sorted(relevant, key=lambda x: relevance_scores.get(x, 0), reverse=True)
        idcg = 0.0
        for i, name in enumerate(ideal_relevant[:k], 1):
            score = relevance_scores.get(name, 0)
            idcg += score / np.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0


class GroundednessMetrics:
    """Metrics to measure if responses are grounded in catalog data"""
    
    @staticmethod
    def url_validity_rate(recommendations: List[Dict], valid_urls: Set[str]) -> float:
        """
        Fraction of recommendations with valid SHL URLs.
        """
        if not recommendations:
            return 1.0
        
        valid_count = sum(1 for r in recommendations if r.get('url') in valid_urls)
        return valid_count / len(recommendations)
    
    @staticmethod
    def hallucination_rate(recommendations: List[Dict], valid_assessments: Set[str]) -> float:
        """
        Fraction of recommendations that don't exist in catalog.
        """
        if not recommendations:
            return 0.0
        
        hallucinated = sum(1 for r in recommendations if r.get('name') not in valid_assessments)
        return hallucinated / len(recommendations)
    
    @staticmethod
    def attribution_score(response: str, retrieved_assessments: List[Dict]) -> float:
        """
        Measure how much of the response is supported by retrieved assessments.
        Simple version: Check if assessment names mentioned appear in retrieved list.
        """
        if not retrieved_assessments:
            return 0.0
        
        retrieved_names = set(a['name'].lower() for a in retrieved_assessments)
        response_lower = response.lower()
        
        # Count how many retrieved assessments are mentioned
        mentioned = sum(1 for name in retrieved_names if name in response_lower)
        return mentioned / len(retrieved_names) if retrieved_names else 0.0


class EffectivenessMetrics:
    """Metrics for overall response effectiveness"""
    
    @staticmethod
    def response_completeness(response: Dict, expected_fields: List[str]) -> float:
        """
        Check if response contains all required fields.
        """
        if not response:
            return 0.0
        
        present = sum(1 for field in expected_fields if field in response)
        return present / len(expected_fields)
    
    @staticmethod
    def conversation_efficiency(conversation_history: List[Dict]) -> int:
        """
        Number of turns needed to reach recommendation.
        Lower is better.
        """
        for i, message in enumerate(conversation_history):
            if message.get('role') == 'assistant':
                if message.get('recommendations') and len(message.get('recommendations', [])) > 0:
                    return i // 2 + 1  # Convert message index to turn number
        return len(conversation_history) // 2
    
    @staticmethod
    def refusal_accuracy(response: Dict, expected_refusal: bool) -> bool:
        """
        Check if agent correctly refuses off-topic queries.
        """
        actual_refusal = len(response.get('recommendations', [])) == 0
        return actual_refusal == expected_refusal


class RecommendationRelevance:
    """Metrics for recommendation relevance"""
    
    @staticmethod
    def role_match_score(recommendations: List[Dict], job_role: str) -> float:
        """
        Measure how well recommendations match the job role.
        Based on keyword overlap between assessment name/description and job role.
        """
        if not recommendations:
            return 0.0
        
        role_keywords = set(job_role.lower().split())
        if not role_keywords:
            return 1.0
        
        total_score = 0.0
        for rec in recommendations:
            rec_text = f"{rec.get('name', '')} {rec.get('description', '')}".lower()
            matches = sum(1 for kw in role_keywords if kw in rec_text)
            total_score += matches / len(role_keywords)
        
        return total_score / len(recommendations)
    
    @staticmethod
    def diversity_score(recommendations: List[Dict]) -> float:
        """
        Measure diversity of test types in recommendations.
        1.0 = all different types, 0.0 = all same type
        """
        if len(recommendations) <= 1:
            return 1.0
        
        test_types = [r.get('test_type', '') for r in recommendations]
        unique_types = len(set(test_types))
        return unique_types / len(test_types)
    
    @staticmethod
    def coverage_score(recommendations: List[Dict], required_skills: List[str]) -> float:
        """
        Measure how well recommendations cover required skills.
        """
        if not required_skills or not recommendations:
            return 0.0
        
        skill_set = set(required_skills)
        covered = set()
        
        for rec in recommendations:
            rec_text = f"{rec.get('name', '')} {rec.get('description', '')}".lower()
            for skill in skill_set:
                if skill.lower() in rec_text:
                    covered.add(skill)
        
        return len(covered) / len(skill_set)