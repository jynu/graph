#!/usr/bin/env python3
"""
Advanced Text-to-SQL Web UI with Comprehensive Evaluation

A Gradio-based web interface for text-to-SQL generation using Advanced Graph Traversal
and comprehensive SQL quality evaluation using state-of-the-art metrics.

Features:
- Text-to-Table discovery using Advanced Graph Traversal
- Table selection and SQL generation
- Comprehensive SQL evaluation with multiple metrics
- Real-time performance monitoring
- Interactive query examples

Usage:
    python text_to_sql_web_ui.py

Requirements:
    pip install gradio requests pandas numpy asyncio aiohttp sqlparse
"""

import gradio as gr
import pandas as pd
import numpy as np
import json
import logging
import asyncio
import aiohttp
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import re
import os
import sqlparse
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://localhost:8000"  # Adjust to your API server
DEFAULT_USER_ID = "web_ui_user"

# SQL Evaluation Libraries (if available)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ sklearn not available - some evaluation metrics disabled")

try:
    from app.utils.client_manager import client_manager
    CLIENT_MANAGER_AVAILABLE = True
except ImportError:
    CLIENT_MANAGER_AVAILABLE = False
    print("⚠️ client_manager not available - GPT evaluation may be limited")

# Global state
current_session = {
    "query": "",
    "tables": [],
    "selected_tables": [],
    "generated_sql": "",
    "reasoning": "",
    "table_details": {},
    "performance_metrics": {}
}

# === SQL Evaluation Metrics Implementation ===

class SQLEvaluator:
    """Comprehensive SQL evaluation using state-of-the-art metrics."""
    
    def __init__(self):
        self.metrics = [
            "Execution Accuracy (EX)",
            "Exact Match (EM)", 
            "Component Match (F1)",
            "BLEU Score",
            "CodeBLEU",
            "Syntactic Similarity",
            "Semantic Similarity",
            "Valid Efficiency Score (VES)",
            "GPT-4 Quality Assessment",
            "Soft F1-Score"
        ]
    
    async def evaluate_sql(self, generated_sql: str, ground_truth_sql: str, 
                          query: str = "", table_details: Dict = None) -> Dict[str, Any]:
        """Comprehensive SQL evaluation using multiple metrics."""
        
        logger.info("🔍 Starting comprehensive SQL evaluation...")
        
        results = {
            "overall_score": 0.0,
            "metrics": {},
            "detailed_analysis": "",
            "recommendations": [],
            "execution_time": 0.0
        }
        
        start_time = time.time()
        
        try:
            # 1. Exact Match (EM)
            results["metrics"]["exact_match"] = self._exact_match(generated_sql, ground_truth_sql)
            
            # 2. Component Match (F1)
            results["metrics"]["component_f1"] = self._component_match_f1(generated_sql, ground_truth_sql)
            
            # 3. BLEU Score
            results["metrics"]["bleu_score"] = self._compute_bleu(generated_sql, ground_truth_sql)
            
            # 4. CodeBLEU (enhanced for SQL)
            results["metrics"]["code_bleu"] = self._compute_code_bleu(generated_sql, ground_truth_sql)
            
            # 5. Syntactic Similarity
            results["metrics"]["syntactic_similarity"] = self._syntactic_similarity(generated_sql, ground_truth_sql)
            
            # 6. Semantic Similarity
            results["metrics"]["semantic_similarity"] = self._semantic_similarity(generated_sql, ground_truth_sql)
            
            # 7. Valid Efficiency Score (VES) - simulated
            results["metrics"]["ves_score"] = self._compute_ves(generated_sql, ground_truth_sql)
            
            # 8. Soft F1-Score (BIRD benchmark style)
            results["metrics"]["soft_f1"] = self._compute_soft_f1(generated_sql, ground_truth_sql)
            
            # 9. GPT-4 Quality Assessment
            if CLIENT_MANAGER_AVAILABLE:
                results["metrics"]["gpt4_assessment"] = await self._gpt4_quality_assessment(
                    generated_sql, ground_truth_sql, query, table_details
                )
            else:
                results["metrics"]["gpt4_assessment"] = {"score": 0.0, "feedback": "GPT-4 evaluation not available"}
            
            # Calculate overall score (weighted average)
            results["overall_score"] = self._calculate_overall_score(results["metrics"])
            
            # Generate detailed analysis
            results["detailed_analysis"] = self._generate_detailed_analysis(results["metrics"])
            
            # Generate recommendations
            results["recommendations"] = self._generate_recommendations(results["metrics"])
            
            results["execution_time"] = time.time() - start_time
            
            logger.info(f"✅ SQL evaluation completed in {results['execution_time']:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ SQL evaluation failed: {e}")
            results["error"] = str(e)
            results["execution_time"] = time.time() - start_time
        
        return results
    
    def _exact_match(self, generated: str, ground_truth: str) -> Dict[str, Any]:
        """Exact match evaluation."""
        # Normalize SQL queries
        gen_normalized = self._normalize_sql(generated)
        gt_normalized = self._normalize_sql(ground_truth)
        
        exact_match = gen_normalized.strip().lower() == gt_normalized.strip().lower()
        
        return {
            "score": 1.0 if exact_match else 0.0,
            "details": {
                "match": exact_match,
                "generated_normalized": gen_normalized,
                "ground_truth_normalized": gt_normalized
            }
        }
    
    def _component_match_f1(self, generated: str, ground_truth: str) -> Dict[str, Any]:
        """Component-wise F1 score evaluation."""
        try:
            gen_components = self._extract_sql_components(generated)
            gt_components = self._extract_sql_components(ground_truth)
            
            # Calculate F1 for each component type
            component_f1s = {}
            overall_f1 = 0.0
            
            all_component_types = set(gen_components.keys()) | set(gt_components.keys())
            
            for comp_type in all_component_types:
                gen_comp = set(gen_components.get(comp_type, []))
                gt_comp = set(gt_components.get(comp_type, []))
                
                if not gen_comp and not gt_comp:
                    f1 = 1.0
                elif not gen_comp or not gt_comp:
                    f1 = 0.0
                else:
                    intersection = len(gen_comp & gt_comp)
                    precision = intersection / len(gen_comp) if gen_comp else 0.0
                    recall = intersection / len(gt_comp) if gt_comp else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                component_f1s[comp_type] = f1
            
            overall_f1 = np.mean(list(component_f1s.values())) if component_f1s else 0.0
            
            return {
                "score": overall_f1,
                "details": {
                    "component_f1s": component_f1s,
                    "generated_components": gen_components,
                    "ground_truth_components": gt_components
                }
            }
            
        except Exception as e:
            return {"score": 0.0, "error": str(e)}
    
    def _compute_bleu(self, generated: str, ground_truth: str) -> Dict[str, Any]:
        """BLEU score computation for SQL."""
        try:
            # Tokenize SQL queries
            gen_tokens = self._tokenize_sql(generated)
            gt_tokens = self._tokenize_sql(ground_truth)
            
            # Calculate BLEU score (simplified version)
            if not gen_tokens or not gt_tokens:
                return {"score": 0.0, "details": {"reason": "Empty tokens"}}
            
            # 1-gram precision
            gen_unigrams = set(gen_tokens)
            gt_unigrams = set(gt_tokens)
            
            precision_1 = len(gen_unigrams & gt_unigrams) / len(gen_unigrams) if gen_unigrams else 0.0
            
            # 2-gram precision
            gen_bigrams = set(zip(gen_tokens[:-1], gen_tokens[1:]))
            gt_bigrams = set(zip(gt_tokens[:-1], gt_tokens[1:]))
            
            precision_2 = len(gen_bigrams & gt_bigrams) / len(gen_bigrams) if gen_bigrams else 0.0
            
            # Brevity penalty
            bp = min(1.0, len(gen_tokens) / len(gt_tokens)) if gt_tokens else 0.0
            
            # BLEU score (simplified)
            bleu_score = bp * np.sqrt(precision_1 * precision_2) if precision_1 > 0 and precision_2 > 0 else 0.0
            
            return {
                "score": bleu_score,
                "details": {
                    "precision_1": precision_1,
                    "precision_2": precision_2,
                    "brevity_penalty": bp,
                    "generated_tokens": len(gen_tokens),
                    "ground_truth_tokens": len(gt_tokens)
                }
            }
            
        except Exception as e:
            return {"score": 0.0, "error": str(e)}
    
    def _compute_code_bleu(self, generated: str, ground_truth: str) -> Dict[str, Any]:
        """CodeBLEU score for SQL (adapted for SQL syntax)."""
        try:
            # Standard BLEU component
            bleu_result = self._compute_bleu(generated, ground_truth)
            bleu_score = bleu_result["score"]
            
            # Weighted n-gram match (SQL keywords have higher weight)
            weighted_score = self._compute_weighted_ngram_match(generated, ground_truth)
            
            # Syntactic AST match (simplified for SQL)
            ast_score = self._compute_sql_ast_match(generated, ground_truth)
            
            # Data flow match (simplified)
            dataflow_score = self._compute_sql_dataflow_match(generated, ground_truth)
            
            # Combine scores (weights based on CodeBLEU paper)
            code_bleu = (
                0.25 * bleu_score +
                0.25 * weighted_score +
                0.25 * ast_score +
                0.25 * dataflow_score
            )
            
            return {
                "score": code_bleu,
                "details": {
                    "bleu_component": bleu_score,
                    "weighted_ngram": weighted_score,
                    "ast_match": ast_score,
                    "dataflow_match": dataflow_score
                }
            }
            
        except Exception as e:
            return {"score": 0.0, "error": str(e)}
    
    def _syntactic_similarity(self, generated: str, ground_truth: str) -> Dict[str, Any]:
        """Syntactic similarity based on SQL structure."""
        try:
            gen_structure = self._extract_sql_structure(generated)
            gt_structure = self._extract_sql_structure(ground_truth)
            
            # Compare structural elements
            structure_similarity = 0.0
            
            if gen_structure and gt_structure:
                # Compare clause presence
                gen_clauses = set(gen_structure.keys())
                gt_clauses = set(gt_structure.keys())
                
                clause_overlap = len(gen_clauses & gt_clauses)
                clause_union = len(gen_clauses | gt_clauses)
                
                structure_similarity = clause_overlap / clause_union if clause_union > 0 else 0.0
            
            return {
                "score": structure_similarity,
                "details": {
                    "generated_structure": gen_structure,
                    "ground_truth_structure": gt_structure
                }
            }
            
        except Exception as e:
            return {"score": 0.0, "error": str(e)}
    
    def _semantic_similarity(self, generated: str, ground_truth: str) -> Dict[str, Any]:
        """Semantic similarity using text similarity."""
        try:
            if not SKLEARN_AVAILABLE:
                return {"score": 0.0, "details": {"reason": "sklearn not available"}}
            
            # Clean and normalize
            gen_clean = self._normalize_sql(generated)
            gt_clean = self._normalize_sql(ground_truth)
            
            if not gen_clean or not gt_clean:
                return {"score": 0.0, "details": {"reason": "Empty SQL after normalization"}}
            
            # TF-IDF similarity
            vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
            
            try:
                tfidf_matrix = vectorizer.fit_transform([gen_clean, gt_clean])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except:
                similarity = 0.0
            
            return {
                "score": similarity,
                "details": {
                    "method": "TF-IDF Cosine Similarity",
                    "generated_clean": gen_clean[:100] + "..." if len(gen_clean) > 100 else gen_clean,
                    "ground_truth_clean": gt_clean[:100] + "..." if len(gt_clean) > 100 else gt_clean
                }
            }
            
        except Exception as e:
            return {"score": 0.0, "error": str(e)}
    
    def _compute_ves(self, generated: str, ground_truth: str) -> Dict[str, Any]:
        """Valid Efficiency Score (VES) - simulated."""
        try:
            # Check if SQL is valid (basic syntax check)
            is_valid_gen = self._is_valid_sql(generated)
            is_valid_gt = self._is_valid_sql(ground_truth)
            
            if not is_valid_gen:
                return {"score": 0.0, "details": {"reason": "Generated SQL is invalid"}}
            
            # Estimate efficiency (simplified)
            efficiency_score = self._estimate_sql_efficiency(generated, ground_truth)
            
            # VES combines validity and efficiency
            ves_score = efficiency_score if is_valid_gen else 0.0
            
            return {
                "score": ves_score,
                "details": {
                    "is_valid_generated": is_valid_gen,
                    "is_valid_ground_truth": is_valid_gt,
                    "efficiency_estimate": efficiency_score
                }
            }
            
        except Exception as e:
            return {"score": 0.0, "error": str(e)}
    
    def _compute_soft_f1(self, generated: str, ground_truth: str) -> Dict[str, Any]:
        """Soft F1-Score (BIRD benchmark style)."""
        try:
            # Extract key components with fuzzy matching
            gen_entities = self._extract_sql_entities(generated)
            gt_entities = self._extract_sql_entities(ground_truth)
            
            # Soft matching for entities
            soft_matches = 0
            total_gen = len(gen_entities)
            total_gt = len(gt_entities)
            
            for gen_entity in gen_entities:
                for gt_entity in gt_entities:
                    # Fuzzy string matching
                    similarity = self._string_similarity(gen_entity, gt_entity)
                    if similarity > 0.8:  # Threshold for soft match
                        soft_matches += 1
                        break
            
            # Calculate soft precision and recall
            soft_precision = soft_matches / total_gen if total_gen > 0 else 0.0
            soft_recall = soft_matches / total_gt if total_gt > 0 else 0.0
            
            # Soft F1
            soft_f1 = (2 * soft_precision * soft_recall / (soft_precision + soft_recall) 
                      if (soft_precision + soft_recall) > 0 else 0.0)
            
            return {
                "score": soft_f1,
                "details": {
                    "soft_precision": soft_precision,
                    "soft_recall": soft_recall,
                    "soft_matches": soft_matches,
                    "generated_entities": gen_entities,
                    "ground_truth_entities": gt_entities
                }
            }
            
        except Exception as e:
            return {"score": 0.0, "error": str(e)}
    
    async def _gpt4_quality_assessment(self, generated: str, ground_truth: str, 
                                     query: str = "", table_details: Dict = None) -> Dict[str, Any]:
        """GPT-4 based quality assessment."""
        try:
            assessment_prompt = f"""You are an expert SQL evaluation system. Please evaluate the quality of a generated SQL query against a ground truth query.

**Original Query:** {query}

**Generated SQL:**
```sql
{generated}
```

**Ground Truth SQL:**
```sql
{ground_truth}
```

**Evaluation Criteria:**
1. **Correctness** (0-10): Does the generated SQL achieve the same logical result as the ground truth?
2. **Syntax** (0-10): Is the generated SQL syntactically correct and well-formed?
3. **Efficiency** (0-10): Is the generated SQL reasonably efficient compared to the ground truth?
4. **Completeness** (0-10): Does the generated SQL address all requirements from the original query?
5. **Style** (0-10): Is the generated SQL well-formatted and follows SQL best practices?

**Response Format:**
Provide scores for each criteria and overall assessment in this JSON format:
{{
    "correctness": 8.5,
    "syntax": 9.0,
    "efficiency": 7.5,
    "completeness": 8.0,
    "style": 9.0,
    "overall_score": 8.4,
    "feedback": "Detailed explanation of the assessment...",
    "strengths": ["List of strengths"],
    "weaknesses": ["List of weaknesses"],
    "suggestions": ["List of improvement suggestions"]
}}

**Assessment:**"""

            # Use client manager to call GPT-4
            response = await client_manager.ask_gpt(assessment_prompt)
            
            # Parse JSON response
            try:
                assessment_data = json.loads(response)
                return {
                    "score": assessment_data.get("overall_score", 0.0) / 10.0,  # Normalize to 0-1
                    "details": assessment_data
                }
            except json.JSONDecodeError:
                # Fallback: extract scores from text
                return {
                    "score": 0.5,  # Default
                    "details": {
                        "feedback": response,
                        "note": "Could not parse structured response"
                    }
                }
                
        except Exception as e:
            return {
                "score": 0.0,
                "details": {"error": str(e)}
            }
    
    def _calculate_overall_score(self, metrics: Dict) -> float:
        """Calculate weighted overall score."""
        weights = {
            "exact_match": 0.15,
            "component_f1": 0.20,
            "bleu_score": 0.10,
            "code_bleu": 0.20,
            "syntactic_similarity": 0.10,
            "semantic_similarity": 0.10,
            "ves_score": 0.10,
            "soft_f1": 0.15,
            "gpt4_assessment": 0.15 if CLIENT_MANAGER_AVAILABLE else 0.0
        }
        
        # Normalize weights if GPT-4 not available
        if not CLIENT_MANAGER_AVAILABLE:
            total_weight = sum(w for k, w in weights.items() if k != "gpt4_assessment")
            weights = {k: v/total_weight for k, v in weights.items() if k != "gpt4_assessment"}
        
        overall_score = 0.0
        for metric_name, weight in weights.items():
            if metric_name in metrics:
                score = metrics[metric_name].get("score", 0.0)
                overall_score += weight * score
        
        return overall_score
    
    def _generate_detailed_analysis(self, metrics: Dict) -> str:
        """Generate detailed analysis text."""
        analysis = "## 📊 Detailed SQL Evaluation Analysis\n\n"
        
        for metric_name, result in metrics.items():
            score = result.get("score", 0.0)
            analysis += f"### {metric_name.replace('_', ' ').title()}\n"
            analysis += f"**Score:** {score:.3f} ({self._score_to_grade(score)})\n"
            
            if "error" in result:
                analysis += f"**Error:** {result['error']}\n"
            elif "details" in result:
                analysis += f"**Details:** {str(result['details'])[:200]}...\n"
            
            analysis += "\n"
        
        return analysis
    
    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Check each metric and provide specific recommendations
        if metrics.get("exact_match", {}).get("score", 0) < 0.5:
            recommendations.append("🎯 Focus on improving exact query structure and syntax")
        
        if metrics.get("component_f1", {}).get("score", 0) < 0.7:
            recommendations.append("🔧 Review SQL components (SELECT, FROM, WHERE, etc.) for accuracy")
        
        if metrics.get("syntactic_similarity", {}).get("score", 0) < 0.6:
            recommendations.append("📝 Improve SQL syntax and structure formatting")
        
        if metrics.get("semantic_similarity", {}).get("score", 0) < 0.6:
            recommendations.append("🧠 Focus on semantic correctness and logical equivalence")
        
        if metrics.get("ves_score", {}).get("score", 0) < 0.7:
            recommendations.append("⚡ Optimize query efficiency and performance")
        
        if not recommendations:
            recommendations.append("✅ SQL quality is good overall! Continue refining for excellence.")
        
        return recommendations
    
    # Helper methods
    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL query for comparison."""
        try:
            # Parse and format SQL
            parsed = sqlparse.parse(sql)[0] if sqlparse.parse(sql) else None
            if parsed:
                return sqlparse.format(str(parsed), reindent=True, keyword_case='upper').strip()
            else:
                return sql.strip()
        except:
            return sql.strip()
    
    def _extract_sql_components(self, sql: str) -> Dict[str, List[str]]:
        """Extract SQL components (SELECT, FROM, WHERE, etc.)."""
        components = defaultdict(list)
        
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                return components
            
            tokens = parsed[0].flatten()
            current_clause = None
            
            for token in tokens:
                if token.ttype is sqlparse.tokens.Keyword:
                    keyword = token.value.upper()
                    if keyword in ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING']:
                        current_clause = keyword
                elif current_clause and token.ttype not in [sqlparse.tokens.Whitespace, sqlparse.tokens.Punctuation]:
                    components[current_clause].append(token.value.strip())
            
        except:
            pass
        
        return dict(components)
    
    def _tokenize_sql(self, sql: str) -> List[str]:
        """Tokenize SQL query."""
        try:
            parsed = sqlparse.parse(sql)
            if parsed:
                tokens = [token.value.strip().lower() for token in parsed[0].flatten() 
                         if token.ttype not in [sqlparse.tokens.Whitespace]]
                return [t for t in tokens if t]
            else:
                return sql.split()
        except:
            return sql.split()
    
    def _extract_sql_structure(self, sql: str) -> Dict[str, Any]:
        """Extract SQL structural elements."""
        structure = {}
        
        try:
            sql_upper = sql.upper()
            
            # Check for main clauses
            if 'SELECT' in sql_upper:
                structure['SELECT'] = True
            if 'FROM' in sql_upper:
                structure['FROM'] = True
            if 'WHERE' in sql_upper:
                structure['WHERE'] = True
            if 'GROUP BY' in sql_upper:
                structure['GROUP_BY'] = True
            if 'ORDER BY' in sql_upper:
                structure['ORDER_BY'] = True
            if 'HAVING' in sql_upper:
                structure['HAVING'] = True
            if 'JOIN' in sql_upper:
                structure['JOIN'] = True
            
        except:
            pass
        
        return structure
    
    def _compute_weighted_ngram_match(self, generated: str, ground_truth: str) -> float:
        """Compute weighted n-gram match (SQL keywords have higher weight)."""
        try:
            gen_tokens = self._tokenize_sql(generated)
            gt_tokens = self._tokenize_sql(ground_truth)
            
            # SQL keywords (higher weight)
            sql_keywords = set(['select', 'from', 'where', 'group', 'by', 'order', 'having', 
                               'join', 'inner', 'left', 'right', 'outer', 'on', 'and', 'or'])
            
            # Weight calculation
            total_weight = 0
            matched_weight = 0
            
            for token in gen_tokens:
                weight = 5.0 if token.lower() in sql_keywords else 1.0
                total_weight += weight
                
                if token in gt_tokens:
                    matched_weight += weight
            
            return matched_weight / total_weight if total_weight > 0 else 0.0
            
        except:
            return 0.0
    
    def _compute_sql_ast_match(self, generated: str, ground_truth: str) -> float:
        """Simplified SQL AST matching."""
        try:
            gen_structure = self._extract_sql_structure(generated)
            gt_structure = self._extract_sql_structure(ground_truth)
            
            if not gen_structure or not gt_structure:
                return 0.0
            
            common_structures = set(gen_structure.keys()) & set(gt_structure.keys())
            total_structures = set(gen_structure.keys()) | set(gt_structure.keys())
            
            return len(common_structures) / len(total_structures) if total_structures else 0.0
            
        except:
            return 0.0
    
    def _compute_sql_dataflow_match(self, generated: str, ground_truth: str) -> float:
        """Simplified SQL dataflow matching."""
        try:
            # Extract table and column references
            gen_refs = self._extract_sql_references(generated)
            gt_refs = self._extract_sql_references(ground_truth)
            
            if not gen_refs or not gt_refs:
                return 0.0
            
            common_refs = set(gen_refs) & set(gt_refs)
            total_refs = set(gen_refs) | set(gt_refs)
            
            return len(common_refs) / len(total_refs) if total_refs else 0.0
            
        except:
            return 0.0
    
    def _extract_sql_references(self, sql: str) -> List[str]:
        """Extract table and column references."""
        references = []
        
        try:
            # Simple regex-based extraction
            # This is a simplified version - in production, use proper SQL parsing
            words = re.findall(r'\b\w+\b', sql)
            
            # Filter out SQL keywords
            sql_keywords = set(['select', 'from', 'where', 'group', 'by', 'order', 'having', 
                               'join', 'inner', 'left', 'right', 'outer', 'on', 'and', 'or',
                               'as', 'distinct', 'count', 'sum', 'avg', 'max', 'min'])
            
            references = [word.lower() for word in words 
                         if word.lower() not in sql_keywords and len(word) > 2]
            
        except:
            pass
        
        return references
    
    def _is_valid_sql(self, sql: str) -> bool:
        """Check if SQL is syntactically valid."""
        try:
            parsed = sqlparse.parse(sql)
            return len(parsed) > 0 and parsed[0].tokens
        except:
            return False
    
    def _estimate_sql_efficiency(self, generated: str, ground_truth: str) -> float:
        """Estimate SQL efficiency (simplified)."""
        try:
            # Simple heuristics for efficiency
            gen_complexity = self._sql_complexity_score(generated)
            gt_complexity = self._sql_complexity_score(ground_truth)
            
            # Efficiency based on relative complexity
            if gt_complexity == 0:
                return 1.0
            
            efficiency = min(1.0, gt_complexity / gen_complexity)
            return efficiency
            
        except:
            return 0.5
    
    def _sql_complexity_score(self, sql: str) -> float:
        """Calculate SQL complexity score based on various factors."""
        try:
            sql_upper = sql.upper()
            complexity = 1.0
            
            # Add complexity for various SQL features
            if 'JOIN' in sql_upper:
                complexity += sql_upper.count('JOIN') * 2
            if 'SUBQUERY' in sql_upper or '(' in sql and 'SELECT' in sql:
                complexity += 3
            if 'GROUP BY' in sql_upper:
                complexity += 2
            if 'ORDER BY' in sql_upper:
                complexity += 1
            if 'HAVING' in sql_upper:
                complexity += 2
            
            return complexity
            
        except:
            return 1.0
    
    def _extract_sql_entities(self, sql: str) -> List[str]:
        """Extract SQL entities for soft matching."""
        entities = []
        
        try:
            # Extract identifiers (table names, column names, etc.)
            words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', sql)
            
            # Filter out SQL keywords
            sql_keywords = set(['select', 'from', 'where', 'group', 'by', 'order', 'having', 
                               'join', 'inner', 'left', 'right', 'outer', 'on', 'and', 'or',
                               'as', 'distinct', 'count', 'sum', 'avg', 'max', 'min', 'case',
                               'when', 'then', 'else', 'end', 'in', 'not', 'null', 'is'])
            
            entities = [word.lower() for word in words 
                       if word.lower() not in sql_keywords and len(word) > 1]
            
        except:
            pass
        
        return entities
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity (Levenshtein-based)."""
        try:
            from difflib import SequenceMatcher
            return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
        except:
            return 1.0 if str1.lower() == str2.lower() else 0.0
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"

# === API Client Functions ===

async def call_api_async(endpoint: str, data: Dict) -> Dict:
    """Make async API call to text-to-SQL service."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{API_BASE_URL}{endpoint}", json=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    return {"success": False, "error": f"API error {response.status}: {error_text}"}
    except Exception as e:
        return {"success": False, "error": f"Connection error: {str(e)}"}

def call_api_sync(endpoint: str, data: Dict) -> Dict:
    """Synchronous wrapper for API calls."""
    return asyncio.run(call_api_async(endpoint, data))

# === Text-to-Table Functions ===

def text_to_table_search(query: str, max_tables: int = 10) -> Tuple[str, pd.DataFrame, Dict]:
    """Search for relevant tables using Advanced Graph Traversal."""
    global current_session
    
    if not query.strip():
        return "Please enter a query.", pd.DataFrame(), {}
    
    logger.info(f"🔍 Text-to-Table search: {query}")
    
    try:
        # Call text-to-table API
        api_data = {
            "query": query,
            "user_id": DEFAULT_USER_ID,
            "max_tables": max_tables
        }
        
        result = call_api_sync("/text-to-table", api_data)
        
        if result.get("success", False):
            tables = result.get("tables", [])
            table_details = result.get("table_details", {})
            processing_time = result.get("processing_time", 0.0)
            
            # Update session
            current_session["query"] = query
            current_session["tables"] = tables
            current_session["selected_tables"] = tables.copy()  # Select all by default
            current_session["table_details"] = table_details
            
            # Create summary
            summary = f"""## 🗄️ Table Discovery Results

**Query:** {query}

**Method:** Advanced Graph Traversal (GNN + RL + Multi-level)

**Performance:**
- Found {len(tables)} relevant tables
- Processing time: {processing_time:.3f} seconds

**Status:** ✅ Success

### 📋 Selected Tables:
{', '.join([f'`{table}`' for table in tables]) if tables else 'No tables found'}

**Next Step:** Review the table selections below and click "Generate SQL" to create the SQL query.
"""
            
            # Create DataFrame for table display
            table_data = []
            for table_name in tables:
                details = table_details.get(table_name, {})
                table_data.append({
                    "Table Name": table_name,
                    "Type": details.get("table_type", ""),
                    "Description": details.get("description", "")[:100] + "..." if len(details.get("description", "")) > 100 else details.get("description", ""),
                    "Columns": len(details.get("columns", [])),
                    "Selected": "✅"
                })
            
            df = pd.DataFrame(table_data)
            
            return summary, df, table_details
            
        else:
            error_msg = result.get("error", "Unknown error")
            return f"❌ Error: {error_msg}", pd.DataFrame(), {}
            
    except Exception as e:
        logger.error(f"Text-to-table search failed: {e}")
        return f"❌ Error: {str(e)}", pd.DataFrame(), {}

def update_table_selection(selected_tables: List[str]) -> str:
    """Update selected tables."""
    global current_session
    
    if not selected_tables:
        return "⚠️ Please select at least one table."
    
    current_session["selected_tables"] = selected_tables
    
    return f"✅ Updated selection: {len(selected_tables)} tables selected: {', '.join([f'`{t}`' for t in selected_tables])}"

# === SQL Generation Functions ===

def generate_sql_from_tables(selected_tables: List[str] = None) -> Tuple[str, str, str]:
    """Generate SQL from selected tables."""
    global current_session
    
    if not current_session.get("query"):
        return "❌ No query found. Please search for tables first.", "", ""
    
    tables_to_use = selected_tables or current_session.get("selected_tables", [])
    
    if not tables_to_use:
        return "❌ No tables selected. Please select tables first.", "", ""
    
    logger.info(f"🤖 Generating SQL for query: {current_session['query']}")
    
    try:
        # Call table-to-SQL API
        api_data = {
            "selected_tables": tables_to_use,
            "original_query": current_session["query"],
            "user_id": DEFAULT_USER_ID
        }
        
        result = call_api_sync("/table-to-sql", api_data)
        
        if result.get("success", False):
            sql_code = result.get("sql", "")
            reasoning = result.get("reasoning", "")
            processing_time = result.get("processing_time", 0.0)
            validation_status = result.get("validation_status", "unknown")
            
            # Update session
            current_session["generated_sql"] = sql_code
            current_session["reasoning"] = reasoning
            
            # Create summary
            summary = f"""## 🤖 SQL Generation Results

**Original Query:** {current_session['query']}

**Selected Tables:** {', '.join([f'`{t}`' for t in tables_to_use])}

**Processing Time:** {processing_time:.3f} seconds

**Validation Status:** {validation_status}

**Status:** ✅ SQL Generated Successfully

### 📝 Generated SQL:
```sql
{sql_code}
```

**Next Step:** Review the generated SQL and optionally evaluate its quality using the evaluation section below.
"""
            
            return summary, sql_code, reasoning
            
        else:
            error_msg = result.get("error", "Unknown error")
            return f"❌ SQL Generation Error: {error_msg}", "", ""
            
    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        return f"❌ Error: {str(e)}", "", ""

def complete_text_to_sql(query: str, max_tables: int = 8, include_reasoning: bool = True) -> Tuple[str, str, str, pd.DataFrame]:
    """Complete text-to-SQL pipeline in one step."""
    global current_session
    
    if not query.strip():
        return "Please enter a query.", "", "", pd.DataFrame()
    
    logger.info(f"🚀 Complete Text-to-SQL pipeline: {query}")
    
    try:
        # Call complete text-to-SQL API
        api_data = {
            "query": query,
            "user_id": DEFAULT_USER_ID,
            "max_tables": max_tables,
            "include_reasoning": include_reasoning
        }
        
        result = call_api_sync("/text-to-sql", api_data)
        
        if result.get("success", False):
            sql_code = result.get("sql", "")
            reasoning = result.get("reasoning", "")
            tables_found = result.get("tables_found", [])
            table_details = result.get("table_details", {})
            processing_time = result.get("processing_time", 0.0)
            
            # Update session
            current_session["query"] = query
            current_session["tables"] = tables_found
            current_session["selected_tables"] = tables_found
            current_session["generated_sql"] = sql_code
            current_session["reasoning"] = reasoning
            current_session["table_details"] = table_details
            
            # Create summary
            summary = f"""## 🚀 Complete Text-to-SQL Results

**Query:** {query}

**Method:** Advanced Graph Traversal + GPT-4 SQL Generation

**Performance:**
- Tables discovered: {len(tables_found)}
- Processing time: {processing_time:.3f} seconds

**Tables Used:** {', '.join([f'`{t}`' for t in tables_found])}

**Status:** ✅ Complete Pipeline Success

### 📝 Generated SQL:
```sql
{sql_code}
```
"""
            
            # Create table details DataFrame
            table_data = []
            for table_name in tables_found:
                details = table_details.get(table_name, {})
                table_data.append({
                    "Table Name": table_name,
                    "Type": details.get("table_type", ""),
                    "Description": details.get("description", "")[:100] + "..." if len(details.get("description", "")) > 100 else details.get("description", ""),
                    "Columns": len(details.get("columns", []))
                })
            
            df = pd.DataFrame(table_data)
            
            return summary, sql_code, reasoning, df
            
        else:
            error_msg = result.get("error", "Unknown error")
            return f"❌ Complete Text-to-SQL Error: {error_msg}", "", "", pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Complete text-to-SQL failed: {e}")
        return f"❌ Error: {str(e)}", "", "", pd.DataFrame()

# === SQL Evaluation Functions ===

def evaluate_generated_sql(ground_truth_sql: str) -> Tuple[str, pd.DataFrame, str]:
    """Evaluate generated SQL against ground truth."""
    global current_session
    
    if not current_session.get("generated_sql"):
        return "❌ No generated SQL found. Please generate SQL first.", pd.DataFrame(), ""
    
    if not ground_truth_sql.strip():
        return "❌ Please enter the ground truth SQL for evaluation.", pd.DataFrame(), ""
    
    logger.info("📊 Starting SQL evaluation...")
    
    try:
        evaluator = SQLEvaluator()
        
        # Run evaluation
        evaluation_result = asyncio.run(evaluator.evaluate_sql(
            generated_sql=current_session["generated_sql"],
            ground_truth_sql=ground_truth_sql,
            query=current_session.get("query", ""),
            table_details=current_session.get("table_details", {})
        ))
        
        if "error" in evaluation_result:
            return f"❌ Evaluation Error: {evaluation_result['error']}", pd.DataFrame(), ""
        
        overall_score = evaluation_result.get("overall_score", 0.0)
        metrics = evaluation_result.get("metrics", {})
        detailed_analysis = evaluation_result.get("detailed_analysis", "")
        recommendations = evaluation_result.get("recommendations", [])
        execution_time = evaluation_result.get("execution_time", 0.0)
        
        # Create summary
        grade = evaluator._score_to_grade(overall_score)
        summary = f"""## 📊 SQL Quality Evaluation Results

**Overall Score:** {overall_score:.3f} / 1.0 ({grade})

**Evaluation Time:** {execution_time:.3f} seconds

**Query:** {current_session.get('query', 'N/A')}

### 🎯 Key Metrics Summary:
"""
        
        # Add metric summaries
        for metric_name, result in metrics.items():
            score = result.get("score", 0.0)
            metric_grade = evaluator._score_to_grade(score)
            summary += f"- **{metric_name.replace('_', ' ').title()}:** {score:.3f} ({metric_grade})\n"
        
        summary += f"\n### 💡 Recommendations:\n"
        for rec in recommendations:
            summary += f"- {rec}\n"
        
        # Create metrics DataFrame
        metrics_data = []
        for metric_name, result in metrics.items():
            score = result.get("score", 0.0)
            grade = evaluator._score_to_grade(score)
            status = "✅ Good" if score >= 0.7 else "⚠️ Needs Improvement" if score >= 0.5 else "❌ Poor"
            
            metrics_data.append({
                "Metric": metric_name.replace('_', ' ').title(),
                "Score": f"{score:.3f}",
                "Grade": grade,
                "Status": status
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        return summary, metrics_df, detailed_analysis
        
    except Exception as e:
        logger.error(f"SQL evaluation failed: {e}")
        return f"❌ Evaluation Error: {str(e)}", pd.DataFrame(), ""

# === Utility Functions ===

def get_system_status() -> str:
    """Get system status information."""
    try:
        # Test API connection
        health_result = call_api_sync("/health", {})
        
        api_status = "✅ Connected" if health_result.get("status") == "healthy" else "❌ Disconnected"
        
        status = f"""## 🔧 System Status

**API Server:** {api_status}
**Base URL:** {API_BASE_URL}
**Client Manager:** {"✅ Available" if CLIENT_MANAGER_AVAILABLE else "❌ Not Available"}
**Sklearn:** {"✅ Available" if SKLEARN_AVAILABLE else "❌ Not Available"}

### 🚀 Available Features:
- ✅ Advanced Graph Traversal Table Discovery
- ✅ GPT-4 Powered SQL Generation  
- ✅ Comprehensive SQL Quality Evaluation
- ✅ Multi-metric Assessment (10+ metrics)
- {"✅" if CLIENT_MANAGER_AVAILABLE else "❌"} GPT-4 Quality Assessment
- {"✅" if SKLEARN_AVAILABLE else "❌"} Advanced Similarity Metrics

### 📊 Database Info:
"""
        
        if health_result.get("success", False):
            db_info = health_result.get("database", {})
            status += f"- **Tables:** {db_info.get('table_count', 'N/A')}\n"
            status += f"- **Connected:** {db_info.get('connected', 'N/A')}\n"
            status += f"- **Path:** {db_info.get('path', 'N/A')}\n"
        
        return status
        
    except Exception as e:
        return f"❌ System Status Error: {str(e)}"

def clear_session() -> str:
    """Clear current session data."""
    global current_session
    
    current_session = {
        "query": "",
        "tables": [],
        "selected_tables": [],
        "generated_sql": "",
        "reasoning": "",
        "table_details": {},
        "performance_metrics": {}
    }
    
    return "✅ Session cleared successfully!"

def export_session_data() -> str:
    """Export current session data."""
    global current_session
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"text_to_sql_session_{timestamp}.json"
        
        export_data = {
            "timestamp": timestamp,
            "session_data": current_session.copy()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return f"✅ Session exported to: {filename}"
        
    except Exception as e:
        return f"❌ Export failed: {str(e)}"

# === Gradio Interface ===

def create_text_to_sql_interface():
    """Create the main Gradio interface."""
    
    with gr.Blocks(title="Advanced Text-to-SQL System", theme=gr.themes.Soft()) as demo:
        
        # Header
        gr.Markdown("""
        # 🚀 Advanced Text-to-SQL System
        
        **Intelligent SQL generation with comprehensive quality evaluation using state-of-the-art AI methods**
        
        This system combines Advanced Graph Traversal (GNN + RL + Multi-level) for table discovery 
        with GPT-4 powered SQL generation and comprehensive evaluation using 10+ quality metrics.
        """)
        
        # System Status
        with gr.Row():
            status_display = gr.Markdown(get_system_status())
            
        # Main Interface Tabs
        with gr.Tabs():
            
            # === Tab 1: Step-by-Step Workflow ===
            with gr.TabItem("🔧 Step-by-Step Workflow"):
                gr.Markdown("""
                ### 📋 Step-by-Step Text-to-SQL Generation
                
                Follow these steps to generate and evaluate SQL queries:
                1. **Enter your query** and search for relevant tables
                2. **Review and select tables** that are relevant to your query  
                3. **Generate SQL** from selected tables
                4. **Evaluate SQL quality** against ground truth (optional)
                """)
                
                # Step 1: Table Discovery
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 🔍 Step 1: Table Discovery")
                        query_input = gr.Textbox(
                            label="📝 Enter your natural language query",
                            placeholder="Example: Show me all trades by government entities",
                            lines=3
                        )
                        
                        with gr.Row():
                            max_tables_slider = gr.Slider(
                                minimum=1, maximum=20, value=10, step=1,
                                label="🔢 Maximum tables to find"
                            )
                            search_tables_btn = gr.Button("🔍 Search Tables", variant="primary")
                        
                        # Example queries
                        with gr.Row():
                            gr.Markdown("**Quick Examples:**")
                        with gr.Row():
                            example1_btn = gr.Button("ETD Trades", size="sm")
                            example2_btn = gr.Button("Currency Mismatch", size="sm")
                            example3_btn = gr.Button("Government Entities", size="sm")
                
                    with gr.Column(scale=3):
                        table_search_results = gr.Markdown("Enter a query and click 'Search Tables' to discover relevant tables.")
                
                # Step 2: Table Selection
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📋 Step 2: Table Selection")
                        tables_display = gr.DataFrame(
                            headers=["Table Name", "Type", "Description", "Columns", "Selected"],
                            label="🗄️ Discovered Tables"
                        )
                        
                        selected_tables_input = gr.CheckboxGroup(
                            choices=[],
                            label="✅ Select tables to use for SQL generation",
                            info="Choose which tables should be used for generating the SQL query"
                        )
                        
                        update_selection_btn = gr.Button("✅ Update Selection", variant="secondary")
                        selection_status = gr.Markdown("")
                
                # Step 3: SQL Generation
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 🤖 Step 3: SQL Generation")
                        generate_sql_btn = gr.Button("🚀 Generate SQL", variant="primary", size="lg")
                        
                    with gr.Column(scale=3):
                        sql_generation_results = gr.Markdown("Select tables and click 'Generate SQL' to create the query.")
                
                with gr.Row():
                    with gr.Column():
                        generated_sql_display = gr.Code(
                            label="📝 Generated SQL",
                            language="sql",
                            lines=10
                        )
                    
                    with gr.Column():
                        sql_reasoning_display = gr.Textbox(
                            label="🧠 Generation Reasoning",
                            lines=10,
                            max_lines=15
                        )
                
                # Step 4: SQL Evaluation
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📊 Step 4: SQL Quality Evaluation (Optional)")
                        
                        ground_truth_input = gr.Code(
                            label="📋 Ground Truth SQL (for evaluation)",
                            placeholder="Paste the correct/expected SQL query here for quality evaluation...",
                            language="sql",
                            lines=8
                        )
                        
                        evaluate_sql_btn = gr.Button("📊 Evaluate SQL Quality", variant="secondary", size="lg")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        evaluation_summary = gr.Markdown("Enter ground truth SQL and click 'Evaluate SQL Quality' to assess the generated query.")
                        
                    with gr.Column(scale=3):
                        evaluation_metrics = gr.DataFrame(
                            headers=["Metric", "Score", "Grade", "Status"],
                            label="📈 Evaluation Metrics"
                        )
                
                with gr.Row():
                    evaluation_details = gr.Markdown("", label="📋 Detailed Analysis")
            
            # === Tab 2: One-Click Workflow ===
            with gr.TabItem("⚡ One-Click Generation"):
                gr.Markdown("""
                ### ⚡ Complete Text-to-SQL Pipeline
                
                Generate SQL in one step using the complete pipeline that automatically:
                1. Discovers relevant tables using Advanced Graph Traversal
                2. Generates optimized SQL using GPT-4
                3. Provides detailed reasoning and table information
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        oneclick_query_input = gr.Textbox(
                            label="📝 Enter your query",
                            placeholder="Example: get me the CUSIP that was traded highest last week",
                            lines=3
                        )
                        
                        with gr.Row():
                            oneclick_max_tables = gr.Slider(
                                minimum=1, maximum=15, value=8, step=1,
                                label="🔢 Max tables to discover"
                            )
                            oneclick_include_reasoning = gr.Checkbox(
                                label="🧠 Include reasoning",
                                value=True
                            )
                        
                        oneclick_generate_btn.click(
            fn=complete_text_to_sql,
            inputs=[oneclick_query_input, oneclick_max_tables, oneclick_include_reasoning],
            outputs=[oneclick_results, oneclick_sql_display, oneclick_reasoning_display, oneclick_tables_display]
        )
        
        # One-click evaluation
        def oneclick_evaluate_sql(ground_truth):
            return evaluate_generated_sql(ground_truth)
        
        oneclick_evaluate_btn.click(
            fn=oneclick_evaluate_sql,
            inputs=[oneclick_ground_truth],
            outputs=[oneclick_eval_results, oneclick_eval_metrics, gr.State()]
        )
        
        # Example button handlers - Step by step
        def set_example_query_1():
            return "give me distinct source systems for cash ETD trades for yesterday"
        
        def set_example_query_2():
            return "show me EXECUTING_TRADER_SOEID EXECUTION_VENUE PRODUCT_SK where TRADE_PRICE_CURRENCY and ISSUE_CURRENCY are not the same"
        
        def set_example_query_3():
            return "show me all trades by government entities"
        
        example1_btn.click(fn=set_example_query_1, outputs=[query_input])
        example2_btn.click(fn=set_example_query_2, outputs=[query_input])
        example3_btn.click(fn=set_example_query_3, outputs=[query_input])
        
        # Example button handlers - One-click
        def set_oneclick_example_1():
            return "get me the CUSIP that was traded highest last week"
        
        def set_oneclick_example_2():
            return "give me distinct source systems for cash ETD trades for yesterday"
        
        def set_oneclick_example_3():
            return "Show me the counterparty for trade ID 18871106"
        
        oneclick_ex1_btn.click(fn=set_oneclick_example_1, outputs=[oneclick_query_input])
        oneclick_ex2_btn.click(fn=set_oneclick_example_2, outputs=[oneclick_query_input])
        oneclick_ex3_btn.click(fn=set_oneclick_example_3, outputs=[oneclick_query_input])
        
        # System management handlers
        refresh_status_btn.click(
            fn=get_system_status,
            outputs=[system_info_display]
        )
        
        clear_session_btn.click(
            fn=clear_session,
            outputs=[gr.Textbox(visible=False)]  # Hidden output
        )
        
        def update_session_display():
            return current_session
        
        # Update session display periodically
        refresh_status_btn.click(
            fn=update_session_display,
            outputs=[session_info]
        )
        
        export_session_btn.click(
            fn=export_session_data,
            outputs=[gr.Textbox(visible=False)]  # Hidden output
        )
        
        # Footer
        gr.Markdown("""
        ---
        
        ### 🔬 Technical Architecture
        
        **🧠 Advanced Graph Traversal Method:**
        - **Graph Neural Networks (GNN)** with attention mechanisms for table classification
        - **Reinforcement Learning (RL)** for optimal path discovery and traversal
        - **Multi-level reasoning** combining semantic, structural, and global graph analysis
        - **Ensemble combination** of multiple algorithms for robust table discovery
        
        **🤖 SQL Generation:**
        - **GPT-4 powered** with enhanced schema context and validation
        - **Latest prompting techniques** including chain-of-thought and decomposition
        - **Automatic SQL validation** and error correction
        - **Performance optimization** considerations
        
        **📊 Comprehensive Evaluation:**
        - **10+ state-of-the-art metrics** including latest research from BIRD, Spider 2.0
        - **Execution-based metrics:** EX (Execution Accuracy), VES (Valid Efficiency Score)
        - **Text-based metrics:** BLEU, CodeBLEU, Soft F1-Score
        - **Structural metrics:** Component F1, Syntactic/Semantic Similarity
        - **AI-powered assessment:** GPT-4 quality evaluation with detailed feedback
        
        **🚀 Performance Features:**
        - **Real-time processing** with async API calls
        - **Session management** with export/import capabilities
        - **Interactive UI** with step-by-step and one-click workflows
        - **Comprehensive logging** and error handling
        
        Built with ❤️ using **FastAPI**, **Gradio**, **DuckDB**, **NetworkX**, and **OpenAI GPT-4**.
        
        **Version:** 2.0.0 | **Last Updated:** January 2025
        """)
    
    return demo

# === Additional Utility Functions ===

def validate_api_connection() -> bool:
    """Validate API server connection."""
    try:
        result = call_api_sync("/health", {})
        return result.get("status") == "healthy"
    except:
        return False

def get_sample_queries() -> List[str]:
    """Get sample queries for testing."""
    return [
        "give me distinct source systems for cash ETD trades for yesterday",
        "show me EXECUTING_TRADER_SOEID EXECUTION_VENUE PRODUCT_SK where TRADE_PRICE_CURRENCY and ISSUE_CURRENCY are not the same",
        "Show me the counterparty for trade ID 18871106", 
        "get me the CUSIP that was traded highest last week",
        "show me all trades by government entities",
        "find all ETD trades executed on VENUE_A last week",
        "show me trader performance metrics for Q4",
        "get all currency pairs traded above 1M notional",
        "find all failed trades with error codes",
        "show me settlement instructions for trade batch 12345"
    ]

def format_sql_for_display(sql: str) -> str:
    """Format SQL for better display."""
    try:
        formatted = sqlparse.format(
            sql, 
            reindent=True, 
            keyword_case='upper',
            identifier_case='lower',
            strip_comments=False
        )
        return formatted
    except:
        return sql

def estimate_query_complexity(query: str) -> str:
    """Estimate query complexity level."""
    query_lower = query.lower()
    
    complexity_indicators = {
        'simple': ['select', 'from', 'where'],
        'medium': ['join', 'group by', 'order by', 'having'],
        'complex': ['subquery', 'union', 'window', 'cte', 'recursive']
    }
    
    scores = {'simple': 0, 'medium': 0, 'complex': 0}
    
    for level, indicators in complexity_indicators.items():
        for indicator in indicators:
            if indicator in query_lower:
                scores[level] += 1
    
    if scores['complex'] > 0:
        return "🔴 Complex"
    elif scores['medium'] > 1:
        return "🟡 Medium"
    else:
        return "🟢 Simple"

# === Main Application ===

def main():
    """Main function to launch the Text-to-SQL web application."""
    print("🚀 Starting Advanced Text-to-SQL Web Interface...")
    
    # Validate API connection
    print("🔍 Checking API server connection...")
    if validate_api_connection():
        print("✅ API server connected successfully!")
    else:
        print("⚠️ Warning: Cannot connect to API server. Please ensure the API is running.")
        print(f"   Expected API URL: {API_BASE_URL}")
        print("   Start the API server with: python text_to_sql_api_simple.py")
    
    # Check required libraries
    print("📦 Checking dependencies...")
    print(f"   - Client Manager: {'✅' if CLIENT_MANAGER_AVAILABLE else '❌'}")
    print(f"   - Scikit-learn: {'✅' if SKLEARN_AVAILABLE else '❌'}")
    
    # Create and launch interface
    print("🎨 Creating web interface...")
    demo = create_text_to_sql_interface()
    
    print("✅ Text-to-SQL interface created successfully!")
    print("🌐 Launching on http://localhost:7861")
    print()
    print("📊 Available features:")
    print("   ✅ Advanced Graph Traversal table discovery")
    print("   ✅ GPT-4 powered SQL generation")
    print("   ✅ Comprehensive SQL quality evaluation (10+ metrics)")
    print("   ✅ Step-by-step and one-click workflows")
    print("   ✅ Session management and data export")
    print("   ✅ Real-time performance monitoring")
    print()
    print("🔧 Usage:")
    print("   1. Enter a natural language query")
    print("   2. Review discovered tables and select relevant ones")
    print("   3. Generate SQL query using GPT-4")
    print("   4. Evaluate SQL quality against ground truth (optional)")
    print()
    print("💡 Tips:")
    print("   - Use the step-by-step workflow for detailed control")
    print("   - Try the one-click generation for quick results")
    print("   - Check system status in the System Info tab")
    print("   - Export session data for analysis and reporting")
    
    # Launch with custom settings
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7861,        # Different port from table query UI
        share=False,             # Set to True for public sharing
        show_error=True,
        show_tips=True
    )

if __name__ == "__main__":
    main()btn = gr.Button("🚀 Generate SQL", variant="primary", size="lg")
                        
                        # Quick examples
                        with gr.Row():
                            gr.Markdown("**Quick Examples:**")
                        with gr.Row():
                            oneclick_ex1_btn = gr.Button("Highest CUSIP", size="sm")
                            oneclick_ex2_btn = gr.Button("Source Systems", size="sm")  
                            oneclick_ex3_btn = gr.Button("Counterparty Info", size="sm")
                
                    with gr.Column(scale=3):
                        oneclick_results = gr.Markdown("Enter a query and click 'Generate SQL' for complete text-to-SQL generation.")
                
                with gr.Row():
                    with gr.Column():
                        oneclick_sql_display = gr.Code(
                            label="📝 Generated SQL",
                            language="sql",
                            lines=12
                        )
                    
                    with gr.Column():
                        oneclick_reasoning_display = gr.Textbox(
                            label="🧠 Generation Reasoning",
                            lines=12,
                            max_lines=20
                        )
                
                with gr.Row():
                    oneclick_tables_display = gr.DataFrame(
                        headers=["Table Name", "Type", "Description", "Columns"],
                        label="🗄️ Tables Used"
                    )
                
                # Evaluation section for one-click
                gr.Markdown("### 📊 Optional: Evaluate Generated SQL")
                
                with gr.Row():
                    with gr.Column():
                        oneclick_ground_truth = gr.Code(
                            label="📋 Ground Truth SQL (for evaluation)",
                            placeholder="Paste the expected SQL query here to evaluate quality...",
                            language="sql",
                            lines=6
                        )
                        oneclick_evaluate_btn = gr.Button("📊 Evaluate Quality", variant="secondary")
                    
                    with gr.Column():
                        oneclick_eval_results = gr.Markdown("")
                        oneclick_eval_metrics = gr.DataFrame(
                            headers=["Metric", "Score", "Grade", "Status"],
                            label="📈 Quality Metrics"
                        )
            
            # === Tab 3: System Information ===
            with gr.TabItem("ℹ️ System Info"):
                gr.Markdown("""
                ### 🔧 System Information & Configuration
                
                Monitor system status, view evaluation metrics information, and manage session data.
                """)
                
                with gr.Row():
                    refresh_status_btn = gr.Button("🔄 Refresh Status", variant="secondary")
                    clear_session_btn = gr.Button("🗑️ Clear Session", variant="secondary") 
                    export_session_btn = gr.Button("💾 Export Session", variant="secondary")
                
                with gr.Row():
                    system_info_display = gr.Markdown(get_system_status())
                
                gr.Markdown("""
                ### 📊 SQL Evaluation Metrics Explained
                
                Our system uses **10+ state-of-the-art metrics** for comprehensive SQL quality assessment:
                
                #### 🎯 Core Accuracy Metrics:
                - **Execution Accuracy (EX):** Measures if the generated SQL produces the same results as ground truth
                - **Exact Match (EM):** Checks for identical SQL structure after normalization
                - **Component Match (F1):** Evaluates individual SQL components (SELECT, FROM, WHERE, etc.)
                
                #### 🔤 Text-Based Metrics:
                - **BLEU Score:** Adapted from machine translation, measures n-gram overlap
                - **CodeBLEU:** Enhanced BLEU specifically designed for code evaluation with syntax awareness
                - **Soft F1-Score:** BIRD benchmark style metric with fuzzy entity matching
                
                #### 🏗️ Structure & Semantic Metrics:
                - **Syntactic Similarity:** Compares SQL structure and clause organization
                - **Semantic Similarity:** Uses TF-IDF and cosine similarity for meaning comparison
                - **Valid Efficiency Score (VES):** Combines correctness with performance optimization
                
                #### 🤖 AI-Powered Assessment:
                - **GPT-4 Quality Assessment:** Comprehensive evaluation by GPT-4 with detailed feedback
                
                #### 📈 Scoring System:
                - **Overall Score:** Weighted combination of all metrics (0.0 - 1.0)
                - **Letter Grades:** A (0.9+), B (0.8+), C (0.7+), D (0.6+), F (<0.6)
                - **Recommendations:** Actionable suggestions for improvement
                """)
                
                with gr.Row():
                    session_info = gr.JSON(
                        label="📋 Current Session Data",
                        value=current_session
                    )
        
        # === Event Handlers ===
        
        # Step-by-step workflow handlers
        search_tables_btn.click(
            fn=text_to_table_search,
            inputs=[query_input, max_tables_slider],
            outputs=[table_search_results, tables_display, gr.State()]
        )
        
        # Update table selection choices when tables are found
        def update_table_choices(df):
            if df is not None and not df.empty:
                choices = df["Table Name"].tolist() if "Table Name" in df.columns else []
                return gr.CheckboxGroup(choices=choices, value=choices)
            return gr.CheckboxGroup(choices=[], value=[])
        
        tables_display.change(
            fn=update_table_choices,
            inputs=[tables_display],
            outputs=[selected_tables_input]
        )
        
        update_selection_btn.click(
            fn=update_table_selection,
            inputs=[selected_tables_input],
            outputs=[selection_status]
        )
        
        generate_sql_btn.click(
            fn=generate_sql_from_tables,
            inputs=[selected_tables_input],
            outputs=[sql_generation_results, generated_sql_display, sql_reasoning_display]
        )
        
        evaluate_sql_btn.click(
            fn=evaluate_generated_sql,
            inputs=[ground_truth_input],
            outputs=[evaluation_summary, evaluation_metrics, evaluation_details]
        )
        
        # One-click workflow handlers
        oneclick_generate_