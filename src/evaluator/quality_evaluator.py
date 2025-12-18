"""
Quality evaluator agent for response scoring and assessment.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

from openai import OpenAI

from src.config import settings
from src.utils import langfuse_client


@dataclass
class EvaluationResult:
    """Dataclass for quality evaluation results."""
    overall_score: float  # 1-10 scale
    dimension_scores: Dict[str, float]  # Individual dimension scores
    reasoning: str  # LLM reasoning for the scores
    recommendations: List[str]  # Improvement recommendations


class ResponseQualityEvaluator:
    """
    Automated evaluator that assesses RAG response quality across multiple dimensions.
    Uses LLM-based evaluation to score relevance, completeness, and accuracy.
    """

    def __init__(self, model_name: str = None):
        """
        Initialize the quality evaluator.

        Args:
            model_name: Name of the evaluation model
        """
        self.model_name = model_name or settings.evaluator_model
        self._llm_client = None
        self._initialized = False

        # Evaluation dimensions with detailed descriptions
        self.evaluation_dimensions = {
            'relevance': {
                'description': 'How well does the answer directly address the user\'s question?',
                'weight': 0.4
            },
            'completeness': {
                'description': 'How comprehensive is the answer? Does it cover all aspects of the question?',
                'weight': 0.3
            },
            'accuracy': {
                'description': 'Is the information provided accurate and consistent with the context?',
                'weight': 0.3
            }
        }

    async def initialize(self) -> None:
        """
        Initialize the evaluator.
        """
        try:
            print("Initializing Response Quality Evaluator...")

            # Initialize LLM client for evaluation
            base_url = settings.openrouter_base_url or "https://openrouter.ai/api/v1"
            self._llm_client = OpenAI(
                api_key=settings.openrouter_api_key,
                base_url=base_url
            )

            self._initialized = True
            print("Response Quality Evaluator initialized successfully")

        except Exception as e:
            print(f"Error initializing quality evaluator: {e}")
            raise

    async def evaluate_response(
        self,
        query: str,
        answer: str,
        context: str,
        source_documents: List[dict],
        trace=None
    ) -> EvaluationResult:
        """
        Evaluate a response across multiple quality dimensions.

        Args:
            query: Original user query
            answer: Generated answer
            context: Retrieved context (concatenated source documents)
            source_documents: List of source document information
            trace: Optional Langfuse trace for observability

        Returns:
            EvaluationResult with scores and recommendations
        """
        if not self._initialized:
            raise RuntimeError(
                "Evaluator not initialized. Call initialize() first.")

        start_time = time.time()

        try:
            # Generate evaluation prompt
            evaluation_prompt = self._create_evaluation_prompt(
                query, answer, context, source_documents
            )

            # Get evaluation from LLM
            evaluation_response = await self._call_evaluation_llm(evaluation_prompt)

            # Parse evaluation response
            evaluation_result = self._parse_evaluation_response(
                evaluation_response)

            # Log evaluation with Langfuse
            if trace:
                langfuse_client.log_quality_evaluation(
                    trace=trace,
                    query=query,
                    answer=answer,
                    context=context,
                    quality_scores=evaluation_result.dimension_scores,
                    overall_score=evaluation_result.overall_score
                )

                # Log evaluation execution
                langfuse_client.log_agent_execution(
                    trace=trace,
                    agent_name="quality_evaluator",
                    agent_type="evaluator",
                    input_data=f"Query: {query[:100]}...",
                    output_data=f"Score: {evaluation_result.overall_score}/10",
                    execution_time=time.time() - start_time,
                    metadata={
                        'evaluation_dimensions': list(self.evaluation_dimensions.keys()),
                        'num_source_documents': len(source_documents),
                        'recommendations_count': len(evaluation_result.recommendations)
                    }
                )

            return evaluation_result

        except Exception as e:
            error_msg = f"Error during quality evaluation: {e}"

            # Log error with Langfuse
            if trace:
                langfuse_client.log_error(
                    trace=trace,
                    error_message=error_msg,
                    error_type="quality_evaluation_error",
                    context={
                        'query_length': len(query),
                        'answer_length': len(answer)
                    }
                )

            # Return default evaluation
            return EvaluationResult(
                overall_score=5.0,
                dimension_scores={
                    dim: 5.0 for dim in self.evaluation_dimensions.keys()},
                reasoning=f"Evaluation failed due to error: {str(e)}",
                recommendations=[
                    "Manual review recommended due to evaluation error"]
            )

    def _create_evaluation_prompt(
        self,
        query: str,
        answer: str,
        context: str,
        source_documents: List[dict]
    ) -> str:
        """
        Create a comprehensive evaluation prompt for the LLM.

        Args:
            query: Original user query
            answer: Generated answer
            context: Retrieved context
            source_documents: List of source documents

        Returns:
            Evaluation prompt string
        """
        dimensions_desc = "\n".join([
            f"- {name}: {details['description']} (Weight: {details['weight']})"
            for name, details in self.evaluation_dimensions.items()
        ])

        prompt = f"""You are an expert evaluator for AI-generated responses in a corporate setting. Your task is to evaluate the quality of a response to a user query based on retrieved company documentation.

QUERY: {query}

RESPONSE TO EVALUATE: {answer}

CONTEXT INFORMATION:
{context}

SOURCE DOCUMENTS: {len(source_documents)} documents retrieved
{chr(10).join([f"- {doc.get('source', 'unknown')} (Similarity: {doc.get('similarity_score', 'N/A')})" for doc in source_documents[:3]])}

EVALUATION DIMENSIONS:
{dimensions_desc}

EVALUATION CRITERIA:
1. Score each dimension on a scale of 1-10 (where 1 = poor, 10 = excellent)
2. Consider:
   - Does the answer directly address the user's question?
   - Is the information accurate based on the provided context?
   - Is the answer comprehensive and complete?
   - Does the answer acknowledge information gaps when present?
   - Is the tone appropriate for a corporate environment?

3. Provide specific recommendations for improvement if the score is below 7 in any dimension

Please provide your evaluation in the following JSON format:
{{
    "relevance": <score 1-10>,
    "completeness": <score 1-10>,
    "accuracy": <score 1-10>,
    "reasoning": "<detailed reasoning for your scores>",
    "recommendations": ["<specific improvement recommendations>"]
}}"""

        return prompt

    async def _call_evaluation_llm(self, prompt: str) -> str:
        """
        Call the LLM for evaluation.

        Args:
            prompt: Evaluation prompt

        Returns:
            LLM response string
        """
        if not self._llm_client:
            raise RuntimeError("LLM client not initialized")

        try:
            response = self._llm_client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com/estebmaister/henry_bot_M4",
                    "X-Title": "henry_bot_M4-QualityEvaluator"
                },
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator. Provide detailed, objective assessments. Always respond in valid JSON format."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1  # Low temperature for consistent evaluation
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error calling evaluation LLM: {e}")
            raise

    def _parse_evaluation_response(self, response: str) -> EvaluationResult:
        """
        Parse the LLM evaluation response into structured data.

        Args:
            response: Raw LLM response

        Returns:
            Parsed EvaluationResult
        """
        try:
            # Try to extract JSON from the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                evaluation_data = json.loads(json_str)
            else:
                # Fallback if JSON parsing fails
                print(
                    "Warning: Could not parse JSON from evaluation response, using defaults")
                evaluation_data = {
                    'relevance': 5.0,
                    'completeness': 5.0,
                    'accuracy': 5.0,
                    'reasoning': 'Unable to parse detailed reasoning',
                    'recommendations': ['Manual review recommended']
                }

            # Calculate weighted overall score
            overall_score = 0.0
            for dimension, score in evaluation_data.items():
                if dimension in self.evaluation_dimensions:
                    weight = self.evaluation_dimensions[dimension]['weight']
                    overall_score += score * weight

            # Normalize to 1-10 scale
            overall_score = min(10.0, max(1.0, overall_score))

            return EvaluationResult(
                overall_score=round(overall_score, 1),
                dimension_scores={
                    dim: round(evaluation_data.get(dim, 5.0), 1)
                    for dim in self.evaluation_dimensions.keys()
                },
                reasoning=evaluation_data.get(
                    'reasoning', 'No reasoning provided'),
                recommendations=evaluation_data.get('recommendations', [])
            )

        except Exception as e:
            print(f"Error parsing evaluation response: {e}")
            return EvaluationResult(
                overall_score=5.0,
                dimension_scores={
                    dim: 5.0 for dim in self.evaluation_dimensions.keys()},
                reasoning=f"Error parsing evaluation: {str(e)}",
                recommendations=[
                    'Manual review recommended due to parsing error']
            )

    def is_initialized(self) -> bool:
        """
        Check if the evaluator is initialized.

        Returns:
            True if initialized, False otherwise
        """
        return self._initialized
