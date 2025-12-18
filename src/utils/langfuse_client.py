"""
Langfuse integration for comprehensive workflow tracing and observability.
"""

import os
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

# Try to import langfuse, provide mocks if not available
try:
    from langfuse import Langfuse, observe
    LANGFUSE_AVAILABLE = True
except (ImportError, Exception) as e:
    # If langfuse fails to import (e.g., due to pydantic compatibility issues)
    LANGFUSE_AVAILABLE = False

    # Mock implementations
    class MockLangfuse:
        def __init__(self, *args, **kwargs):
            pass

        def trace(self, *args, **kwargs):
            return MockTrace()

        def flush(self):
            pass

    class MockTrace:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def span(self, *args, **kwargs):
            return MockSpan()

        def update(self, *args, **kwargs):
            pass

    class MockSpan:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, *args, **kwargs):
            pass

    def observe(*args, **kwargs):
        """Mock observe decorator that does nothing."""
        def decorator(func):
            return func
        return decorator

    Langfuse = MockLangfuse

    # Mock implementations
    class MockLangfuse:
        def __init__(self, *args, **kwargs):
            pass

        def trace(self, *args, **kwargs):
            return MockTrace()

        def flush(self):
            pass

    class MockTrace:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def span(self, *args, **kwargs):
            return MockSpan()

        def update(self, *args, **kwargs):
            pass

    class MockSpan:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, *args, **kwargs):
            pass

    def observe(*args, **kwargs):
        """Mock observe decorator that does nothing."""
        def decorator(func):
            return func
        return decorator

    Langfuse = MockLangfuse

from src.config import settings


class LangfuseClient:
    """
    Wrapper for Langfuse client with enhanced tracing capabilities.
    Uses Langfuse's @observe decorator pattern for proper parent-child relationships.
    """

    def __init__(self):
        """Initialize Langfuse client with configuration."""
        self.client = None
        self.enabled = self._initialize_client()

    def _initialize_client(self) -> bool:
        """Initialize the Langfuse client with proper configuration."""
        try:
            # Check if credentials are available
            secret_key = settings.langfuse_secret_key or os.getenv(
                'LANGFUSE_SECRET_KEY')
            public_key = settings.langfuse_public_key or os.getenv(
                'LANGFUSE_PUBLIC_KEY')
            host = settings.langfuse_base_url or os.getenv('LANGFUSE_BASE_URL')

            if not all([secret_key, public_key, host]):
                print("Langfuse credentials not found. Observability disabled.")
                return False

            # Initialize client with correct parameters
            self.client = Langfuse(
                secret_key=secret_key,
                public_key=public_key,
                host=host  # Note: parameter is 'host', not 'base_url'
            )

            print("Langfuse client initialized successfully")
            return True

        except Exception as e:
            print(f"Failed to initialize Langfuse client: {e}")
            print("Continuing without observability...")
            return False

    def create_trace(
        self,
        name: str,
        input: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Create a new trace for tracking a complete workflow.
        Uses Langfuse's @observe decorator on the context manager for proper hierarchy.
        """
        if not self.enabled or not self.client:
            print("📝 [Langfuse] Creating trace (dummy mode)")
            return self._create_dummy_trace()

        # Include user_id in metadata if provided
        trace_metadata = metadata or {}
        if user_id:
            trace_metadata['user_id'] = user_id

        # Create trace context that will create the root trace with @observe
        trace_context = LangfuseTraceContext(
            name=name,
            input=input,
            user_id=user_id,
            metadata=trace_metadata,
            client=self.client
        )

        return trace_context

    def _create_dummy_trace(self):
        """Create a dummy trace object for backward compatibility."""
        class DummyTrace:
            def span(self, **kwargs):
                return self._create_dummy_observation()

            def generation(self, **kwargs):
                return self._create_dummy_observation()

            def event(self, **kwargs):
                return self._create_dummy_observation()

            def update(self, **kwargs):
                pass

            def __getattr__(self, name):
                return lambda *args, **kwargs: self._create_dummy_observation()

            def _create_dummy_observation(self):
                class DummyObservation:
                    def update(self, **kwargs):
                        pass

                    def end(self):
                        pass

                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        pass
                return DummyObservation()

        return DummyTrace()

    def add_observation(
        self,
        trace,
        name: str,
        observation_type: str = "span",
        input: Optional[str] = None,
        output: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ):
        """Add an observation to an existing trace using the new trace context methods."""
        if not self.enabled or not self.client or not trace:
            return None

        try:
            # Use the trace context's methods for proper parent-child relationships
            if observation_type == "span":
                return trace.span(name=name, input=input, output=output, metadata=metadata)
            elif observation_type == "generation":
                return trace.generation(name=name, input=input, output=output, metadata=metadata)
            elif observation_type == "event":
                return trace.event(name=name, input=input, output=output, metadata=metadata)
            else:
                # Fallback to span if observation type not found
                return trace.span(name=name, input=input, output=output, metadata=metadata)

        except Exception as e:
            print(f"❌ [Langfuse] Failed to add {observation_type}: {e}")
            return None

    def log_agent_execution(
        self,
        trace,
        agent_name: str,
        agent_type: str,
        input_data: str,
        output_data: str,
        execution_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log agent execution with performance metrics using agent observation type."""
        agent_metadata = {
            "agent_name": agent_name,
            "agent_type": agent_type,
            "execution_time_seconds": execution_time,
            "input_length": len(input_data) if input_data else 0,
            "output_length": len(output_data) if output_data else 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "performance_tier": "fast" if execution_time < 1.0 else "medium" if execution_time < 3.0 else "slow"
        }

        if metadata:
            agent_metadata.update(metadata)

        return self.add_observation(
            trace=trace,
            name=f"{agent_name}_workflow",
            observation_type="agent",  # Use proper agent observation type
            input=input_data[:1000] +
            "..." if input_data and len(input_data) > 1000 else input_data,
            output=output_data[:1000] +
            "..." if output_data and len(
                output_data) > 1000 else output_data,
            metadata=agent_metadata
        )

    def log_classification_result(
        self,
        trace,
        query: str,
        predicted_class: str,
        confidence: float,
        all_scores: Dict[str, float]
    ):
        """Log intent classification results with confidence scores."""
        # Calculate additional metrics
        max_score = max(all_scores.values()) if all_scores else 0.0
        score_gap = max_score - \
            (sorted(all_scores.values())[-2] if len(all_scores) > 1 else 0.0)
        is_high_confidence = confidence > settings.confidence_threshold

        return self.add_observation(
            trace=trace,
            name="intent_classification",
            observation_type="event",
            input=query[:500] + "..." if query and len(query) > 500 else query,
            output=predicted_class,
            metadata={
                "confidence": confidence,
                "max_score": max_score,
                "score_gap": score_gap,
                "all_scores": all_scores,
                "threshold": settings.confidence_threshold,
                "classification_method": "semantic_similarity",
                "is_high_confidence": is_high_confidence,
                "alternative_departments": sorted(all_scores.keys(), key=lambda x: all_scores[x], reverse=True)[1:3] if len(all_scores) > 1 else []
            }
        )

    def log_rag_retrieval(
        self,
        trace,
        query: str,
        retrieved_docs: List[str],
        similarity_scores: List[float],
        retrieval_time: float
    ):
        """Log RAG document retrieval results using retriever observation type."""
        # Calculate detailed metrics
        avg_similarity = sum(similarity_scores) / \
            len(similarity_scores) if similarity_scores else 0.0
        max_similarity = max(similarity_scores) if similarity_scores else 0.0
        min_similarity = min(similarity_scores) if similarity_scores else 0.0

        # Categorize retrieval quality
        if avg_similarity > 0.8:
            quality_tier = "excellent"
        elif avg_similarity > 0.6:
            quality_tier = "good"
        elif avg_similarity > 0.4:
            quality_tier = "fair"
        else:
            quality_tier = "poor"

        # Create document summaries for metadata
        doc_summaries = []
        # Only include first 3 docs
        for i, doc in enumerate(retrieved_docs[:3]):
            similarity_score = similarity_scores[i] if i < len(
                similarity_scores) else 0.0
            doc_preview = doc[:100] + "..." if doc and len(doc) > 100 else doc
            doc_summaries.append({
                "doc_index": i,
                "similarity_score": similarity_score,
                "preview": doc_preview
            })

        return self.add_observation(
            trace=trace,
            name="vector_search",
            observation_type="retriever",  # Use proper retriever observation type
            input={
                "query": query[:500] + "..." if query and len(query) > 500 else query,
                "top_k": len(retrieved_docs)
            },
            output={
                "documents": retrieved_docs,
                "scores": similarity_scores
            },
            metadata={
                "num_documents": len(retrieved_docs),
                "similarity_scores": similarity_scores,
                "average_similarity": avg_similarity,
                "max_similarity": max_similarity,
                "min_similarity": min_similarity,
                "retrieval_time_seconds": retrieval_time,
                "embedding_model": settings.embedding_model,
                "quality_tier": quality_tier,
                "document_previews": doc_summaries,
                "similarity_distribution": {
                    "high_confidence_docs": sum(1 for s in similarity_scores if s > 0.8),
                    "medium_confidence_docs": sum(1 for s in similarity_scores if 0.5 <= s <= 0.8),
                    "low_confidence_docs": sum(1 for s in similarity_scores if s < 0.5)
                },
                "vector_store": "faiss",
                "similarity_metric": "cosine"
            }
        )

    def log_quality_evaluation(
        self,
        trace,
        query: str,
        answer: str,
        context: str,
        quality_scores: Dict[str, float],
        overall_score: float
    ):
        """Log response quality evaluation results using evaluator observation type."""
        # Categorize overall quality
        if overall_score >= 8.5:
            quality_tier = "excellent"
        elif overall_score >= 7.0:
            quality_tier = "good"
        elif overall_score >= 5.0:
            quality_tier = "acceptable"
        else:
            quality_tier = "needs_improvement"

        return self.add_observation(
            trace=trace,
            name="response_quality_evaluator",
            observation_type="evaluator",  # Use proper evaluator observation type
            input={
                "query": query[:500] + "..." if query and len(query) > 500 else query,
                "context": context[:200] + "..." if context and len(context) > 200 else context,
                "answer": answer[:1000] + "..." if answer and len(answer) > 1000 else answer
            },
            output={
                "overall_score": overall_score,
                "quality_tier": quality_tier,
                "verdict": "acceptable" if overall_score >= 7.0 else "needs_improvement"
            },
            metadata={
                "context_length": len(context),
                "answer_length": len(answer),
                "quality_scores": quality_scores,
                "evaluation_dimensions": settings.quality_dimensions,
                "evaluator_model": settings.evaluator_model,
                "weakest_dimension": min(quality_scores.keys(), key=lambda x: quality_scores[x]) if quality_scores else None,
                "strongest_dimension": max(quality_scores.keys(), key=lambda x: quality_scores[x]) if quality_scores else None,
                "score_variance": max(quality_scores.values()) - min(quality_scores.values()) if quality_scores else 0.0,
                "evaluation_timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    def log_embedding_generation(
        self,
        trace,
        texts: List[str],
        model_name: str,
        embeddings: List[List[float]],
        token_usage: Optional[Dict[str, int]] = None,
        generation_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log embedding generation results using embedding observation type."""
        embedding_metadata = {
            "model_name": model_name,
            "num_texts": len(texts),
            "embedding_dimension": len(embeddings[0]) if embeddings else 0,
            "total_embeddings": len(embeddings),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        if generation_time:
            embedding_metadata["generation_time_seconds"] = generation_time
            embedding_metadata["texts_per_second"] = len(
                texts) / generation_time if generation_time > 0 else 0

        if token_usage:
            embedding_metadata["token_usage"] = token_usage

        if metadata:
            embedding_metadata.update(metadata)

        return self.add_observation(
            trace=trace,
            name=f"{model_name}_embedding",
            observation_type="embedding",  # Use proper embedding observation type
            input=texts,
            output=embeddings,
            metadata=embedding_metadata
        )

    def log_chain_execution(
        self,
        trace,
        chain_name: str,
        input_data: Any,
        output_data: Any,
        steps: List[Dict[str, Any]],
        execution_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log chain execution results using chain observation type."""
        chain_metadata = {
            "chain_name": chain_name,
            "num_steps": len(steps),
            "execution_time_seconds": execution_time,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step_names": [step.get("name", "unknown") for step in steps],
            "performance_tier": "fast" if execution_time < 2.0 else "medium" if execution_time < 5.0 else "slow"
        }

        if metadata:
            chain_metadata.update(metadata)

        return self.add_observation(
            trace=trace,
            name=chain_name,
            observation_type="chain",  # Use proper chain observation type
            input=input_data,
            output=output_data,
            metadata={
                **chain_metadata,
                "steps": steps
            }
        )

    def log_error(
        self,
        trace,
        error_message: str,
        error_type: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log error events for debugging and monitoring."""
        return self.add_observation(
            trace=trace,
            name="error",
            observation_type="event",
            input=str(context)[:500] if context else "",
            output=error_message,
            metadata={
                "error_type": error_type,
                "error_severity": self._categorize_error_severity(error_type),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error_category": self._categorize_error_type(error_type),
                **(context or {})
            }
        )

    def _categorize_error_severity(self, error_type: str) -> str:
        """Categorize error severity based on error type."""
        critical_errors = ["timeout", "connection_error",
                           "api_key_error", "authentication_error"]
        warning_errors = ["low_confidence",
                          "fallback_used", "retrieval_failed"]

        if any(critical in error_type.lower() for critical in critical_errors):
            return "critical"
        elif any(warning in error_type.lower() for warning in warning_errors):
            return "warning"
        else:
            return "info"

    def _categorize_error_type(self, error_type: str) -> str:
        """Categorize error type for better debugging."""
        if "classification" in error_type.lower():
            return "intent_classification"
        elif "retrieval" in error_type.lower():
            return "document_retrieval"
        elif "llm" in error_type.lower() or "generation" in error_type.lower():
            return "llm_generation"
        elif "agent" in error_type.lower():
            return "agent_processing"
        elif "orchestrator" in error_type.lower():
            return "orchestration"
        else:
            return "general"

    def log_llm_call(
        self,
        trace,
        model_name: str,
        prompt: str,
        response: str,
        token_usage: Dict[str, int],
        response_time: float,
        temperature: float = None,
        max_tokens: int = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log LLM API calls with detailed metrics using generation observation type."""
        # Calculate cost estimation (rough estimate)
        estimated_cost = self._estimate_llm_cost(model_name, token_usage)

        generation_metadata = {
            "model_name": model_name,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "response_time_seconds": response_time,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "estimated_cost_usd": estimated_cost,
            "performance_tier": "fast" if response_time < 2.0 else "medium" if response_time < 5.0 else "slow",
            "tokens_per_second": (token_usage.get("total_tokens", 0) / response_time) if response_time > 0 else 0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Prepare usage details for Langfuse
        usage_details = None
        if token_usage.get("prompt_tokens") and token_usage.get("completion_tokens"):
            usage_details = {
                "input": token_usage.get("prompt_tokens", 0),
                "output": token_usage.get("completion_tokens", 0),
                "total": token_usage.get("total_tokens", 0)
            }

        if metadata:
            generation_metadata.update(metadata)

        return self.add_observation(
            trace=trace,
            name=f"{model_name}_generation",
            observation_type="generation",  # Use proper generation observation type
            input=prompt[:2000] +
            "..." if prompt and len(prompt) > 2000 else prompt,
            output=response[:2000] +
            "..." if response and len(response) > 2000 else response,
            metadata=generation_metadata,
            # Note: In real implementation, you'd set usage directly on the Langfuse observation
            # This would require updating the actual Langfuse SDK integration
        )

    def _estimate_llm_cost(self, model_name: str, token_usage: Dict[str, int]) -> float:
        """Estimate cost for LLM call in USD (rough estimates)."""
        # Rough cost per 1M tokens (update with actual pricing)
        cost_per_million = {
            "gpt-3.5-turbo": 0.50,
            "gpt-4": 30.0,
            "gpt-4-turbo": 10.0,
            "anthropic-claude-3-sonnet": 15.0,
            "anthropic-claude-3-haiku": 1.0,
        }

        # Extract model family
        model_family = None
        for family in cost_per_million.keys():
            if family in model_name.lower():
                model_family = family
                break

        if not model_family:
            return 0.0  # Unknown model, can't estimate cost

        cost_per_token = cost_per_million[model_family] / 1_000_000
        total_tokens = token_usage.get("total_tokens", 0)

        return total_tokens * cost_per_token

    def log_function_execution(
        self,
        trace,
        function_name: str,
        function_args: Dict[str, Any],
        function_result: Any,
        execution_success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log function execution results as a tool observation."""
        execution_metadata = {
            "function_name": function_name,
            "function_args": function_args,
            "execution_success": execution_success,
            "result_type": type(function_result).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        if metadata:
            execution_metadata.update(metadata)

        return self.add_observation(
            trace=trace,
            name=function_name,
            observation_type="tool",  # Use proper tool observation type
            input=str(function_args)[:500],
            output=str(function_result)[:1000] + "..." if str(function_result) and len(
                str(function_result)) > 1000 else str(function_result),
            metadata=execution_metadata
        )

    def flush(self) -> None:
        """Flush any pending traces to Langfuse"""
        if self.enabled and self.client:
            try:
                if hasattr(self.client, 'flush'):
                    self.client.flush()
                else:
                    print("Langfuse client does not have flush method available")
            except Exception as e:
                print(f"Failed to flush Langfuse: {e}")


class LangfuseTraceContext:
    """
    Trace context that creates proper parent-child relationships.
    Uses a single @observe decorated method as the root, with nested observations as children.
    """

    def __init__(self, name: str, input: str, user_id: str, metadata: Dict[str, Any], client):
        self.name = name
        self.input = input
        self.user_id = user_id
        self.metadata = metadata
        self.client = client
        self.observations = []

    @observe(name="multi_agent_query_processing")
    def _root_trace_execution(self, func):
        """
        Root trace execution method decorated with @observe.
        All nested calls within this context will automatically become child observations.
        """
        return func

    def execute_with_trace(self, func, *args, **kwargs):
        """
        Execute a function within the @observe decorated root trace context.
        This creates the parent trace and ensures all nested calls become children.
        """
        decorated_func = self._root_trace_execution(func)
        return decorated_func(*args, **kwargs)

    def span(self, name: str, input: Optional[str] = None, output: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Create a span observation as a child of the current trace.
        Uses @observe decorator to automatically link as a child.
        """
        return LangfuseObservationContext("span", name, input, output, metadata, None, self.client)

    def generation(self, name: str, input: Optional[str] = None, output: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Create a generation observation as a child of the current trace.
        Uses @observe decorator to automatically link as a child.
        """
        print(f"🎯 [Langfuse] Creating child generation: {name}")
        return LangfuseObservationContext("generation", name, input, output, metadata, None, self.client)

    def event(self, name: str, input: Optional[str] = None, output: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Create an event observation as a child of the current trace.
        Uses @observe decorator to automatically link as a child.
        """
        print(f"📝 [Langfuse] Creating child event: {name}")
        return LangfuseObservationContext("event", name, input, output, metadata, None, self.client)

    @property
    def enabled(self) -> bool:
        """Check if tracing is enabled."""
        return hasattr(self, 'client') and self.client is not None

    def _create_dummy_observation(self):
        """Create a dummy observation for fallback mode."""
        class DummyObservation:
            def update(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass
        return DummyObservation()

    def update(self, **kwargs):
        """Update the trace with new data."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class LangfuseObservationContext:
    """
    Simple context manager for @observe decorated methods.
    The @observe decorator handles all the trace management automatically.
    """

    def __init__(self, observation_type: str, name: str, input: Optional[str] = None, output: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, observation_data=None, client=None):
        self.observation_type = observation_type
        self.name = name
        self.input = input
        self.output = output
        self.metadata = metadata or {}
        self.start_time = datetime.now(timezone.utc)

    def update(self, output: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Update the observation with new data."""
        if output is not None:
            self.output = output
        if metadata:
            self.metadata.update(metadata)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# Global Langfuse client instance
langfuse_client = LangfuseClient()
