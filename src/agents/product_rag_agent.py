"""
Product RAG Agent for M4 delivery chatbot.

AI Assistant Notes:
- Transforms the existing RAG architecture for product catalog searches
- Uses FAISS with product embeddings for semantic product discovery
- Integrates with database for real-time product information
- Provides conversational product recommendations and comparisons
- Maintains product availability and pricing accuracy
"""

from ..utils import langfuse_client
from ..config import settings
from ..retrievers import FAISSRetriever, RetrievedDocument
from ..database import ProductRepository
from ..database.models import ProductModel, ProductCategory
from .base import BaseAgent, AgentResponse
from src.utils import observe
from openai import OpenAI
from typing import List, Dict, Any, Optional
import time
import logging

logger = logging.getLogger(__name__)


class ProductRAGAgent(BaseAgent):
    """
    Retrieval-Augmented Generation agent specialized for product catalog queries.
    Combines product database information with semantic search for accurate, up-to-date product responses.
    """

    def __init__(
        self,
        name: str = "Product Assistant",
        model_name: str = None,
        similarity_top_k: int = None
    ):
        """
        Initialize Product RAG agent with configuration.

        Args:
            name: Name of the agent
            model_name: LLM model name
            similarity_top_k: Number of products to retrieve
        """
        super().__init__(
            name=name,
            department="products",
            model_name=model_name or settings.model_name
        )

        self.similarity_top_k = similarity_top_k or settings.similarity_top_k

        # Initialize product retriever with persistent storage
        self._retriever = FAISSRetriever(
            embedding_model=settings.embedding_model,
            similarity_top_k=self.similarity_top_k,
            department_name="products",
            use_persistent_storage=settings.use_persistent_storage,
            force_rebuild=settings.force_rebuild_indices
        )

        # Initialize product repository for database operations
        self._product_repo = ProductRepository(
            self._retriever.db if hasattr(self._retriever, 'db') else None)

        # Initialize OpenAI client
        self._llm_client = None

    async def initialize(self, products_path: str = None) -> None:
        """
        Initialize the agent and its components.

        Args:
            products_path: Path to product documents for indexing
        """
        try:
            logger.info(
                f"Initializing {self.name} agent for product catalog...")

            # Initialize product retriever
            if products_path:
                await self._retriever.initialize(products_path)
            else:
                # Initialize with default product data path
                await self._retriever.initialize("data/products")

            # Initialize database if not already done or if connection is None
            if not self._product_repo or (hasattr(self._product_repo, 'db') and not self._product_repo.db):
                from ..database import DatabaseConnection, DatabaseMigrations
                db = DatabaseConnection()
                migrations = DatabaseMigrations(db)
                migrations.run_migrations()
                self._product_repo = ProductRepository(db)

            # Initialize LLM client
            base_url = settings.openrouter_base_url or "https://openrouter.ai/api/v1"
            self._llm_client = OpenAI(
                api_key=settings.openrouter_api_key,
                base_url=base_url
            )

            self._initialized = True
            logger.info(f"{self.name} agent initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing {self.name} agent: {e}")
            raise

    @observe(name="product_rag_agent_processing", as_type="retriever")
    async def process_query(self, query: str, trace=None) -> AgentResponse:
        """
        Process a product query and return a specialized response.

        Args:
            query: The input product query to process
            trace: Optional Langfuse trace for observability

        Returns:
            AgentResponse with product information, confidence, and metadata
        """
        if not self._initialized:
            raise RuntimeError(
                f"Agent {self.name} not initialized. Call initialize() first.")

        start_time = time.time()

        try:
            # Create comprehensive product RAG processing span
            if trace:
                with trace.span(
                    name="product_rag_pipeline",
                    input=query,
                    metadata={
                        'stage': 'product_rag_processing',
                        'agent_name': self.name,
                        'embedding_model': settings.embedding_model,
                        'query_type': 'product_inquiry'
                    }
                ) as rag_pipeline_span:

                    # Product Retrieval with span
                    with trace.span(
                        name="product_retrieval",
                        input=query,
                        metadata={
                            'stage': 'product_search',
                            'similarity_top_k': self.similarity_top_k,
                            'retriever_type': 'faiss',
                            'product_categories': ['electronics', 'books', 'clothing', 'home_goods']
                        }
                    ) as retrieval_span:

                        retrieval_start = time.time()
                        retrieved_docs = await self._retriever.retrieve(query)

                        # Enhance with real-time database information
                        enhanced_products = await self._enhance_products_with_db_data(retrieved_docs)

                        retrieval_time = time.time() - retrieval_start

                        # Update retrieval span with results
                        retrieval_span.update(
                            output=f"Retrieved {len(enhanced_products)} products",
                            metadata={
                                'num_products_retrieved': len(enhanced_products),
                                'retrieval_time_seconds': retrieval_time,
                                'similarity_scores': [doc.similarity_score for doc in retrieved_docs],
                                'avg_similarity': sum(doc.similarity_score for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0,
                                'in_stock_count': sum(1 for p in enhanced_products if p.get('stock_count', 0) > 0),
                                'avg_price': sum(p.get('price', 0) for p in enhanced_products) / len(enhanced_products) if enhanced_products else 0
                            }
                        )

                    # Product Context Generation with span
                    with trace.span(
                        name="product_context_generation",
                        input=f"Query + {len(enhanced_products)} products",
                        metadata={
                            'stage': 'product_prompt_construction',
                            'context_length': sum(len(str(p)) for p in enhanced_products),
                            'num_context_products': len(enhanced_products)
                        }
                    ) as context_span:

                        contextual_prompt = await self._generate_product_contextual_prompt(query, enhanced_products)

                        context_span.update(
                            output=f"Generated product prompt of {len(contextual_prompt)} characters",
                            metadata={
                                'prompt_length': len(contextual_prompt),
                                'context_products_used': len(enhanced_products)
                            }
                        )

                    # LLM Generation with span
                    with trace.span(
                        name="product_response_generation",
                        input=contextual_prompt[:500] + "..." if len(
                            contextual_prompt) > 500 else contextual_prompt,
                        metadata={
                            'stage': 'product_response_generation',
                            'model_name': self.model_name,
                            'agent_type': 'product_rag',
                            'response_intent': self._categorize_query(query),
                            'products_found': len(enhanced_products),
                            'confidence_boost': len(enhanced_products) > 0
                        }
                    ) as llm_span:

                        llm_start = time.time()
                        llm_response = await self._call_llm(contextual_prompt, trace)
                        llm_time = time.time() - llm_start

                        llm_span.update(
                            output=llm_response[:200] + "..." if len(
                                llm_response) > 200 else llm_response,
                            metadata={
                                'llm_response_time_seconds': llm_time,
                                'response_length': len(llm_response),
                                'model_used': self.model_name
                            }
                        )

                    # Update the main product RAG pipeline span
                    rag_pipeline_span.update(
                        output=llm_response[:300] +
                        "..." if len(llm_response) > 300 else llm_response,
                        metadata={
                            'total_pipeline_time_seconds': time.time() - start_time,
                            'retrieval_time_seconds': retrieval_time,
                            'llm_time_seconds': llm_time,
                            'products_used': len(enhanced_products),
                            'query_category': self._categorize_query(query)
                        }
                    )
            else:
                # Fallback processing without tracing
                retrieval_start = time.time()
                retrieved_docs = await self._retriever.retrieve(query)
                enhanced_products = await self._enhance_products_with_db_data(retrieved_docs)
                retrieval_time = time.time() - retrieval_start

                contextual_prompt = await self._generate_product_contextual_prompt(query, enhanced_products)

                llm_start = time.time()
                llm_response = await self._call_llm(contextual_prompt, trace)
                llm_time = time.time() - llm_start

            # Calculate confidence based on retrieval quality and response
            confidence = self._calculate_product_confidence(
                enhanced_products, llm_response, query)

            # Create response with enhanced product information
            response = AgentResponse(
                answer=llm_response,
                confidence=confidence,
                source_documents=[{
                    'content': self._format_product_for_context(product),
                    'source': f"Product SKU: {product.get('sku', 'Unknown')}",
                    'similarity_score': doc.similarity_score if doc else 0.0,
                    'metadata': {
                        'sku': product.get('sku'),
                        'name': product.get('name'),
                        'price': product.get('price'),
                        'stock_count': product.get('stock_count'),
                        'category': product.get('category'),
                        'is_available': product.get('stock_count', 0) > 0
                    }
                } for doc, product in zip(retrieved_docs, enhanced_products)],
                metadata={
                    'agent_type': 'product_rag',
                    'agent_name': self.name,
                    'retrieval_time': retrieval_time,
                    'llm_time': llm_time,
                    'total_time': time.time() - start_time,
                    'num_products_retrieved': len(enhanced_products),
                    'retrieval_success': len(enhanced_products) > 0,
                    'query_category': self._categorize_query(query),
                    'in_stock_products': sum(1 for p in enhanced_products if p.get('stock_count', 0) > 0)
                }
            )

            # Log product agent execution with Langfuse
            if trace:
                langfuse_client.log_agent_execution(
                    trace=trace,
                    agent_name=self.name,
                    agent_type="product_rag",
                    input_data=query,
                    output_data=llm_response,
                    execution_time=time.time() - start_time,
                    metadata=response.metadata
                )

            return response

        except Exception as e:
            error_msg = f"Error processing product query in {self.name}: {e}"
            logger.error(error_msg)

            # Log error with Langfuse
            if trace:
                langfuse_client.log_error(
                    trace=trace,
                    error_message=error_msg,
                    error_type="product_agent_processing_error",
                    context={'agent_name': self.name, 'query': query}
                )

            # Return error response
            return AgentResponse(
                answer="I apologize, but I encountered an error while searching for products. Please try again or contact customer support for assistance.",
                confidence=0.0,
                source_documents=[],
                metadata={
                    'error': str(e),
                    'agent_name': self.name,
                    'agent_type': 'product_rag'
                }
            )

    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the product agent.

        Returns:
            System prompt string for product assistance
        """
        return """You are an expert e-commerce sales assistant for our delivery platform.
        Your primary goal is to help customers find the perfect products and guide them toward making a purchase.

        Your sales responsibilities:
        - Showcase our available products with enthusiasm and expertise
        - Provide compelling product information that highlights benefits and value
        - Make personalized recommendations that address customer needs
        - Demonstrate product features and explain how they solve customer problems
        - Create urgency and reasons to buy now
        - Upsell and cross-sell relevant accessories
        - Guide customers through the purchase decision process

        Sales approach:
        - Always lead with our best-selling, in-stock products
        - Emphasize limited stock, special offers, and time-sensitive benefits
        - Connect product features to customer's specific needs and goals
        - Handle objections by finding alternatives or highlighting value
        - Guide toward adding to cart and checkout rather than just browsing
        - Build trust through expertise and genuine helpfulness

        Store policy:
        - ALWAYS prioritize our store products over external retailers
        - Never suggest shopping at Apple Store, Best Buy, Amazon, or competitors
        - Focus on products we have in stock and ready to ship
        - Create compelling reasons to buy from us (service, speed, guarantees)
        - Make it easy and attractive to purchase from our platform

        When responding:
        1. Start with enthusiasm about our amazing products and current inventory
        2. Always highlight pricing, stock levels, and compelling reasons to buy now
        3. Create urgency (limited stock, special offers, today-only deals)
        4. Suggest accessories or upgrades to increase order value
        5. Guide toward adding items to cart rather than just browsing
        6. Make it easy and attractive to complete the purchase
        7. Handle questions by focusing on benefits and solving customer problems
        8. NEVER mention external retailers - always sell from our store"""

    async def _enhance_products_with_db_data(self, retrieved_docs: List[RetrievedDocument]) -> List[Dict[str, Any]]:
        """
        Enhance retrieved documents with real-time database information.

        Args:
            retrieved_docs: List of retrieved product documents

        Returns:
            List of enhanced product dictionaries with database data
        """
        enhanced_products = []

        for doc in retrieved_docs:
            try:
                # Extract SKU from document metadata or content
                product_sku = None
                if hasattr(doc, 'metadata') and doc.metadata:
                    product_sku = doc.metadata.get('sku')

                if not product_sku:
                    # Try to extract SKU from content (fallback)
                    import re
                    sku_match = re.search(r'SKU:\s*([A-Z0-9-_]+)', doc.content)
                    if sku_match:
                        product_sku = sku_match.group(1)

                # Get real-time product data from database
                product_data = {}
                if product_sku:
                    db_product = self._product_repo.get_by_sku(product_sku)
                    if db_product:
                        product_data = {
                            'sku': db_product.sku,
                            'name': db_product.name,
                            'price': db_product.price,
                            'stock_count': db_product.stock_count,
                            'category': db_product.category.value if hasattr(db_product.category, 'value') else str(db_product.category),
                            'weight_kg': db_product.weight_kg,
                            'is_active': db_product.is_active,
                            'availability_status': db_product.get_availability_status()
                        }
                else:
                    # No SKU found, try to find product by name from the document content
                    doc_title = doc.metadata.get('title', '') if hasattr(
                        doc, 'metadata') and doc.metadata else ''
                    content_title = doc.content.split('\n')[0].replace(
                        '#', '').strip() if doc.content else ''
                    product_name = doc_title or content_title

                    # Try to find matching product in database by name
                    if product_name:
                        # Try exact match first, then partial match
                        db_products = self._product_repo.search_by_name(
                            product_name, limit=1)
                        if db_products:
                            db_product = db_products[0]
                            product_data = {
                                'sku': db_product.sku,
                                'name': db_product.name,
                                'price': db_product.price,
                                'stock_count': db_product.stock_count,
                                'category': db_product.category.value if hasattr(db_product.category, 'value') else str(db_product.category),
                                'weight_kg': db_product.weight_kg,
                                'is_active': db_product.is_active,
                                'availability_status': db_product.get_availability_status()
                            }

                # Combine retrieved document with database data
                enhanced_product = {
                    'content': doc.content,
                    'source': doc.source,
                    'similarity_score': doc.similarity_score,
                    'metadata': doc.metadata or {},
                    **product_data  # Database data overrides document data
                }

                enhanced_products.append(enhanced_product)

            except Exception as e:
                logger.warning(f"Error enhancing product data: {e}")
                # Include original document if enhancement fails
                enhanced_products.append({
                    'content': doc.content,
                    'source': doc.source,
                    'similarity_score': doc.similarity_score,
                    'metadata': doc.metadata or {}
                })

        return enhanced_products

    async def _generate_product_contextual_prompt(self, query: str, products: List[Dict[str, Any]]) -> str:
        """
        Generate a contextual prompt using retrieved product information.

        Args:
            query: The original user query
            products: List of enhanced product dictionaries

        Returns:
            Contextual prompt for LLM
        """
        system_prompt = self.get_system_prompt()

        # Build product context
        product_parts = []
        for i, product in enumerate(products, 1):
            product_info = self._format_product_for_response(product, i)
            product_parts.append(product_info)

        product_context = "\n\n".join(
            product_parts) if product_parts else "No specific products were found for this query."

        # Categorize the query for better response generation
        query_category = self._categorize_query(query)

        # Create the full prompt
        full_prompt = f"""{system_prompt}

PRODUCT INFORMATION:
{product_context}

USER QUERY: {query}
Query Category: {query_category}

RESPONSE GUIDELINES:
1. Provide accurate product information based on the data above
2. Always mention current prices and stock availability
3. If products are out of stock, suggest suitable alternatives
4. Use specific SKU numbers when referring to products
5. Be conversational and helpful while staying factual
6. Guide the customer toward making a decision
7. If appropriate, suggest adding items to cart or proceeding to checkout

RESPONSE:"""

        return full_prompt

    def _format_product_for_response(self, product: Dict[str, Any], index: int) -> str:
        """
        Format product information for response context.

        Args:
            product: Product dictionary
            index: Product index number

        Returns:
            Formatted product string
        """
        sku = product.get('sku', 'Unknown')
        name = product.get('name', 'Unknown Product')
        price = product.get('price', 0)
        stock = product.get('stock_count', 0)
        category = product.get('category', 'general')
        description = product.get('content', '').split('\n')[
            0]  # First line as description

        availability = "In Stock" if stock > 0 else "Out of Stock"
        if stock > 0 and stock < 5:
            availability = f"Only {stock} left in stock"

        return f"""Product {index}: {name}
SKU: {sku}
Price: ${price:.2f}
Category: {category}
Availability: {availability}
{f'Description: {description}' if description else ''}"""

    def _format_product_for_context(self, product: Dict[str, Any]) -> str:
        """
        Format product for context in source documents.

        Args:
            product: Product dictionary

        Returns:
            Formatted product string for context
        """
        return self._format_product_for_response(product, 1)

    def _categorize_query(self, query: str) -> str:
        """
        Categorize the type of product query.

        Args:
            query: User query string

        Returns:
            Query category string
        """
        query_lower = query.lower()

        if any(word in query_lower for word in ['search', 'find', 'look for', 'show me']):
            return 'product_search'
        elif any(word in query_lower for word in ['recommend', 'suggest', 'what do you think', 'best']):
            return 'product_recommendation'
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            return 'product_comparison'
        elif any(word in query_lower for word in ['price', 'cost', 'how much']):
            return 'pricing_inquiry'
        elif any(word in query_lower for word in ['available', 'in stock', 'have']):
            return 'availability_check'
        elif any(word in query_lower for word in ['specification', 'feature', 'detail', 'about']):
            return 'product_information'
        else:
            return 'general_inquiry'

    def _calculate_product_confidence(self, products: List[Dict[str, Any]], response: str, query: str) -> float:
        """
        Calculate confidence score based on product retrieval and response quality.

        Args:
            products: List of enhanced products
            response: Generated response
            query: Original query

        Returns:
            Confidence score between 0 and 1
        """
        if not products:
            return 0.2  # Very low confidence without products

        # Base confidence from retrieval scores
        retrieval_confidence = sum(p.get('similarity_score', 0)
                                   for p in products) / len(products)

        # Boost confidence if products are in stock
        stock_confidence = sum(1 for p in products if p.get(
            'stock_count', 0) > 0) / len(products)

        # Adjust for response quality (length and content)
        # Normalize to 300 characters
        response_length_factor = min(1.0, len(response) / 300)

        # Check if response mentions specific products or SKUs
        has_product_details = any(
            p.get('sku', '').lower() in response.lower() or
            p.get('name', '').lower() in response.lower()
            for p in products
        )
        detail_factor = 1.2 if has_product_details else 1.0

        # Combine all factors
        confidence = (retrieval_confidence * 0.4) + \
            (stock_confidence * 0.3) + (response_length_factor * 0.2)
        confidence *= detail_factor

        return min(1.0, max(0.0, confidence))

    async def _call_llm(self, prompt: str, trace=None) -> str:
        """
        Call the LLM with the product prompt.

        Args:
            prompt: The prompt to send to the LLM
            trace: Optional Langfuse trace for observability

        Returns:
            LLM response string
        """
        if not self._llm_client:
            raise RuntimeError("LLM client not initialized")

        start_time = time.time()
        try:
            response = self._llm_client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com/estebmaister/henry_bot_M4",
                    "X-Title": "henry_bot_M4-ProductRAGAgent"
                },
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,  # Longer responses for product information
                temperature=0.4,  # Slightly higher for conversational tone
            )

            response_time = time.time() - start_time

            # Validate response structure
            if not response or not hasattr(response, 'choices') or len(response.choices) == 0:
                logger.error(f"Invalid LLM response: {response}")
                return ""

            if not hasattr(response.choices[0], 'message') or not hasattr(response.choices[0].message, 'content'):
                logger.error(
                    f"Invalid response structure: {response.choices[0]}")
                return ""

            content = response.choices[0].message.content or ""
            content = content.strip()

            if not content:
                logger.warning(
                    f"Empty response content for prompt: {prompt[:100]}...")
                return ""

            # Log LLM call with enhanced debugging
            if trace:
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }

                langfuse_client.log_llm_call(
                    trace=trace,
                    model_name=self.model_name,
                    prompt=prompt,
                    response=content,
                    token_usage=token_usage,
                    response_time=response_time,
                    temperature=0.4,
                    max_tokens=1200,
                    metadata={
                        'agent_name': self.name,
                        'agent_type': 'product_rag',
                        'model_provider': 'openrouter',
                        'response_finish_reason': response.choices[0].finish_reason if response.choices else None,
                        'response_id': response.id
                    }
                )

            logger.info(f"LLM response received: {content[:200]}...")
            return content

        except Exception as e:
            error_time = time.time() - start_time
            error_msg = f"LLM API call failed: {e}"
            logger.error(f"{error_msg}")
            logger.error(f"Prompt was: {prompt[:200]}...")

            # Log LLM error
            if trace:
                langfuse_client.log_error(
                    trace=trace,
                    error_message=error_msg,
                    error_type="llm_call_error",
                    context={
                        'model_name': self.model_name,
                        'agent_name': self.name,
                        'agent_type': 'product_rag',
                        'error_time_seconds': error_time,
                        'prompt_preview': prompt[:200] if prompt else None
                    }
                )

            raise
