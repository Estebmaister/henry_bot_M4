"""
Function routing and tool selection engine.

AI Assistant Notes:
- Intelligently selects appropriate tools based on conversation context
- Analyzes user intent to determine when to use function calling
- Handles tool selection for both product and order operations
- Provides fallback mechanisms for ambiguous queries
- Optimizes tool usage patterns for better user experience
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class FunctionRouter:
    """
    Routes user queries to appropriate function calls based on intent analysis.
    Analyzes conversation context to determine optimal tool selection.
    """

    def __init__(self):
        """Initialize function router with intent patterns."""
        self.intent_patterns = self._initialize_intent_patterns()
        self.tool_mappings = self._initialize_tool_mappings()

    def _initialize_intent_patterns(self) -> Dict[str, List[str]]:
        """
        Initialize regex patterns for intent recognition.

        Returns:
            Dictionary mapping intents to regex patterns
        """
        return {
            'product_search': [
                r'\b(search|find|look for|show me|browse|looking for)\b.*\b(product|item|thing)\b',
                r'\b(do you have|got any|what products|available)\b',
                r'\b(recommend|suggest|what do you think)\b.*\b(product|item)\b',
                r'\b(compare|difference|versus|vs)\b.*\b(product|item)\b'
            ],
            'order_creation': [
                r'\b(buy|purchase|order|want to buy|interested in|take)\b',
                r'\b(add to cart|checkout|proceed to payment)\b',
                r'\b(how much for|price for|cost of)\b.*\b(and|plus|with)\b',
                r'\b(ill take|i want|i need|i would like)\b',
                r'\b(ready to order|finalize order|confirm order)\b'
            ],
            'order_status': [
                r'\b(order status|track order|where is my|status of)\b',
                r'\b(check|verify|look up)\b.*\b(order)\b',
                r'\b(ORD-\w+)\b',  # Order number format
                r'\b(delivery|shipping|when will i get)\b'
            ],
            'order_update': [
                r'\b(change|update|modify)\b.*\b(order|address)\b',
                r'\b(different address|new address|change delivery)\b',
                r'\b(add item|remove item|modify order)\b'
            ],
            'order_cancellation': [
                r'\b(cancel|stop|undo)\b.*\b(order)\b',
                r'\b(no longer want|dont need|changed my mind)\b',
                r'\b(refund|cancel my purchase)\b'
            ],
            'pricing_inquiry': [
                r'\b(how much|what price|cost|price of)\b',
                r'\b(cheapest|most expensive|affordable)\b',
                r'\b(in stock|available)\b.*\b(how many)\b'
            ]
        }

    def _initialize_tool_mappings(self) -> Dict[str, str]:
        """
        Initialize mapping from intents to tool names.

        Returns:
            Dictionary mapping intents to primary tool names
        """
        return {
            'product_search': 'search_products',
            'order_creation': 'create_order',
            'order_status': 'check_order_status',
            'order_update': 'update_order',
            'order_cancellation': 'cancel_order',
            'pricing_inquiry': 'search_products'  # Use product search for pricing
        }

    def analyze_intent(self, query: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Analyze user query to determine intent and confidence.

        Args:
            query: User query string
            conversation_history: Previous conversation context

        Returns:
            Dictionary with intent analysis results
        """
        query_lower = query.lower()

        # Score each intent
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower, re.IGNORECASE))
                score += matches

            # Boost score based on conversation context
            if conversation_history:
                context_boost = self._calculate_context_boost(intent, query_lower, conversation_history)
                score += context_boost

            intent_scores[intent] = score

        # Determine primary intent
        if not any(intent_scores.values()):
            return {
                'intent': 'general_inquiry',
                'confidence': 0.0,
                'all_scores': intent_scores
            }

        primary_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[primary_intent]
        total_score = sum(intent_scores.values())

        # Calculate confidence
        confidence = max_score / total_score if total_score > 0 else 0.0

        return {
            'intent': primary_intent,
            'confidence': min(confidence * 2, 1.0),  # Normalize and boost
            'all_scores': intent_scores,
            'query_contains_order_number': bool(re.search(r'\bORD-\w+\b', query.upper()))
        }

    def _calculate_context_boost(self, intent: str, query: str, conversation_history: List[Dict[str, str]]) -> float:
        """
        Calculate context boost based on conversation history.

        Args:
            intent: Intent being evaluated
            query: Current query
            conversation_history: Previous messages

        Returns:
            Context boost score
        """
        if not conversation_history:
            return 0.0

        boost = 0.0
        recent_messages = conversation_history[-3:]  # Look at last 3 messages

        # Check for context patterns in recent conversation
        context_patterns = {
            'order_creation': [
                (r'\b(product|item|price|cost)\b', 0.3),
                (r'\b(interested|want|like)\b', 0.5),
                (r'\b(add|include|also)\b', 0.4)
            ],
            'product_search': [
                (r'\b(looking for|search|find)\b', 0.4),
                (r'\b(category|type|kind)\b', 0.3)
            ],
            'order_status': [
                (r'\b(order|purchase|delivery)\b', 0.5),
                (r'\b(track|status|check)\b', 0.4)
            ]
        }

        if intent in context_patterns:
            for msg in recent_messages:
                msg_text = msg.get('content', '').lower()
                for pattern, weight in context_patterns[intent]:
                    if re.search(pattern, msg_text):
                        boost += weight

        return min(boost, 1.0)

    def select_tools(self, intent_analysis: Dict[str, Any], query: str) -> List[str]:
        """
        Select appropriate tools based on intent analysis.

        Args:
            intent_analysis: Results from intent analysis
            query: Original user query

        Returns:
            List of recommended tools
        """
        primary_intent = intent_analysis['intent']
        confidence = intent_analysis['confidence']
        tools = []

        # High confidence - use primary tool
        if confidence > 0.6 and primary_intent in self.tool_mappings:
            tools.append(self.tool_mappings[primary_intent])

        # Medium confidence - consider multiple tools
        elif confidence > 0.3:
            if primary_intent in self.tool_mappings:
                tools.append(self.tool_mappings[primary_intent])

            # Add secondary tools based on query content
            if 'price' in query.lower() or 'cost' in query.lower():
                if 'search_products' not in tools:
                    tools.append('search_products')

        # Low confidence - general inquiry with potential product search
        else:
            # Check if query mentions product-related terms
            product_terms = ['product', 'item', 'buy', 'price', 'available']
            if any(term in query.lower() for term in product_terms):
                tools.append('search_products')

        return tools

    def should_use_function_calling(self, intent_analysis: Dict[str, Any]) -> bool:
        """
        Determine if function calling should be used for this query.

        Args:
            intent_analysis: Results from intent analysis

        Returns:
            True if function calling is recommended
        """
        intent = intent_analysis['intent']
        confidence = intent_analysis['confidence']

        # Always use function calling for these intents if confidence is reasonable
        function_calling_intents = [
            'order_creation',
            'order_status',
            'order_update',
            'order_cancellation'
        ]

        if intent in function_calling_intents and confidence > 0.3:
            return True

        # Use function calling for product searches if confident
        if intent == 'product_search' and confidence > 0.4:
            return True

        # Check for explicit indicators
        query_contains_order_number = intent_analysis.get('query_contains_order_number', False)
        if query_contains_order_number:
            return True

        return False

    def extract_contextual_parameters(
        self, query: str, conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Extract parameters from conversation context.

        Args:
            query: Current user query
            conversation_history: Previous conversation messages

        Returns:
            Dictionary with extracted parameters
        """
        context_params = {}

        if not conversation_history:
            return context_params

        recent_messages = conversation_history[-5:]  # Look at last 5 messages

        # Extract product references
        product_mentions = []
        for msg in recent_messages:
            content = msg.get('content', '')
            # Look for SKU patterns
            skus = re.findall(r'\b[A-Z]{3,}-[A-Z0-9-]+\b', content.upper())
            product_mentions.extend(skus)

            # Look for product names (simplified)
            products = re.findall(r'\b(iPhone|MacBook|PlayStation|Samsung|Sony|Apple)\s*\w*\b', content)
            product_mentions.extend(products)

        if product_mentions:
            context_params['mentioned_products'] = list(set(product_mentions))

        # Extract customer information
        customer_info = {}
        for msg in recent_messages:
            content = msg.get('content', '').lower()

            # Email extraction
            email_match = re.search(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', content)
            if email_match and 'email' not in customer_info:
                customer_info['email'] = email_match.group()

            # Phone extraction (simplified)
            phone_match = re.search(r'\b(\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\d{10})\b', content)
            if phone_match and 'phone' not in customer_info:
                customer_info['phone'] = phone_match.group()

            # Name extraction (very simplified)
            name_patterns = [
                r'\b(my name is|i am|i\'m)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
                r'\b(call me)\s+([A-Z][a-z]+)\b'
            ]
            for pattern in name_patterns:
                name_match = re.search(pattern, content, re.IGNORECASE)
                if name_match and 'name' not in customer_info:
                    customer_info['name'] = name_match.group(2)
                    break

        if customer_info:
            context_params['customer_info'] = customer_info

        # Extract order numbers
        order_numbers = []
        for msg in recent_messages:
            content = msg.get('content', '')
            orders = re.findall(r'\bORD-\w+\b', content.upper())
            order_numbers.extend(orders)

        if order_numbers:
            context_params['mentioned_orders'] = list(set(order_numbers))

        return context_params

    def get_tool_suggestions(self, query: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Get comprehensive tool suggestions for a query.

        Args:
            query: User query
            conversation_history: Conversation context

        Returns:
            Dictionary with tool suggestions and reasoning
        """
        # Analyze intent
        intent_analysis = self.analyze_intent(query, conversation_history)

        # Select tools
        recommended_tools = self.select_tools(intent_analysis, query)

        # Determine if function calling should be used
        use_function_calling = self.should_use_function_calling(intent_analysis)

        # Extract contextual parameters
        context_params = self.extract_contextual_parameters(query, conversation_history)

        return {
            'use_function_calling': use_function_calling,
            'recommended_tools': recommended_tools,
            'primary_intent': intent_analysis['intent'],
            'confidence': intent_analysis['confidence'],
            'reasoning': f"Intent: {intent_analysis['intent']} (confidence: {intent_analysis['confidence']:.2f})",
            'context_parameters': context_params,
            'intent_scores': intent_analysis['all_scores']
        }