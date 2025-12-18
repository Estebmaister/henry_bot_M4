"""
Order information extraction from conversation context.

AI Assistant Notes:
- Extracts order details from multi-turn conversations
- Identifies products, quantities, customer information
- Detects purchase intent and order completion signals
- Validates extracted information against database
- Provides confidence scores for extracted data
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class OrderInformationExtractor:
    """
    Extracts and validates order information from conversation context.
    Helps build complete orders from conversational interactions.
    """

    def __init__(self):
        """Initialize the extractor with patterns and rules."""
        self.extraction_patterns = self._initialize_extraction_patterns()
        self.validation_rules = self._initialize_validation_rules()

    def _initialize_extraction_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Initialize regex patterns for information extraction."""
        return {
            'product_sku': [
                re.compile(r'\b([A-Z]{2,4}-[A-Z0-9-]{3,})\b'),
                re.compile(r'\b(SKU:?\s*([A-Z0-9-]+))\b', re.IGNORECASE),
            ],
            'quantity': [
                re.compile(r'\b(\d+)\s+(?:of|pieces?|items?|units?)\b', re.IGNORECASE),
                re.compile(r'\b(?:want|need|take|get|order)\s+(\d+)\b', re.IGNORECASE),
                re.compile(r'\b(\d+)\s+(?:x|times?)\b', re.IGNORECASE),
            ],
            'customer_name': [
                re.compile(r'\b(?:my name is|i am|i\'m|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', re.IGNORECASE),
                re.compile(r'\b(?:this is|this is for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', re.IGNORECASE),
            ],
            'customer_email': [
                re.compile(r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'),
            ],
            'customer_phone': [
                re.compile(r'\b(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})\b'),
                re.compile(r'\b(\+1\s?\d{3}[-.\s]?\d{3}[-.\s]?\d{4})\b'),
                re.compile(r'\b(\d{10})\b'),
            ],
            'delivery_address': [
                re.compile(r'\b(?:deliver to|address is|ship to)\s+(.+?)(?:\.|$)', re.IGNORECASE),
                re.compile(r'\b(\d+\s+[^,]+\s+(?:street|st|avenue|ave|road|rd|boulevard|blvd)[^.]*)', re.IGNORECASE),
            ],
            'purchase_intent': [
                re.compile(r'\b(?:buy|purchase|order|want to buy|interested in|take|get|need)\b', re.IGNORECASE),
                re.compile(r'\b(?:add to cart|checkout|proceed|ready to order)\b', re.IGNORECASE),
                re.compile(r'\b(?:how much|price|cost).*(?:for|of)\b', re.IGNORECASE),
            ],
            'product_names': [
                re.compile(r'\b(iPhone|iPad|MacBook|PlayStation|PS\d|Samsung|Sony|Apple|Canon)\s*\w*\b', re.IGNORECASE),
                re.compile(r'\b(laptop|phone|tablet|headphones|speaker|camera|tv|television)\b', re.IGNORECASE),
            ],
        }

    def _initialize_validation_rules(self) -> Dict[str, callable]:
        """Initialize validation rules for extracted data."""
        return {
            'sku': lambda x: len(x) >= 3 and x.replace('-', '').isalnum(),
            'quantity': lambda x: isinstance(x, int) and 1 <= x <= 100,
            'email': lambda x: '@' in x and '.' in x.split('@')[-1],
            'phone': lambda x: len(re.sub(r'\D', '', str(x))) >= 10,
            'price': lambda x: isinstance(x, (int, float)) and x > 0,
        }

    def extract_from_conversation(
        self,
        conversation_history: List[Dict[str, str]],
        current_query: str = None
    ) -> Dict[str, Any]:
        """
        Extract order information from conversation context.

        Args:
            conversation_history: List of conversation messages
            current_query: Current user query (optional)

        Returns:
            Dictionary with extracted information and confidence scores
        """
        extracted_data = {
            'products': [],
            'customer_info': {},
            'order_details': {},
            'intent_signals': {},
            'confidence_scores': {},
            'extraction_summary': {}
        }

        try:
            # Combine conversation text for analysis
            full_conversation = self._combine_conversation_text(conversation_history, current_query)

            # Extract products
            products = self._extract_products(full_conversation, conversation_history)
            extracted_data['products'] = products

            # Extract customer information
            customer_info = self._extract_customer_info(full_conversation)
            extracted_data['customer_info'] = customer_info

            # Extract order details
            order_details = self._extract_order_details(full_conversation)
            extracted_data['order_details'] = order_details

            # Analyze purchase intent
            intent_signals = self._analyze_purchase_intent(full_conversation, conversation_history)
            extracted_data['intent_signals'] = intent_signals

            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(extracted_data)
            extracted_data['confidence_scores'] = confidence_scores

            # Generate extraction summary
            summary = self._generate_extraction_summary(extracted_data)
            extracted_data['extraction_summary'] = summary

            return extracted_data

        except Exception as e:
            logger.error(f"Error extracting order information: {e}")
            extracted_data['error'] = str(e)
            return extracted_data

    def _combine_conversation_text(
        self,
        conversation_history: List[Dict[str, str]],
        current_query: str = None
    ) -> str:
        """Combine conversation text for analysis."""
        texts = []

        # Include recent conversation history
        recent_messages = conversation_history[-8:]  # Last 8 messages
        for msg in recent_messages:
            content = msg.get('content', '')
            role = msg.get('role', '')
            # Add weight to user messages
            if role == 'user':
                texts.append(content)

        # Add current query
        if current_query:
            texts.append(current_query)

        return ' '.join(texts)

    def _extract_products(self, text: str, conversation_history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Extract product information from text."""
        products = []

        # Extract SKUs with context
        sku_matches = []
        for pattern in self.extraction_patterns['product_sku']:
            for match in pattern.finditer(text):
                sku = match.group(1) if match.lastindex == 1 else match.group(2)
                sku_matches.append({
                    'sku': sku,
                    'position': match.start(),
                    'context': text[max(0, match.start()-50):match.end()+50]
                })

        # Extract product names
        name_matches = []
        for pattern in self.extraction_patterns['product_names']:
            for match in pattern.finditer(text):
                name_matches.append({
                    'name': match.group(),
                    'position': match.start()
                })

        # Extract quantities
        quantity_matches = []
        for pattern in self.extraction_patterns['quantity']:
            for match in pattern.finditer(text):
                quantity_matches.append({
                    'quantity': int(match.group(1)),
                    'position': match.start()
                })

        # Combine extracted information
        processed_skus = set()
        for sku_match in sku_matches:
            sku = sku_match['sku'].upper()

            if sku in processed_skus:
                continue
            processed_skus.add(sku)

            product_info = {
                'sku': sku,
                'quantity': 1,  # Default quantity
                'confidence': 0.8,
                'source': 'explicit_sku',
                'context': sku_match['context']
            }

            # Try to find associated quantity
            nearest_quantity = self._find_nearest_quantity(sku_match['position'], quantity_matches)
            if nearest_quantity:
                product_info['quantity'] = nearest_quantity['quantity']
                product_info['confidence'] += 0.1

            products.append(product_info)

        # Add products mentioned by name (without SKU)
        for name_match in name_matches:
            product_name = name_match['name']

            # Check if this product is already captured by SKU
            already_captured = False
            for product in products:
                if product_name.lower() in product.get('context', '').lower():
                    already_captured = True
                    break

            if not already_captured:
                nearest_quantity = self._find_nearest_quantity(name_match['position'], quantity_matches)

                products.append({
                    'sku': None,  # Will need to be resolved
                    'name': product_name,
                    'quantity': nearest_quantity['quantity'] if nearest_quantity else 1,
                    'confidence': 0.5 if nearest_quantity else 0.4,
                    'source': 'product_name',
                    'position': name_match['position']
                })

        return products

    def _extract_customer_info(self, text: str) -> Dict[str, Any]:
        """Extract customer information from text."""
        customer_info = {}

        # Extract name
        for pattern in self.extraction_patterns['customer_name']:
            match = pattern.search(text)
            if match:
                customer_info['name'] = match.group(1).strip()
                customer_info['name_confidence'] = 0.8
                break

        # Extract email
        for pattern in self.extraction_patterns['customer_email']:
            match = pattern.search(text)
            if match:
                email = match.group(1).lower()
                if self._validate_field('email', email):
                    customer_info['email'] = email
                    customer_info['email_confidence'] = 0.9
                    break

        # Extract phone
        for pattern in self.extraction_patterns['customer_phone']:
            match = pattern.search(text)
            if match:
                phone = re.sub(r'[^\d]', '', match.group(1))
                if self._validate_field('phone', phone):
                    customer_info['phone'] = phone
                    customer_info['phone_confidence'] = 0.8
                    break

        return customer_info

    def _extract_order_details(self, text: str) -> Dict[str, Any]:
        """Extract order-specific details from text."""
        order_details = {}

        # Extract delivery address
        for pattern in self.extraction_patterns['delivery_address']:
            match = pattern.search(text)
            if match:
                address = match.group(1).strip()
                if len(address) > 10:  # Minimum address length
                    order_details['delivery_address'] = address
                    order_details['address_confidence'] = 0.7
                    break

        return order_details

    def _analyze_purchase_intent(
        self,
        text: str,
        conversation_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Analyze purchase intent signals."""
        intent_signals = {
            'purchase_detected': False,
            'confidence': 0.0,
            'signals': []
        }

        # Check for explicit purchase intent
        for pattern in self.extraction_patterns['purchase_intent']:
            matches = pattern.findall(text)
            if matches:
                intent_signals['purchase_detected'] = True
                intent_signals['signals'].extend(matches)

        # Analyze conversation flow
        recent_messages = conversation_history[-4:]
        product_mentions = 0
        question_count = 0

        for msg in recent_messages:
            content = msg.get('content', '').lower()
            role = msg.get('role', '')

            if role == 'user':
                # Count product-related mentions
                product_keywords = ['buy', 'price', 'cost', 'order', 'purchase', 'interested']
                product_mentions += sum(1 for keyword in product_keywords if keyword in content)

                # Count questions (indicates consideration phase)
                question_count += content.count('?')

        # Calculate intent confidence
        if intent_signals['purchase_detected']:
            intent_signals['confidence'] = min(0.8, 0.5 + (product_mentions * 0.1))
        elif product_mentions >= 2:
            intent_signals['confidence'] = 0.4
            intent_signals['signals'].append('multiple_product_mentions')
        elif question_count >= 2:
            intent_signals['confidence'] = 0.3
            intent_signals['signals'].append('consideration_questions')

        return intent_signals

    def _find_nearest_quantity(
        self,
        position: int,
        quantity_matches: List[Dict[str, Any]],
        max_distance: int = 100
    ) -> Optional[Dict[str, Any]]:
        """Find the nearest quantity to a given position."""
        nearest = None
        min_distance = max_distance

        for qty_match in quantity_matches:
            distance = abs(position - qty_match['position'])
            if distance < min_distance:
                min_distance = distance
                nearest = qty_match

        return nearest

    def _validate_field(self, field_type: str, value: str) -> bool:
        """Validate extracted field using rules."""
        if field_type in self.validation_rules:
            return self.validation_rules[field_type](value)
        return True

    def _calculate_confidence_scores(self, extracted_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall confidence scores for extracted information."""
        scores = {}

        # Products confidence
        products = extracted_data.get('products', [])
        if products:
            product_confidences = [p.get('confidence', 0) for p in products]
            scores['products'] = sum(product_confidences) / len(product_confidences)
        else:
            scores['products'] = 0.0

        # Customer info confidence
        customer_info = extracted_data.get('customer_info', {})
        info_confidences = [
            customer_info.get('name_confidence', 0),
            customer_info.get('email_confidence', 0),
            customer_info.get('phone_confidence', 0)
        ]
        scores['customer_info'] = sum(info_confidences) / len(info_confidences) if info_confidences else 0.0

        # Order details confidence
        order_details = extracted_data.get('order_details', {})
        scores['order_details'] = order_details.get('address_confidence', 0.0)

        # Intent confidence
        intent_signals = extracted_data.get('intent_signals', {})
        scores['purchase_intent'] = intent_signals.get('confidence', 0.0)

        # Overall confidence
        scores['overall'] = (
            scores['products'] * 0.4 +
            scores['customer_info'] * 0.3 +
            scores['order_details'] * 0.2 +
            scores['purchase_intent'] * 0.1
        )

        return scores

    def _generate_extraction_summary(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of extraction results."""
        products = extracted_data.get('products', [])
        customer_info = extracted_data.get('customer_info', {})
        intent_signals = extracted_data.get('intent_signals', {})

        summary = {
            'ready_for_order': False,
            'missing_information': [],
            'actionable_items': []
        }

        # Check if we have enough information for order creation
        has_products = len(products) > 0
        has_customer_name = 'name' in customer_info
        has_contact_info = 'email' in customer_info or 'phone' in customer_info
        has_purchase_intent = intent_signals.get('purchase_detected', False)

        # Determine missing information
        if not has_products:
            summary['missing_information'].append('products')
        if not has_customer_name:
            summary['missing_information'].append('customer_name')
        if not has_contact_info:
            summary['missing_information'].append('contact_info')

        # Determine actionable items
        if has_products and not has_customer_name:
            summary['actionable_items'].append('Ask for customer name')
        if has_products and not has_contact_info:
            summary['actionable_items'].append('Ask for email or phone')
        if has_purchase_intent and has_products:
            summary['actionable_items'].append('Proceed with order creation')

        # Determine if ready for order
        if has_products and has_customer_name and has_contact_info:
            summary['ready_for_order'] = True

        return summary

    def suggest_next_actions(self, extracted_data: Dict[str, Any]) -> List[str]:
        """
        Suggest next actions based on extracted information.

        Args:
            extracted_data: Extracted order information

        Returns:
            List of suggested actions
        """
        suggestions = []

        products = extracted_data.get('products', [])
        customer_info = extracted_data.get('customer_info', {})
        intent_signals = extracted_data.get('intent_signals', {})
        summary = extracted_data.get('extraction_summary', {})

        # High purchase intent - suggest order completion
        if intent_signals.get('confidence', 0) > 0.6:
            if summary.get('ready_for_order'):
                suggestions.append("Ready to create order - ask for confirmation")
            else:
                missing = summary.get('missing_information', [])
                for item in missing:
                    if item == 'customer_name':
                        suggestions.append("Ask for customer's full name")
                    elif item == 'contact_info':
                        suggestions.append("Ask for email address or phone number")
                    elif item == 'products':
                        suggestions.append("Help customer find products")

        # Medium purchase intent - suggest product confirmation
        elif intent_signals.get('confidence', 0) > 0.3:
            if products:
                suggestions.append("Confirm product selection and quantities")
                if not customer_info.get('name'):
                    suggestions.append("Ask for customer name")

        # Low purchase intent - suggest exploration
        else:
            if products:
                suggestions.append("Provide more product details and recommendations")
            else:
                suggestions.append("Help customer explore product catalog")

        return suggestions