"""
Function execution engine for order processing.

AI Assistant Notes:
- Executes validated function calls from OpenAI Function Calling
- Integrates with database repositories for order operations
- Handles product validation and inventory management
- Provides comprehensive error handling and validation
- Returns structured results for LLM processing
"""

from src.utils import observe
from ..database import OrderRepository, ProductRepository
from ..database.models import OrderModel, OrderItemModel, ProductModel, OrderStatus
from ..utils import langfuse_client
from typing import Dict, Any, List, Optional
import logging
import uuid
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FunctionExecutor:
    """
    Executes function calls for order processing operations.
    Handles order creation, tracking, updates, and product searches.
    """

    def __init__(self, order_repo: OrderRepository, product_repo: ProductRepository):
        """
        Initialize function executor.

        Args:
            order_repo: Order repository for database operations
            product_repo: Product repository for product operations
        """
        self.order_repo = order_repo
        self.product_repo = product_repo

    @observe(name="function_execution", as_type="tool")
    async def execute_function(self, function_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a function by name with given arguments.

        Args:
            function_name: Name of the function to execute
            args: Function arguments

        Returns:
            Dictionary with execution results
        """
        # Use dummy trace context for backward compatibility
        trace = langfuse_client.create_trace(
            name="function_execution_trace",
            input={'function_name': function_name, 'arguments': args}
        )

        try:
            # Create function execution span
            with trace.span(
                name=f"execute_{function_name}",
                input={'function_name': function_name, 'arguments': args},
                metadata={
                    'tool_name': function_name,
                    'tool_category': self._get_tool_category(function_name),
                    'argument_count': len(args),
                    'execution_stage': 'function_call'
                }
            ) as execution_span:

                if function_name == "search_products":
                    result = await self._search_products(args)
                elif function_name == "create_order":
                    result = await self._create_order(args)
                elif function_name == "check_order_status":
                    result = await self._check_order_status(args)
                elif function_name == "update_order":
                    result = await self._update_order(args)
                elif function_name == "cancel_order":
                    result = await self._cancel_order(args)
                else:
                    raise ValueError(f"Unknown function: {function_name}")

                # Update execution span with results
                execution_span.update(
                    output={
                        'success': result.get('success', False),
                        'result_summary': self._get_result_summary(function_name, result)
                    },
                    metadata={
                        'execution_success': result.get('success', False),
                        'execution_time_seconds': execution_span.end_time - execution_span.start_time if hasattr(execution_span, 'end_time') else None,
                        'result_size': len(str(result))
                    }
                )

                return result

        except Exception as e:
            logger.error(f"Error executing function {function_name}: {e}")

            # Log error with Langfuse
            if trace:
                langfuse_client.log_error(
                    trace=trace,
                    error_message=f"Function execution failed: {e}",
                    error_type="function_execution_error",
                    context={
                        'function_name': function_name,
                        'arguments': args
                    }
                )

            return {
                "success": False,
                "error": str(e),
                "function_name": function_name
            }

    def _get_tool_category(self, function_name: str) -> str:
        """Get the category of a tool for better observability."""
        categories = {
            "search_products": "product_discovery",
            "create_order": "order_management",
            "check_order_status": "order_tracking",
            "update_order": "order_management",
            "cancel_order": "order_management"
        }
        return categories.get(function_name, "unknown")

    def _get_result_summary(self, function_name: str, result: Dict[str, Any]) -> str:
        """Get a summary of the function result for observability."""
        if not result.get('success', False):
            return f"Failed: {result.get('error', 'Unknown error')[:100]}"

        if function_name == "search_products":
            return f"Found {result.get('total_found', 0)} products"
        elif function_name == "create_order":
            return f"Order {result.get('order_number', 'unknown')} created with {result.get('item_count', 0)} items"
        elif function_name == "check_order_status":
            return f"Order {result.get('order_number', 'unknown')} status: {result.get('order_status', 'unknown')}"
        elif function_name == "update_order":
            return f"Order {result.get('order_number', 'unknown')} updated: {', '.join(result.get('updated_fields', []))}"
        elif function_name == "cancel_order":
            return f"Order {result.get('order_number', 'unknown')} cancelled"
        else:
            return "Function executed successfully"

    @observe(name="product_search_execution", as_type="tool")
    async def _search_products(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for products in the catalog.

        Args:
            args: Search parameters including query, category, price range, etc.

        Returns:
            Dictionary with search results
        """
        try:
            query = args.get('query', '')
            category = args.get('category')
            min_price = args.get('min_price')
            max_price = args.get('max_price')
            in_stock_only = args.get('in_stock_only', True)
            limit = min(args.get('limit', 10), 50)  # Cap at 50 for safety

            # Search products using repository
            products = self.product_repo.search(
                query_text=query,
                category=category,
                limit=limit
            )

            # Apply additional filters
            filtered_products = []
            for product in products:
                # Price filtering
                if min_price and product.price < min_price:
                    continue
                if max_price and product.price > max_price:
                    continue

                # Stock filtering
                if in_stock_only and not product.is_in_stock():
                    continue

                filtered_products.append({
                    "sku": product.sku,
                    "name": product.name,
                    "description": product.description,
                    "price": product.price,
                    "category": product.category.value if hasattr(product.category, 'value') else str(product.category),
                    "stock_count": product.stock_count,
                    "weight_kg": product.weight_kg,
                    "availability": product.get_availability_status()
                })

            return {
                "success": True,
                "products": filtered_products,
                "total_found": len(filtered_products),
                "search_params": {
                    "query": query,
                    "category": category,
                    "price_range": f"${min_price or '0'}-${max_price or 'unlimited'}",
                    "in_stock_only": in_stock_only
                }
            }

        except Exception as e:
            logger.error(f"Product search error: {e}")
            return {
                "success": False,
                "error": f"Failed to search products: {str(e)}",
                "products": []
            }

    @observe(name="order_creation_execution", as_type="tool")
    async def _create_order(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new customer order.

        Args:
            args: Order details including customer info and items

        Returns:
            Dictionary with order creation results
        """
        try:
            # Validate order items first
            items_data = args.get('items', [])
            if not items_data:
                return {
                    "success": False,
                    "error": "Order must contain at least one item"
                }

            validated_items = []
            total_amount = 0.0

            # Validate each product and calculate total
            for item_data in items_data:
                product_sku = item_data.get('product_sku')
                quantity = item_data.get('quantity', 1)

                # Get product from database
                product = self.product_repo.get_by_sku(product_sku.upper())
                if not product:
                    return {
                        "success": False,
                        "error": f"Product with SKU {product_sku} not found"
                    }

                if not product.is_in_stock(quantity):
                    return {
                        "success": False,
                        "error": f"Product {product.name} (SKU: {product_sku}) has insufficient stock. Available: {product.stock_count}"
                    }

                # Create order item
                unit_price = product.price
                item_total = unit_price * quantity
                total_amount += item_total

                validated_items.append(OrderItemModel(
                    product_sku=product.sku,
                    product_name=product.name,
                    quantity=quantity,
                    unit_price=unit_price,
                    total_price=item_total
                ))

            # Create order model
            order_data = OrderModel(
                customer_name=args.get('customer_name'),
                customer_email=args.get('customer_email'),
                customer_phone=args.get('customer_phone'),
                delivery_address=args.get('delivery_address'),
                items=validated_items,
                special_instructions=args.get('special_instructions'),
                total_amount=total_amount
            )

            # Save order to database
            order_id = self.order_repo.create(order_data)

            # Update product stock
            for item in validated_items:
                self.product_repo.update_stock(
                    self.product_repo.get_by_sku(item.product_sku).id,
                    -item.quantity  # Subtract from stock
                )

            # Get created order with order number
            created_order = self.order_repo.get_by_id(order_id)

            return {
                "success": True,
                "order_id": order_id,
                "order_number": created_order.order_number,
                "total_amount": total_amount,
                "item_count": len(validated_items),
                "estimated_delivery": (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"),
                "customer_name": order_data.customer_name,
                "items": [
                    {
                        "product_sku": item.product_sku,
                        "product_name": item.product_name,
                        "quantity": item.quantity,
                        "unit_price": item.unit_price,
                        "total_price": item.total_price
                    }
                    for item in validated_items
                ]
            }

        except Exception as e:
            logger.error(f"Order creation error: {e}")
            return {
                "success": False,
                "error": f"Failed to create order: {str(e)}"
            }

    @observe(name="order_status_check_execution", as_type="tool")
    async def _check_order_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check the status of an existing order.

        Args:
            args: Order number and customer email for verification

        Returns:
            Dictionary with order status information
        """
        try:
            order_number = args.get('order_number', '').upper()
            customer_email = args.get('customer_email', '').lower()

            # Validate order number format
            if not order_number.startswith('ORD-'):
                return {
                    "success": False,
                    "error": "Invalid order number format. Please use format: ORD-XXXXXX"
                }

            # Get order by number
            order = self.order_repo.get_by_number(order_number)
            if not order:
                return {
                    "success": False,
                    "error": f"Order {order_number} not found"
                }

            # Verify customer email (if provided)
            if customer_email and order.customer_email and order.customer_email.lower() != customer_email:
                return {
                    "success": False,
                    "error": "Email address does not match the one on file for this order"
                }

            # Calculate estimated delivery if not set
            if not order.estimated_delivery and order.order_status in [OrderStatus.CONFIRMED, OrderStatus.PREPARING]:
                estimated_delivery = (
                    datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
            else:
                estimated_delivery = order.estimated_delivery.strftime(
                    "%Y-%m-%d") if order.estimated_delivery else None

            return {
                "success": True,
                "order_id": order.id,
                "order_number": order.order_number,
                "order_status": order.order_status.value if hasattr(order.order_status, 'value') else str(order.order_status),
                "status_display": order.get_status_display(),
                "total_amount": order.total_amount,
                "item_count": order.get_item_count(),
                "customer_name": order.customer_name,
                "created_date": order.created_at.strftime("%Y-%m-%d"),
                "estimated_delivery": estimated_delivery,
                "delivery_address": order.delivery_address,
                "items": [
                    {
                        "product_sku": item.product_sku,
                        "product_name": item.product_name,
                        "quantity": item.quantity,
                        "unit_price": item.unit_price
                    }
                    for item in order.items
                ]
            }

        except Exception as e:
            logger.error(f"Order status check error: {e}")
            return {
                "success": False,
                "error": f"Failed to check order status: {str(e)}"
            }

    async def _update_order(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing order.

        Args:
            args: Order update details including new address, items, etc.

        Returns:
            Dictionary with update results
        """
        try:
            order_number = args.get('order_number', '').upper()
            customer_email = args.get('customer_email', '').lower()

            # Get and verify order
            order = self.order_repo.get_by_number(order_number)
            if not order:
                return {
                    "success": False,
                    "error": f"Order {order_number} not found"
                }

            if order.customer_email and order.customer_email.lower() != customer_email:
                return {
                    "success": False,
                    "error": "Email verification failed"
                }

            # Check if order can be updated
            if not order.can_be_cancelled():  # Same logic for updates
                return {
                    "success": False,
                    "error": "Order cannot be updated. It may already be shipped or delivered."
                }

            updated_fields = []

            # Update delivery address if provided
            new_address = args.get('delivery_address')
            if new_address and new_address != order.delivery_address:
                order.delivery_address = new_address
                updated_fields.append("delivery_address")

            # Update special instructions if provided
            new_instructions = args.get('special_instructions')
            if new_instructions != order.special_instructions:
                order.special_instructions = new_instructions
                updated_fields.append("special_instructions")

            # Handle item updates (complex operation)
            new_items = args.get('items')
            if new_items:
                # Validate new items
                validated_items = []
                total_amount = 0.0

                for item_data in new_items:
                    product_sku = item_data.get('product_sku')
                    quantity = item_data.get('quantity', 1)

                    product = self.product_repo.get_by_sku(product_sku.upper())
                    if not product:
                        return {
                            "success": False,
                            "error": f"Product {product_sku} not found"
                        }

                    if not product.is_in_stock(quantity):
                        return {
                            "success": False,
                            "error": f"Insufficient stock for {product.name}"
                        }

                    unit_price = product.price
                    item_total = unit_price * quantity
                    total_amount += item_total

                    validated_items.append(OrderItemModel(
                        product_sku=product.sku,
                        product_name=product.name,
                        quantity=quantity,
                        unit_price=unit_price,
                        total_price=item_total
                    ))

                # Restore stock from original items
                for old_item in order.items:
                    product = self.product_repo.get_by_sku(
                        old_item.product_sku)
                    if product:
                        self.product_repo.update_stock(
                            product.id, old_item.quantity)

                # Update order items
                order.items = validated_items
                order.total_amount = total_amount
                updated_fields.append("items")

                # Update stock for new items
                for new_item in validated_items:
                    product = self.product_repo.get_by_sku(
                        new_item.product_sku)
                    if product:
                        self.product_repo.update_stock(
                            product.id, -new_item.quantity)

            if updated_fields:
                # Save updated order
                self.order_repo.update(order.id, order)

                return {
                    "success": True,
                    "order_id": order.id,
                    "order_number": order.order_number,
                    "updated_fields": updated_fields,
                    "new_total": order.total_amount,
                    "message": f"Order {order_number} has been successfully updated"
                }
            else:
                return {
                    "success": False,
                    "error": "No changes provided for order update"
                }

        except Exception as e:
            logger.error(f"Order update error: {e}")
            return {
                "success": False,
                "error": f"Failed to update order: {str(e)}"
            }

    async def _cancel_order(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cancel an existing order.

        Args:
            args: Order cancellation details with verification

        Returns:
            Dictionary with cancellation results
        """
        try:
            order_number = args.get('order_number', '').upper()
            customer_email = args.get('customer_email', '').lower()
            reason = args.get('reason', 'Customer requested cancellation')

            # Get and verify order
            order = self.order_repo.get_by_number(order_number)
            if not order:
                return {
                    "success": False,
                    "error": f"Order {order_number} not found"
                }

            if order.customer_email and order.customer_email.lower() != customer_email:
                return {
                    "success": False,
                    "error": "Email verification failed"
                }

            # Check if order can be cancelled
            if not order.can_be_cancelled():
                return {
                    "success": False,
                    "error": f"Order cannot be cancelled. Current status: {order.get_status_display()}"
                }

            # Restore stock for all items
            for item in order.items:
                product = self.product_repo.get_by_sku(item.product_sku)
                if product:
                    self.product_repo.update_stock(product.id, item.quantity)

            # Update order status to cancelled
            self.order_repo.update_status(order.id, OrderStatus.CANCELLED)

            return {
                "success": True,
                "order_id": order.id,
                "order_number": order.order_number,
                "refunded_amount": order.total_amount,
                "cancellation_reason": reason,
                "message": f"Order {order_number} has been cancelled. Refund of ${order.total_amount:.2f} will be processed."
            }

        except Exception as e:
            logger.error(f"Order cancellation error: {e}")
            return {
                "success": False,
                "error": f"Failed to cancel order: {str(e)}"
            }
