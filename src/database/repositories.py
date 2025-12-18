"""
Repository pattern implementation for database operations.

AI Assistant Notes:
- Clean separation of data access logic
- Type-safe operations with Pydantic model integration
- Error handling and logging
- Performance optimization with proper indexing
- Transaction management
"""

from src.utils import observe
from typing import List, Optional, Dict, Any, Tuple
import logging
import json
from datetime import datetime, timedelta

from .connection import DatabaseConnection
from .models import ProductModel, OrderModel, OrderItemModel, CustomerModel, ConversationModel, OrderStatus
from src.utils import langfuse_client

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common functionality."""

    def __init__(self, db: DatabaseConnection, table_name: str):
        """
        Initialize base repository.

        Args:
            db: Database connection instance
            table_name: Name of the database table
        """
        self.db = db
        self.table_name = table_name

    def _row_to_model(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Convert database row to model-compatible dictionary."""
        # Handle datetime string conversion
        for key, value in row.items():
            if isinstance(value, str) and key.endswith('_at'):
                try:
                    row[key] = datetime.fromisoformat(value)
                except ValueError:
                    pass  # Keep original value if not a valid datetime

        # Handle enum conversions
        if 'category' in row and row['category']:
            try:
                from .models import ProductCategory
                if isinstance(row['category'], str):
                    row['category'] = ProductCategory(row['category'])
            except Exception as e:
                logger.warning(f"Failed to convert category to enum: {e}")

        if 'order_status' in row and row['order_status']:
            try:
                from .models import OrderStatus
                if isinstance(row['order_status'], str):
                    row['order_status'] = OrderStatus(row['order_status'])
            except Exception as e:
                logger.warning(f"Failed to convert order_status to enum: {e}")

        if 'message_type' in row and row['message_type']:
            try:
                from .models import MessageType
                if isinstance(row['message_type'], str):
                    row['message_type'] = MessageType(row['message_type'])
            except Exception as e:
                logger.warning(f"Failed to convert message_type to enum: {e}")

        # Handle dimensions JSON conversion
        if row.get('dimensions') and isinstance(row['dimensions'], str):
            try:
                from .models import Dimensions
                row['dimensions'] = Dimensions.from_json(row['dimensions'])
            except Exception as e:
                # Log the error for debugging and set to None
                logger.warning(
                    f"Failed to parse dimensions JSON '{row['dimensions'][:50]}...': {e}")
                row['dimensions'] = None

        return row


class ProductRepository(BaseRepository):
    """Repository for product operations."""

    def __init__(self, db: DatabaseConnection):
        super().__init__(db, "products")

    @observe(name="database_product_creation", as_type="tool")
    def create(self, product: ProductModel) -> int:
        """
        Create a new product.

        Args:
            product: Product model with data

        Returns:
            ID of the created product
        """
        query = """
        INSERT INTO products (
            sku, name, description, price, category, stock_count,
            weight_kg, dimensions, is_active
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        # Handle category conversion (enum or string)
        category_value = product.category.value if hasattr(
            product.category, 'value') else str(product.category)

        params = (
            product.sku,
            product.name,
            product.description,
            product.price,
            category_value,
            product.stock_count,
            product.weight_kg,
            product.dimensions.to_json() if product.dimensions else None,
            product.is_active
        )

        product_id = self.db.execute_insert(query, params)
        logger.info(f"Created product {product.sku} with ID {product_id}")
        return product_id

    def get_by_id(self, product_id: int) -> Optional[ProductModel]:
        """
        Get product by ID.

        Args:
            product_id: Product database ID

        Returns:
            Product model or None if not found
        """
        query = "SELECT * FROM products WHERE id = ?"
        rows = self.db.execute_query(query, (product_id,))

        if not rows:
            return None

        row = self._row_to_model(rows[0])
        return ProductModel(**row)

    def get_by_sku(self, sku: str) -> Optional[ProductModel]:
        """
        Get product by SKU.

        Args:
            sku: Product SKU

        Returns:
            Product model or None if not found
        """
        query = "SELECT * FROM products WHERE sku = ? AND is_active = 1"
        rows = self.db.execute_query(query, (sku.upper(),))

        if not rows:
            return None

        row = self._row_to_model(rows[0])
        return ProductModel(**row)

    @observe(name="database_product_search", as_type="tool")
    def search(self, query_text: str = "", category: Optional[str] = None,
               limit: int = 20, offset: int = 0) -> List[ProductModel]:
        """
        Search products with optional filters.

        Args:
            query_text: Search term for name/description
            category: Filter by category
            limit: Maximum number of results
            offset: Results offset for pagination

        Returns:
            List of matching products
        """
        base_query = "SELECT * FROM products WHERE is_active = 1"
        params = []

        if query_text:
            base_query += " AND (name LIKE ? OR description LIKE ?)"
            params.extend([f"%{query_text}%", f"%{query_text}%"])

        if category:
            base_query += " AND category = ?"
            params.append(category)

        base_query += " ORDER BY name LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self.db.execute_query(base_query, tuple(params))
        return [ProductModel(**self._row_to_model(row)) for row in rows]

    def update_stock(self, product_id: int, quantity_change: int) -> bool:
        """
        Update product stock quantity.

        Args:
            product_id: Product ID
            quantity_change: Change in stock (positive or negative)

        Returns:
            True if update successful, False otherwise
        """
        query = """
        UPDATE products
        SET stock_count = stock_count + ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ? AND stock_count + ? >= 0
        """
        success = self.db.execute_update(
            query, (quantity_change, product_id, quantity_change)) > 0

        if success:
            logger.info(
                f"Updated stock for product {product_id} by {quantity_change}")
        else:
            logger.warning(f"Failed to update stock for product {product_id}")

        return success

    def get_low_stock_products(self, threshold: int = 5) -> List[ProductModel]:
        """
        Get products with low stock.

        Args:
            threshold: Stock count threshold

        Returns:
            List of low stock products
        """
        query = """
        SELECT * FROM products
        WHERE is_active = 1 AND stock_count <= ?
        ORDER BY stock_count ASC
        """
        rows = self.db.execute_query(query, (threshold,))
        return [ProductModel(**self._row_to_model(row)) for row in rows]

    def get_by_category(self, category: str) -> List[ProductModel]:
        """
        Get all products in a category.

        Args:
            category: Product category

        Returns:
            List of products in the category
        """
        query = "SELECT * FROM products WHERE category = ? AND is_active = 1 ORDER BY name"
        rows = self.db.execute_query(query, (category,))
        return [ProductModel(**self._row_to_model(row)) for row in rows]

    def update(self, product_id: int, product: ProductModel) -> bool:
        """
        Update product information.

        Args:
            product_id: Product ID to update
            product: Updated product data

        Returns:
            True if update successful
        """
        query = """
        UPDATE products SET
            name = ?, description = ?, price = ?, category = ?,
            stock_count = ?, weight_kg = ?, dimensions = ?, is_active = ?
        WHERE id = ?
        """
        # Handle category conversion (enum or string)
        category_value = product.category.value if hasattr(
            product.category, 'value') else str(product.category)

        params = (
            product.name,
            product.description,
            product.price,
            category_value,
            product.stock_count,
            product.weight_kg,
            product.dimensions.to_json() if product.dimensions else None,
            product.is_active,
            product_id
        )

        success = self.db.execute_update(query, params) > 0
        if success:
            logger.info(f"Updated product {product_id}")
        return success

    @observe(name="database_product_search_by_name", as_type="tool")
    def search_by_name(self, name: str, limit: int = 5) -> List[ProductModel]:
        """
        Search products by name using fuzzy matching.

        Args:
            name: Product name to search for
            limit: Maximum number of results to return

        Returns:
            List of matching product models
        """
        query = """
        SELECT id, sku, name, description, price, category, stock_count,
               weight_kg, dimensions, is_active, created_at, updated_at
        FROM products
        WHERE is_active = true
        AND (LOWER(name) LIKE LOWER(?) OR LOWER(name) LIKE LOWER(?) OR LOWER(name) LIKE LOWER(?))
        ORDER BY
            CASE
                WHEN LOWER(name) = LOWER(?) THEN 1
                WHEN LOWER(name) LIKE LOWER(?) THEN 2
                WHEN LOWER(name) LIKE LOWER(?) THEN 3
                ELSE 4
            END,
            stock_count DESC
        LIMIT ?
        """

        params = (
            f"%{name}%",  # contains
            f"{name}%",   # starts with
            f"%{name}",   # ends with
            name,         # exact match
            f"{name}%",   # starts with for ordering
            f"%{name}%",  # contains for ordering
            limit
        )

        results = self.db.execute_query(query, params)
        products = []

        for row in results:
            try:
                # Parse dimensions if present
                if row['dimensions']:
                    try:
                        dimensions = json.loads(row['dimensions'])
                    except json.JSONDecodeError:
                        dimensions = None
                else:
                    dimensions = None

                product = ProductModel(
                    id=row['id'],
                    sku=row['sku'],
                    name=row['name'],
                    description=row['description'],
                    price=float(row['price']),
                    category=row['category'],
                    stock_count=row['stock_count'],
                    weight_kg=row['weight_kg'],
                    dimensions=dimensions,
                    is_active=bool(row['is_active']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
                products.append(product)
            except Exception as e:
                logger.error(f"Error creating product model from row: {e}")
                continue

        return products


class OrderRepository(BaseRepository):
    """Repository for order operations."""

    def __init__(self, db: DatabaseConnection):
        super().__init__(db, "orders")

    @observe(name="database_order_creation", as_type="tool")
    def create(self, order: OrderModel) -> int:
        """
        Create a new order.

        Args:
            order: Order model with data

        Returns:
            ID of the created order
        """
        # Generate order number if not provided
        order_number = order.order_number or order.generate_order_number()

        query = """
        INSERT INTO orders (
            order_number, customer_name, customer_email, customer_phone,
            delivery_address, order_status, total_amount, items,
            special_instructions, estimated_delivery
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            order_number,
            order.customer_name,
            order.customer_email,
            order.customer_phone,
            order.delivery_address,
            order.order_status.value if hasattr(
                order.order_status, 'value') else str(order.order_status),
            order.total_amount,
            str([item.model_dump() for item in order.items]),
            order.special_instructions,
            order.estimated_delivery
        )

        order_id = self.db.execute_insert(query, params)

        # Create order items
        for item in order.items:
            self._create_order_item(order_id, item)

        logger.info(f"Created order {order_number} with ID {order_id}")
        return order_id

    def _create_order_item(self, order_id: int, item: OrderItemModel) -> int:
        """Create an order item record."""
        query = """
        INSERT INTO order_items (order_id, product_id, quantity, unit_price, total_price)
        VALUES (?, ?, ?, ?, ?)
        """
        # Resolve product_id from SKU
        product_repo = ProductRepository(self.db)
        product = product_repo.get_by_sku(item.product_sku)

        if not product:
            raise ValueError(f"Product with SKU {item.product_sku} not found")

        params = (order_id, product.id, item.quantity,
                  item.unit_price, item.total_price)
        return self.db.execute_insert(query, params)

    def get_by_id(self, order_id: int) -> Optional[OrderModel]:
        """
        Get order by ID.

        Args:
            order_id: Order database ID

        Returns:
            Order model or None if not found
        """
        query = "SELECT * FROM orders WHERE id = ?"
        rows = self.db.execute_query(query, (order_id,))

        if not rows:
            return None

        row = self._row_to_model(rows[0])

        # Parse items JSON
        items_data = eval(row['items']) if isinstance(
            row['items'], str) else row['items']
        items = [OrderItemModel(**item_data) for item_data in items_data]

        row['items'] = items
        return OrderModel(**row)

    def get_by_number(self, order_number: str) -> Optional[OrderModel]:
        """
        Get order by order number.

        Args:
            order_number: Order number

        Returns:
            Order model or None if not found
        """
        query = "SELECT * FROM orders WHERE order_number = ?"
        rows = self.db.execute_query(query, (order_number,))

        if not rows:
            return None

        row = self._row_to_model(rows[0])

        # Parse items JSON
        items_data = eval(row['items']) if isinstance(
            row['items'], str) else row['items']
        items = [OrderItemModel(**item_data) for item_data in items_data]

        row['items'] = items
        return OrderModel(**row)

    def get_by_customer_email(self, email: str, limit: int = 10) -> List[OrderModel]:
        """
        Get orders by customer email.

        Args:
            email: Customer email
            limit: Maximum number of orders to return

        Returns:
            List of customer orders
        """
        query = """
        SELECT * FROM orders
        WHERE customer_email = ?
        ORDER BY created_at DESC
        LIMIT ?
        """
        rows = self.db.execute_query(query, (email.lower(), limit))

        orders = []
        for row in rows:
            row = self._row_to_model(row)
            # Parse items JSON
            items_data = eval(row['items']) if isinstance(
                row['items'], str) else row['items']
            items = [OrderItemModel(**item_data) for item_data in items_data]
            row['items'] = items
            orders.append(OrderModel(**row))

        return orders

    def update_status(self, order_id: int, new_status: OrderStatus) -> bool:
        """
        Update order status.

        Args:
            order_id: Order ID
            new_status: New order status

        Returns:
            True if update successful
        """
        query = """
        UPDATE orders
        SET order_status = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """
        success = self.db.execute_update(
            query, (new_status.value, order_id)) > 0

        if success:
            logger.info(
                f"Updated order {order_id} status to {new_status.value}")

        return success

    def get_recent_orders(self, hours: int = 24, limit: int = 50) -> List[OrderModel]:
        """
        Get recent orders within specified hours.

        Args:
            hours: Number of hours to look back
            limit: Maximum number of orders

        Returns:
            List of recent orders
        """
        query = """
        SELECT * FROM orders
        WHERE created_at >= datetime('now', '-{} hours')
        ORDER BY created_at DESC
        LIMIT ?
        """.format(hours)

        rows = self.db.execute_query(query, (limit,))

        orders = []
        for row in rows:
            row = self._row_to_model(row)
            # Parse items JSON
            items_data = eval(row['items']) if isinstance(
                row['items'], str) else row['items']
            items = [OrderItemModel(**item_data) for item_data in items_data]
            row['items'] = items
            orders.append(OrderModel(**row))

        return orders

    def get_order_statistics(self) -> Dict[str, Any]:
        """
        Get order statistics.

        Returns:
            Dictionary with order statistics
        """
        stats_query = """
        SELECT
            COUNT(*) as total_orders,
            COUNT(CASE WHEN order_status = 'pending' THEN 1 END) as pending_orders,
            COUNT(CASE WHEN order_status = 'delivered' THEN 1 END) as delivered_orders,
            COALESCE(SUM(total_amount), 0) as total_revenue,
            COALESCE(AVG(total_amount), 0) as average_order_value
        FROM orders
        WHERE created_at >= datetime('now', '-30 days')
        """

        result = self.db.execute_query(stats_query)
        return result[0] if result else {}


class ConversationRepository(BaseRepository):
    """Repository for conversation management."""

    def __init__(self, db: DatabaseConnection):
        super().__init__(db, "conversations")

    def save_message(self, conversation: ConversationModel) -> int:
        """
        Save a conversation message.

        Args:
            conversation: Conversation model

        Returns:
            ID of the saved message
        """
        query = """
        INSERT INTO conversations (
            session_id, user_id, message_sequence, message_type,
            content, metadata, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            conversation.session_id,
            conversation.user_id,
            conversation.message_sequence,
            conversation.message_type.value if hasattr(
                conversation.message_type, 'value') else conversation.message_type,
            conversation.content,
            str(conversation.metadata) if conversation.metadata else None,
            conversation.timestamp or datetime.now()
        )

        message_id = self.db.execute_insert(query, params)
        return message_id

    def get_conversation_history(self, session_id: str, limit: int = 20) -> List[ConversationModel]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve

        Returns:
            List of conversation messages in chronological order
        """
        query = """
        SELECT * FROM conversations
        WHERE session_id = ?
        ORDER BY message_sequence ASC
        LIMIT ?
        """
        rows = self.db.execute_query(query, (session_id, limit))

        conversations = []
        for row in rows:
            row = self._row_to_model(row)
            # Parse metadata JSON
            if row['metadata']:
                try:
                    row['metadata'] = eval(row['metadata']) if isinstance(
                        row['metadata'], str) else row['metadata']
                except:
                    row['metadata'] = {}
            conversations.append(ConversationModel(**row))

        return conversations

    def get_last_sequence_number(self, session_id: str) -> int:
        """
        Get the last message sequence number for a session.

        Args:
            session_id: Session identifier

        Returns:
            Last sequence number or 0 if no messages exist
        """
        query = """
        SELECT COALESCE(MAX(message_sequence), 0) as last_sequence
        FROM conversations
        WHERE session_id = ?
        """
        result = self.db.execute_query(query, (session_id,))
        return result[0]['last_sequence'] if result else 0

    def delete_old_conversations(self, days: int = 30) -> int:
        """
        Delete old conversation messages.

        Args:
            days: Number of days to keep messages

        Returns:
            Number of deleted messages
        """
        query = """
        DELETE FROM conversations
        WHERE timestamp < datetime('now', '-{} days')
        """.format(days)

        deleted_count = self.db.execute_update(query)
        logger.info(f"Deleted {deleted_count} old conversation messages")
        return deleted_count
