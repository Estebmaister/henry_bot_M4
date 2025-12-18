"""
Pydantic models for data validation and serialization.

AI Assistant Notes:
- Comprehensive field validation with custom constraints
- Business logic validation through custom validators
- Type safety with proper type hints
- Serialization/deserialization support
- Integration with database layer
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any, Union, Self
from datetime import datetime
from enum import Enum
import re
import json


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PREPARING = "preparing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class MessageType(str, Enum):
    """Conversation message type enumeration."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ProductCategory(str, Enum):
    """Product category enumeration."""
    ELECTRONICS = "electronics"
    BOOKS = "books"
    CLOTHING = "clothing"
    HOME_GOODS = "home_goods"
    SPORTS = "sports"
    TOYS = "toys"
    BEAUTY = "beauty"
    FOOD = "food"


class Dimensions(BaseModel):
    """Product dimensions model."""
    length_cm: float = Field(..., gt=0, description="Length in centimeters")
    width_cm: float = Field(..., gt=0, description="Width in centimeters")
    height_cm: float = Field(..., gt=0, description="Height in centimeters")

    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "Dimensions":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(**data)


class ProductModel(BaseModel):
    """Product model with comprehensive validation."""

    # Database fields
    id: Optional[int] = Field(None, description="Database ID")
    sku: str = Field(..., min_length=3, max_length=50, description="Unique product identifier")
    name: str = Field(..., min_length=1, max_length=200, description="Product name")
    description: Optional[str] = Field(None, max_length=2000, description="Product description")
    price: float = Field(..., gt=0, le=99999.99, description="Product price")
    category: ProductCategory = Field(..., description="Product category")
    stock_count: int = Field(..., ge=0, le=1000000, description="Available stock quantity")
    weight_kg: Optional[float] = Field(None, ge=0, le=1000, description="Weight in kilograms")
    dimensions: Optional[Dimensions] = Field(None, description="Product dimensions")
    is_active: bool = Field(True, description="Product availability status")

    # Timestamps
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

    @field_validator('sku')
    @classmethod
    def validate_sku(cls, v: str) -> str:
        """Validate SKU format."""
        if not re.match(r'^[A-Z0-9-_]+$', v.upper()):
            raise ValueError('SKU must contain only uppercase letters, numbers, hyphens, and underscores')
        return v.upper()

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate product name."""
        if not v.strip():
            raise ValueError('Product name cannot be empty')
        return v.strip()

    @field_validator('dimensions')
    @classmethod
    def validate_dimensions(cls, v):
        """Validate and convert dimensions from JSON string if needed."""
        if v is None:
            return None
        if isinstance(v, str):
            return Dimensions.from_json(v)
        return v

    @field_validator('description')
    @classmethod
    def validate_description(cls, v: Optional[str]) -> Optional[str]:
        """Validate product description."""
        if v:
            return v.strip()
        return v

    @field_validator('price')
    @classmethod
    def validate_price(cls, v: float) -> float:
        """Validate price format."""
        return round(v, 2)

    def is_in_stock(self, quantity: int = 1) -> bool:
        """Check if product is in stock for given quantity."""
        return self.is_active and self.stock_count >= quantity

    def get_availability_status(self) -> str:
        """Get human-readable availability status."""
        if not self.is_active:
            return "discontinued"
        elif self.stock_count == 0:
            return "out of stock"
        elif self.stock_count < 5:
            return "low stock"
        else:
            return "in stock"

    def calculate_total_price(self, quantity: int) -> float:
        """Calculate total price for given quantity."""
        return round(self.price * quantity, 2)

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class OrderItemModel(BaseModel):
    """Order item model with validation."""

    # Fields
    product_sku: str = Field(..., min_length=3, description="Product SKU")
    product_name: Optional[str] = Field(None, description="Product name (cached)")
    quantity: int = Field(..., gt=0, le=100, description="Order quantity")
    unit_price: float = Field(..., gt=0, description="Price per unit")
    total_price: Optional[float] = Field(None, description="Total price for this item")

    @field_validator('quantity')
    @classmethod
    def validate_quantity(cls, v: int) -> int:
        """Validate order quantity."""
        if v <= 0:
            raise ValueError('Quantity must be positive')
        return v

    @field_validator('unit_price')
    @classmethod
    def validate_unit_price(cls, v: float) -> float:
        """Validate unit price format."""
        return round(v, 2)

    @model_validator(mode='before')
    @classmethod
    def calculate_total_price(cls, values: dict) -> dict:
        """Calculate total price if not provided."""
        if 'total_price' not in values or values['total_price'] is None:
            quantity = values.get('quantity', 0)
            unit_price = values.get('unit_price', 0)
            values['total_price'] = round(quantity * unit_price, 2)
        return values

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class OrderModel(BaseModel):
    """Order model with comprehensive validation."""

    # Database fields
    id: Optional[int] = Field(None, description="Database ID")
    order_number: Optional[str] = Field(None, description="Unique order number")
    customer_name: str = Field(..., min_length=2, max_length=100, description="Customer full name")
    customer_email: Optional[str] = Field(None, description="Customer email address")
    customer_phone: Optional[str] = Field(None, description="Customer phone number")
    delivery_address: str = Field(..., min_length=10, max_length=500, description="Delivery address")
    order_status: OrderStatus = Field(OrderStatus.PENDING, description="Current order status")
    total_amount: Optional[float] = Field(None, description="Total order amount")
    items: List[OrderItemModel] = Field(..., min_items=1, description="Order items list")
    special_instructions: Optional[str] = Field(None, max_length=1000, description="Special delivery instructions")
    estimated_delivery: Optional[datetime] = Field(None, description="Estimated delivery date")

    # Timestamps
    created_at: Optional[datetime] = Field(None, description="Order creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

    @field_validator('customer_name')
    @classmethod
    def validate_customer_name(cls, v: str) -> str:
        """Validate customer name."""
        if not v.strip():
            raise ValueError('Customer name cannot be empty')
        return v.strip().title()

    @field_validator('customer_email')
    @classmethod
    def validate_customer_email(cls, v: Optional[str]) -> Optional[str]:
        """Validate email format."""
        if v and not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
            raise ValueError('Invalid email format')
        return v.lower().strip() if v else v

    @field_validator('customer_phone')
    @classmethod
    def validate_customer_phone(cls, v: Optional[str]) -> Optional[str]:
        """Validate phone number format."""
        if v:
            # Remove all non-digit characters
            digits = re.sub(r'[^\d]', '', v)
            if len(digits) < 10 or len(digits) > 15:
                raise ValueError('Phone number must have 10-15 digits')
            return digits
        return v

    @field_validator('delivery_address')
    @classmethod
    def validate_delivery_address(cls, v: str) -> str:
        """Validate delivery address."""
        if not v.strip():
            raise ValueError('Delivery address cannot be empty')
        return v.strip()

    @field_validator('order_number')
    @classmethod
    def validate_order_number(cls, v: Optional[str]) -> Optional[str]:
        """Validate order number format."""
        if v and not re.match(r'^ORD-\d{6}$', v):
            raise ValueError('Order number must be in format ORD-XXXXXX')
        return v

    @model_validator(mode='before')
    @classmethod
    def calculate_total_amount(cls, values: dict) -> dict:
        """Calculate total order amount."""
        if 'total_amount' not in values or values['total_amount'] is None:
            items = values.get('items', [])
            total = sum(item.total_price for item in items)
            values['total_amount'] = round(total, 2)
        return values

    @model_validator(mode='before')
    @classmethod
    def validate_order_items(cls, values: dict) -> dict:
        """Validate order items consistency."""
        items = values.get('items', [])
        if not items:
            raise ValueError('Order must have at least one item')

        # Check for duplicate SKUs
        sku_counts = {}
        for item in items:
            sku = item.product_sku
            sku_counts[sku] = sku_counts.get(sku, 0) + 1

        duplicates = [sku for sku, count in sku_counts.items() if count > 1]
        if duplicates:
            raise ValueError(f'Duplicate product SKUs in order: {duplicates}')

        return values

    def generate_order_number(self) -> str:
        """Generate a unique order number."""
        if self.order_number:
            return self.order_number

        # Generate timestamp-based order number
        timestamp = datetime.now().strftime("%y%m%d%H%M%S")
        return f"ORD-{timestamp[-6:]}"

    def get_item_count(self) -> int:
        """Get total number of items in order."""
        return sum(item.quantity for item in self.items)

    def can_be_cancelled(self) -> bool:
        """Check if order can be cancelled."""
        return self.order_status in [OrderStatus.PENDING, OrderStatus.CONFIRMED]

    def get_status_display(self) -> str:
        """Get human-readable status display."""
        status_map = {
            OrderStatus.PENDING: "Pending Confirmation",
            OrderStatus.CONFIRMED: "Order Confirmed",
            OrderStatus.PREPARING: "Preparing Your Order",
            OrderStatus.SHIPPED: "Out for Delivery",
            OrderStatus.DELIVERED: "Delivered",
            OrderStatus.CANCELLED: "Order Cancelled"
        }
        return status_map.get(self.order_status, "Unknown Status")

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class CustomerModel(BaseModel):
    """Customer model for order management."""

    id: Optional[int] = Field(None, description="Database ID")
    name: str = Field(..., min_length=2, max_length=100, description="Customer name")
    email: Optional[str] = Field(None, description="Customer email")
    phone: Optional[str] = Field(None, description="Customer phone")
    default_address: Optional[str] = Field(None, description="Default delivery address")
    created_at: Optional[datetime] = Field(None, description="Account creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate customer name."""
        if not v.strip():
            raise ValueError('Customer name cannot be empty')
        return v.strip().title()

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        """Validate email format."""
        if v and not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
            raise ValueError('Invalid email format')
        return v.lower().strip() if v else v

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class ConversationModel(BaseModel):
    """Conversation model for chat history management."""

    id: Optional[int] = Field(None, description="Database ID")
    session_id: str = Field(..., description="Chat session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    message_sequence: int = Field(..., ge=0, description="Message sequence number")
    message_type: MessageType = Field(..., description="Message type")
    content: str = Field(..., description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    timestamp: Optional[datetime] = Field(None, description="Message timestamp")

    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate message content."""
        if not v.strip():
            raise ValueError('Message content cannot be empty')
        return v.strip()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for OpenAI chat format."""
        return {
            "role": self.message_type.value if self.message_type != MessageType.ASSISTANT else "assistant",
            "content": self.content
        }

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }