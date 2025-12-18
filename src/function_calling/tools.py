"""
Function tool definitions for order processing.

AI Assistant Notes:
- Defines structured function schemas for OpenAI Function Calling
- Product search and order creation capabilities
- Parameter validation with clear descriptions
- Type-safe function definitions
- Error handling for edge cases
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class SortOrder(str, Enum):
    """Sort order options for product search."""
    RELEVANCE = "relevance"
    PRICE_LOW = "price_low"
    PRICE_HIGH = "price_high"
    NAME_AZ = "name_az"
    NEWEST = "newest"
    POPULARITY = "popularity"


class SearchProductsInput(BaseModel):
    """Input model for product search function."""
    query: Optional[str] = Field(None, description="Search query for product names or descriptions")
    category: Optional[str] = Field(None, description="Product category to filter by (electronics, books, clothing, home_goods)")
    min_price: Optional[float] = Field(None, ge=0, description="Minimum price filter")
    max_price: Optional[float] = Field(None, ge=0, description="Maximum price filter")
    in_stock_only: bool = Field(True, description="Only show products that are currently in stock")
    limit: int = Field(10, ge=1, le=50, description="Maximum number of results to return")
    sort_by: SortOrder = Field(SortOrder.RELEVANCE, description="How to sort the results")


class CreateOrderItemInput(BaseModel):
    """Input model for individual order items."""
    product_sku: str = Field(..., description="Product SKU (unique identifier)")
    quantity: int = Field(..., ge=1, le=100, description="Quantity of the product to order")


class CreateOrderInput(BaseModel):
    """Input model for order creation function."""
    customer_name: str = Field(..., min_length=2, description="Customer's full name")
    customer_email: Optional[str] = Field(None, description="Customer's email address for order confirmation")
    customer_phone: Optional[str] = Field(None, description="Customer's phone number for delivery contact")
    delivery_address: str = Field(..., min_length=10, description="Complete delivery address including street, city, and postal code")
    items: List[CreateOrderItemInput] = Field(..., min_items=1, description="List of products to order")
    special_instructions: Optional[str] = Field(None, description="Special delivery instructions or preferences")


class CheckOrderStatusInput(BaseModel):
    """Input model for order status checking."""
    order_number: str = Field(..., description="Order number (format: ORD-XXXXXX)")
    customer_email: Optional[str] = Field(None, description="Customer email for verification")


class UpdateOrderInput(BaseModel):
    """Input model for order updates."""
    order_number: str = Field(..., description="Order number to update")
    customer_email: str = Field(..., description="Customer email for verification")
    delivery_address: Optional[str] = Field(None, description="New delivery address")
    special_instructions: Optional[str] = Field(None, description="Updated special instructions")
    items: Optional[List[CreateOrderItemInput]] = Field(None, description="Updated order items")


class CancelOrderInput(BaseModel):
    """Input model for order cancellation."""
    order_number: str = Field(..., description="Order number to cancel")
    customer_email: str = Field(..., description="Customer email for verification")
    reason: Optional[str] = Field(None, description="Reason for cancellation")


class FunctionTools:
    """Collection of function tool definitions for the chatbot."""

    @staticmethod
    def get_search_products_tool() -> Dict[str, Any]:
        """Get the search_products function definition."""
        return {
            "name": "search_products",
            "description": "Search for products in the catalog based on query, category, price range, or other filters. Returns detailed product information including prices and availability.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term to find products by name or description. Leave empty to browse all products."
                    },
                    "category": {
                        "type": "string",
                        "enum": ["electronics", "books", "clothing", "home_goods"],
                        "description": "Filter products by category"
                    },
                    "min_price": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Minimum price to filter products"
                    },
                    "max_price": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Maximum price to filter products"
                    },
                    "in_stock_only": {
                        "type": "boolean",
                        "description": "Only return products that are currently in stock",
                        "default": True
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Maximum number of products to return",
                        "default": 10
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["price_low", "price_high", "name_az", "newest", "popularity"],
                        "description": "How to sort the search results",
                        "default": "popularity"
                    }
                },
                "required": []
            }
        }

    @staticmethod
    def get_create_order_tool() -> Dict[str, Any]:
        """Get the create_order function definition."""
        return {
            "name": "create_order",
            "description": "Create a new customer order with products and delivery details. Automatically calculates total cost and generates order confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_name": {
                        "type": "string",
                        "description": "Customer's full name for order processing"
                    },
                    "customer_email": {
                        "type": "string",
                        "description": "Customer's email address for order confirmation and tracking"
                    },
                    "customer_phone": {
                        "type": "string",
                        "description": "Customer's phone number for delivery coordination"
                    },
                    "delivery_address": {
                        "type": "string",
                        "description": "Complete delivery address including street address, city, state/province, and postal code"
                    },
                    "items": {
                        "type": "array",
                        "description": "List of products to include in the order",
                        "items": {
                            "type": "object",
                            "properties": {
                                "product_sku": {
                                    "type": "string",
                                    "description": "Product SKU identifier (from search results)"
                                },
                                "quantity": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 100,
                                    "description": "Quantity of this product to order"
                                }
                            },
                            "required": ["product_sku", "quantity"]
                        }
                    },
                    "special_instructions": {
                        "type": "string",
                        "description": "Special delivery instructions, preferences, or notes for the order"
                    }
                },
                "required": ["customer_name", "delivery_address", "items"]
            }
        }

    @staticmethod
    def get_check_order_status_tool() -> Dict[str, Any]:
        """Get the check_order_status function definition."""
        return {
            "name": "check_order_status",
            "description": "Check the current status of an existing order including preparation progress, shipping information, and estimated delivery date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_number": {
                        "type": "string",
                        "description": "Order number (format: ORD-XXXXXX) to check"
                    },
                    "customer_email": {
                        "type": "string",
                        "description": "Customer email associated with the order for verification"
                    }
                },
                "required": ["order_number"]
            }
        }

    @staticmethod
    def get_update_order_tool() -> Dict[str, Any]:
        """Get the update_order function definition."""
        return {
            "name": "update_order",
            "description": "Update an existing order's delivery address, special instructions, or items. Only possible before order is shipped.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_number": {
                        "type": "string",
                        "description": "Order number to update"
                    },
                    "customer_email": {
                        "type": "string",
                        "description": "Customer email associated with the order for verification"
                    },
                    "delivery_address": {
                        "type": "string",
                        "description": "New delivery address if updating"
                    },
                    "special_instructions": {
                        "type": "string",
                        "description": "Updated special instructions for delivery"
                    },
                    "items": {
                        "type": "array",
                        "description": "Updated list of order items",
                        "items": {
                            "type": "object",
                            "properties": {
                                "product_sku": {
                                    "type": "string",
                                    "description": "Product SKU identifier"
                                },
                                "quantity": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 100,
                                    "description": "Quantity of this product"
                                }
                            },
                            "required": ["product_sku", "quantity"]
                        }
                    }
                },
                "required": ["order_number", "customer_email"]
            }
        }

    @staticmethod
    def get_cancel_order_tool() -> Dict[str, Any]:
        """Get the cancel_order function definition."""
        return {
            "name": "cancel_order",
            "description": "Cancel an existing order. Only possible before order is shipped. Process refund according to cancellation policy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_number": {
                        "type": "string",
                        "description": "Order number to cancel"
                    },
                    "customer_email": {
                        "type": "string",
                        "description": "Customer email associated with the order for verification"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for cancellation"
                    }
                },
                "required": ["order_number", "customer_email"]
            }
        }

    @classmethod
    def get_all_tools(cls) -> List[Dict[str, Any]]:
        """Get all available function tools in OpenAI API format."""
        tools = [
            cls.get_search_products_tool(),
            cls.get_create_order_tool(),
            cls.get_check_order_status_tool(),
            cls.get_update_order_tool(),
            cls.get_cancel_order_tool()
        ]

        # Wrap each tool in the OpenAI API format
        return [
            {
                "type": "function",
                "function": tool
            }
            for tool in tools
        ]

    @classmethod
    def get_tool_names(cls) -> List[str]:
        """Get list of all available tool names."""
        return [tool["function"]["name"] for tool in cls.get_all_tools()]

    @classmethod
    def get_tool_by_name(cls, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific tool definition by name."""
        for tool in cls.get_all_tools():
            if tool["function"]["name"] == tool_name:
                return tool["function"]
        return None