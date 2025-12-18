"""
Database seeding script for products.

AI Assistant Notes:
- Loads product data from JSON files into the database
- Creates embeddings for RAG functionality
- Handles duplicate SKUs gracefully
- Provides progress tracking and error handling
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.database import DatabaseConnection, ProductRepository, DatabaseMigrations
from src.database.models import ProductModel, ProductCategory

logger = logging.getLogger(__name__)


class ProductSeeder:
    """Handles seeding products into the database."""

    def __init__(self, db_path: str = "database/orders.db"):
        """
        Initialize product seeder.

        Args:
            db_path: Path to SQLite database
        """
        self.db = DatabaseConnection(db_path)
        self.product_repo = ProductRepository(self.db)
        self.migrations = DatabaseMigrations(self.db)

    def setup_database(self) -> None:
        """Set up database with latest migrations."""
        logger.info("Setting up database...")
        self.migrations.run_migrations()
        logger.info("Database setup complete")

    def load_products_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load products from a JSON file.

        Args:
            file_path: Path to product JSON file

        Returns:
            List of product dictionaries
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                products = json.load(f)
            logger.info(f"Loaded {len(products)} products from {file_path}")
            return products
        except Exception as e:
            logger.error(f"Failed to load products from {file_path}: {e}")
            return []

    def create_product_model(self, product_data: Dict[str, Any]) -> ProductModel:
        """
        Create a ProductModel from dictionary data.

        Args:
            product_data: Product dictionary from JSON

        Returns:
            ProductModel instance
        """
        # Convert dimensions data if present
        if 'dimensions' in product_data and product_data['dimensions']:
            from src.database.models import Dimensions
            product_data['dimensions'] = Dimensions(**product_data['dimensions'])

        # Convert category string to enum
        if 'category' in product_data:
            product_data['category'] = ProductCategory(product_data['category'])

        return ProductModel(**product_data)

    def seed_products_from_file(self, file_path: Path) -> int:
        """
        Seed products from a single JSON file.

        Args:
            file_path: Path to product JSON file

        Returns:
            Number of products successfully seeded
        """
        products_data = self.load_products_from_file(file_path)
        if not products_data:
            return 0

        seeded_count = 0
        for product_data in products_data:
            try:
                # Check if product already exists
                existing_product = self.product_repo.get_by_sku(product_data['sku'])
                if existing_product:
                    logger.info(f"Product {product_data['sku']} already exists, skipping")
                    continue

                # Create and save product
                product_model = self.create_product_model(product_data)
                product_id = self.product_repo.create(product_model)
                logger.info(f"Created product: {product_model.name} (ID: {product_id})")
                seeded_count += 1

            except Exception as e:
                logger.error(f"Failed to seed product {product_data.get('sku', 'unknown')}: {e}")

        return seeded_count

    def seed_all_products(self) -> Dict[str, int]:
        """
        Seed products from all JSON files in the products directory.

        Returns:
            Dictionary with counts per category
        """
        products_dir = Path(__file__).parent.parent / "products"
        results = {}

        # Process each JSON file
        for json_file in products_dir.glob("*.json"):
            category_name = json_file.stem
            logger.info(f"Processing {category_name} products...")

            count = self.seed_products_from_file(json_file)
            results[category_name] = count

        return results

    def get_seeding_summary(self) -> Dict[str, Any]:
        """
        Get a summary of seeded products.

        Returns:
            Dictionary with seeding statistics
        """
        # Get total products
        query = "SELECT COUNT(*) as total, category, COUNT(CASE WHEN is_active = 1 THEN 1 END) as active FROM products GROUP BY category"
        category_stats = self.db.execute_query(query)

        # Get total value
        value_query = "SELECT COUNT(*) as total_products, COALESCE(SUM(price * stock_count), 0) as total_value FROM products WHERE is_active = 1"
        total_stats = self.db.execute_query(value_query)[0] if self.db.execute_query(value_query) else {"total_products": 0, "total_value": 0}

        return {
            "category_counts": {row['category']: row['total'] for row in category_stats},
            "total_products": total_stats['total_products'],
            "total_inventory_value": round(total_stats['total_value'], 2)
        }


def main():
    """Main seeding function."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info("Starting product database seeding...")

    # Initialize seeder
    seeder = ProductSeeder()

    try:
        # Set up database
        seeder.setup_database()

        # Seed all products
        results = seeder.seed_all_products()

        # Print results
        logger.info("Seeding complete! Results:")
        for category, count in results.items():
            logger.info(f"  {category}: {count} products")

        # Print summary
        summary = seeder.get_seeding_summary()
        logger.info(f"\nDatabase Summary:")
        logger.info(f"Total Products: {summary['total_products']}")
        logger.info(f"Total Inventory Value: ${summary['total_inventory_value']:,.2f}")
        logger.info(f"Products by Category:")
        for category, count in summary['category_counts'].items():
            logger.info(f"  {category}: {count}")

    except Exception as e:
        logger.error(f"Seeding failed: {e}")
        raise


if __name__ == "__main__":
    main()