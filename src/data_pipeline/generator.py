import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from src.logger import logger

PRODUCTS = [
    {"product_id": "PROD-TECH-001", "name": "Wireless Noise-Cancelling Headphones", "category": "Electronics", "base_price": 149.99, "base_demand": 45},
    {"product_id": "PROD-TECH-002", "name": "4K Ultra-HD Smart Monitor 27-inch", "category": "Electronics", "base_price": 299.99, "base_demand": 25},
    {"product_id": "PROD-TECH-003", "name": "Mechanical Gaming Keyboard RGB", "category": "Electronics", "base_price": 89.99, "base_demand": 60},
    {"product_id": "PROD-APPR-001", "name": "Classic Merino Wool Sweater", "category": "Apparel", "base_price": 79.50, "base_demand": 50},
    {"product_id": "PROD-APPR-002", "name": "Performance Waterproof Jacket", "category": "Apparel", "base_price": 120.00, "base_demand": 35},
    {"product_id": "PROD-HOME-001", "name": "Smart Multi-Cooker Air Fryer 6Qt", "category": "Home & Kitchen", "base_price": 119.00, "base_demand": 40},
    {"product_id": "PROD-HOME-002", "name": "Robotic Vacuum & Mop Combo", "category": "Home & Kitchen", "base_price": 349.99, "base_demand": 20},
    {"product_id": "PROD-HLTH-001", "name": "Vitamin C + Hyaluronic Acid Serum", "category": "Health & Beauty", "base_price": 34.00, "base_demand": 80},
    {"product_id": "PROD-HLTH-002", "name": "Electric Sonic Toothbrush Set", "category": "Health & Beauty", "base_price": 59.99, "base_demand": 55},
    {"product_id": "PROD-SPRT-001", "name": "Adjustable Dumbbell Pair 50lb", "category": "Sports", "base_price": 199.99, "base_demand": 30},
]

COUNTRIES = ["United Kingdom", "United States", "Germany", "France", "Australia", "Canada"]

class DataGenerator:
    """Generates high-fidelity e-commerce transaction streams with seasonalities and anomalies."""

    @staticmethod
    def generate_raw_ecommerce_data(
        days: int = 120,
        start_date: Optional[datetime] = None,
        include_dirty_data: bool = True
    ) -> pd.DataFrame:
        """
        Generates realistic e-commerce transactions.
        If include_dirty_data is True, injects realistic flaws (nulls, cancelled invoices 'C...',
        negative quantities, zero/extreme price errors, duplicate rows) for pipeline cleaning validation.
        """
        logger.info(f"Generating synthetic real-world e-commerce dataset for {days} days...")
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)

        records: List[Dict[str, Any]] = []
        customer_pool = [f"CUST-{random.randint(1000, 3500)}" for _ in range(500)]

        # Specific anomaly dates for realistic case studies
        outage_start_day = days - 25  # Day when payment gateway crashed (causes sudden sales drop)
        outage_end_day = days - 22

        for day_offset in range(days):
            current_date = start_date + timedelta(days=day_offset)
            dow = current_date.weekday()
            is_weekend = dow in (5, 6)
            day_multiplier = 1.35 if is_weekend else 1.0

            # Simulate sudden platform payment gateway outage on days 22-25 ago
            is_outage_period = (outage_start_day <= day_offset <= outage_end_day)

            for prod in PRODUCTS:
                # Base daily demand with Poisson variance + day of week multiplier
                mean_demand = prod["base_demand"] * day_multiplier
                if is_outage_period and prod["category"] == "Electronics":
                    # Severe sales drop anomaly (80% drop)
                    mean_demand *= 0.18

                # Promo day simulation (every ~14 days)
                is_promo_day = (day_offset % 14 == 0)
                unit_price = prod["base_price"] * (0.85 if is_promo_day else 1.0)
                if is_promo_day:
                    mean_demand *= 1.6

                daily_order_count = max(2, int(np.random.poisson(mean_demand / 3)))

                for _ in range(daily_order_count):
                    quantity = max(1, int(np.random.geometric(p=0.4)))
                    # Assign random hour throughout the day
                    order_time = current_date + timedelta(
                        hours=random.randint(8, 22),
                        minutes=random.randint(0, 59),
                        seconds=random.randint(0, 59)
                    )

                    invoice_no = f"INV-{uuid.uuid4().hex[:8].upper()}"
                    cust_id = random.choice(customer_pool)
                    country = random.choice(COUNTRIES)

                    records.append({
                        "invoice_no": invoice_no,
                        "stock_code": prod["product_id"],
                        "description": prod["name"],
                        "quantity": float(quantity),
                        "invoice_date": order_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "unit_price": float(round(unit_price, 2)),
                        "customer_id": cust_id,
                        "country": country,
                        "category": prod["category"]
                    })

        df = pd.DataFrame(records)
        logger.info(f"Generated {len(df)} base clean transactions.")

        if include_dirty_data:
            # 1. Inject cancellations (C-prefix invoices with negative quantities) ~3%
            cancel_indices = df.sample(frac=0.035, random_state=42).index
            df.loc[cancel_indices, "invoice_no"] = "C" + df.loc[cancel_indices, "invoice_no"]
            df.loc[cancel_indices, "quantity"] = -1 * df.loc[cancel_indices, "quantity"]

            # 2. Inject missing customer IDs ~4%
            null_cust_indices = df.sample(frac=0.04, random_state=43).index
            df.loc[null_cust_indices, "customer_id"] = np.nan

            # 3. Inject missing descriptions ~1.5%
            null_desc_indices = df.sample(frac=0.015, random_state=44).index
            df.loc[null_desc_indices, "description"] = None

            # 4. Inject unit price bugs ($0.00 or extreme typo $49999.00) ~0.5%
            zero_price_indices = df.sample(n=12, random_state=45).index
            df.loc[zero_price_indices, "unit_price"] = 0.0

            extreme_price_indices = df.sample(n=6, random_state=46).index
            df.loc[extreme_price_indices, "unit_price"] = 49999.00

            # 5. Inject fraudulent transaction bursts (suspicious high quantity / bot attacks)
            fraud_indices = df.sample(n=15, random_state=47).index
            df.loc[fraud_indices, "quantity"] = [random.randint(180, 500) for _ in range(15)]
            df.loc[fraud_indices, "customer_id"] = "CUST-FRAUD-BOT"

            # 6. Inject duplicate rows ~1%
            duplicate_rows = df.sample(frac=0.01, random_state=48)
            df = pd.concat([df, duplicate_rows], ignore_index=True)

            logger.info(f"Injected real-world dataset dirtiness. Total raw records: {len(df)}")

        return df

    @staticmethod
    def generate_single_live_transaction(is_anomaly: bool = False) -> Dict[str, Any]:
        """Generates a single live real-time transaction for streaming simulation."""
        prod = random.choice(PRODUCTS)
        order_time = datetime.now()
        invoice_no = f"INV-{uuid.uuid4().hex[:8].upper()}"

        if is_anomaly:
            # Generate anomalous transaction: abnormal quantity or extreme value
            quantity = random.randint(250, 750)
            unit_price = prod["base_price"] * random.choice([1.0, 3.5])
            cust_id = "CUST-SUSPECT-BOT"
        else:
            quantity = max(1, int(np.random.geometric(p=0.45)))
            unit_price = prod["base_price"]
            cust_id = f"CUST-{random.randint(1000, 3500)}"

        return {
            "invoice_no": invoice_no,
            "stock_code": prod["product_id"],
            "description": prod["name"],
            "quantity": float(quantity),
            "invoice_date": order_time.strftime("%Y-%m-%d %H:%M:%S"),
            "unit_price": float(round(unit_price, 2)),
            "customer_id": cust_id,
            "country": random.choice(COUNTRIES),
            "category": prod["category"]
        }
