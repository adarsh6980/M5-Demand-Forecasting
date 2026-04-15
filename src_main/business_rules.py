"""
Business Rules Module
Applies capacity, budget, and perishability constraints to raw demand forecasts
before they become purchase orders.
"""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BusinessRuleEngine:
    """Applies per-SKU business constraints (shelf capacity, budget, perishability)."""

    def __init__(self, config_path=None):
        self.rules = {}
        if config_path:
            self.load_rules(config_path)

    def load_rules(self, config_path):
        """Load SKU-level rules from a YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        for entry in config.get('skus', []):
            self.rules[entry['sku']] = {
                'max_shelf_capacity': entry.get('max_shelf_capacity', 100),
                'unit_cost': entry.get('unit_cost', 1.0),
                'max_budget_per_order': entry.get('max_budget_per_order', 500),
                'perishability_days': entry.get('perishability_days', 30),
                'safety_stock_days': entry.get('safety_stock_days', 2),
            }
        logger.info(f"Loaded rules for {len(self.rules)} SKUs")

    def save_rules(self, config_path):
        """Persist current rules back to YAML."""
        config = {'skus': [{'sku': sku, **vals} for sku, vals in self.rules.items()]}
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Saved rules to {config_path}")

    def update_rule(self, sku, field, value):
        """Change a single rule field for a SKU."""
        if sku in self.rules:
            self.rules[sku][field] = value
            logger.info(f"Updated {sku}.{field} = {value}")

    def apply_rules(self, forecast_qty, current_stock, sku,
                    daily_demand=None):
        """
        Compute the final order quantity after applying all constraints.

        Returns a dict with 'final_qty' and 'explanation'.
        """
        if sku not in self.rules:
            return {
                'final_qty': max(0, forecast_qty - current_stock),
                'explanation': 'No rules configured, using raw forecast minus stock',
            }

        r = self.rules[sku]
        explanations = []

        # Start from net requirement
        order_qty = max(0, forecast_qty - current_stock)

        # Cap by shelf capacity
        capacity_room = max(0, r['max_shelf_capacity'] - current_stock)
        if order_qty > capacity_room:
            order_qty = capacity_room
            explanations.append(f"Capped by shelf capacity ({r['max_shelf_capacity']})")

        # Cap by per-order budget
        budget_units = r['max_budget_per_order'] / r['unit_cost']
        if order_qty > budget_units:
            order_qty = int(budget_units)
            explanations.append(f"Capped by budget (${r['max_budget_per_order']})")

        # Cap by perishability
        if daily_demand and daily_demand > 0:
            perish_limit = daily_demand * r['perishability_days']
            if order_qty > perish_limit:
                order_qty = int(perish_limit)
                explanations.append(f"Capped by perishability ({r['perishability_days']} days)")

        return {
            'sku': sku,
            'forecast_qty': forecast_qty,
            'current_stock': current_stock,
            'final_qty': int(order_qty),
            'explanation': '; '.join(explanations) or 'Within all constraints',
        }

    def get_rules_df(self):
        """Return all rules as a DataFrame."""
        rows = [{'sku': sku, **vals} for sku, vals in self.rules.items()]
        return pd.DataFrame(rows)


def apply_business_rules(forecast_df, stock_df, rules_config):
    """
    Batch-apply business rules to a forecast DataFrame.

    Expects forecast_df with [sku, forecast] and stock_df with [sku, current_stock].
    Returns a DataFrame with [sku, forecast_qty, final_order_qty, rule_explanation].
    """
    engine = BusinessRuleEngine(rules_config)

    merged = forecast_df.merge(stock_df, on='sku', how='left')
    merged['current_stock'] = merged['current_stock'].fillna(0)

    results = []
    for _, row in merged.iterrows():
        result = engine.apply_rules(
            forecast_qty=row['forecast'],
            current_stock=row['current_stock'],
            sku=row['sku'],
        )
        results.append({
            'sku': row['sku'],
            'forecast_qty': result['forecast_qty'],
            'final_order_qty': result['final_qty'],
            'rule_explanation': result['explanation'],
        })

    return pd.DataFrame(results)
