"""
Business Rules Module
Applies capacity, budget, and perishability constraints to forecasts.
"""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BusinessRuleEngine:
    """Engine for applying business constraints to demand forecasts."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.rules: Dict[str, Dict] = {}
        if config_path:
            self.load_rules(config_path)
    
    def load_rules(self, config_path: str):
        """Load business rules from YAML config."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        for sku_config in config.get('skus', []):
            sku = sku_config['sku']
            self.rules[sku] = {
                'max_shelf_capacity': sku_config.get('max_shelf_capacity', 100),
                'unit_cost': sku_config.get('unit_cost', 1.0),
                'max_budget_per_order': sku_config.get('max_budget_per_order', 500),
                'perishability_days': sku_config.get('perishability_days', 30),
                'safety_stock_days': sku_config.get('safety_stock_days', 2)
            }
        
        logger.info(f"Loaded rules for {len(self.rules)} SKUs")
    
    def save_rules(self, config_path: str):
        """Save current rules to YAML config."""
        config = {'skus': []}
        for sku, rules in self.rules.items():
            config['skus'].append({'sku': sku, **rules})
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Saved rules to {config_path}")
    
    def update_rule(self, sku: str, field: str, value):
        """Update a specific rule for a SKU."""
        if sku in self.rules:
            self.rules[sku][field] = value
            logger.info(f"Updated {sku}.{field} = {value}")
    
    def apply_rules(self, forecast_qty: float, current_stock: float, 
                    sku: str, daily_demand: Optional[float] = None) -> Dict:
        """
        Apply business constraints to compute final order quantity.
        
        Args:
            forecast_qty: Raw demand forecast
            current_stock: Current inventory level
            sku: SKU identifier
            daily_demand: Average daily demand (for perishability calc)
            
        Returns:
            Dict with final_qty and explanation
        """
        if sku not in self.rules:
            return {
                'final_qty': max(0, forecast_qty - current_stock),
                'explanation': 'No rules configured, using raw forecast minus stock'
            }
        
        rules = self.rules[sku]
        explanations = []
        
        # Step 1: Required = forecast - stock (min 0)
        required = max(0, forecast_qty - current_stock)
        order_qty = required
        
        # Step 2: Capacity constraint
        capacity_limit = rules['max_shelf_capacity'] - current_stock
        if order_qty > capacity_limit:
            order_qty = max(0, capacity_limit)
            explanations.append(f"Capped by shelf capacity ({rules['max_shelf_capacity']})")
        
        # Step 3: Budget constraint
        budget_limit = rules['max_budget_per_order'] / rules['unit_cost']
        if order_qty > budget_limit:
            order_qty = int(budget_limit)
            explanations.append(f"Capped by budget (${rules['max_budget_per_order']})")
        
        # Step 4: Perishability constraint
        if daily_demand and daily_demand > 0:
            perish_limit = daily_demand * rules['perishability_days']
            if order_qty > perish_limit:
                order_qty = int(perish_limit)
                explanations.append(f"Capped by perishability ({rules['perishability_days']} days)")
        
        explanation = '; '.join(explanations) if explanations else 'Within all constraints'
        
        return {
            'sku': sku,
            'forecast_qty': forecast_qty,
            'current_stock': current_stock,
            'final_qty': int(order_qty),
            'explanation': explanation
        }
    
    def get_rules_df(self) -> pd.DataFrame:
        """Get all rules as a DataFrame for display."""
        rows = []
        for sku, rules in self.rules.items():
            rows.append({'sku': sku, **rules})
        return pd.DataFrame(rows)


def apply_business_rules(forecast_df: pd.DataFrame, stock_df: pd.DataFrame, 
                        rules_config: str) -> pd.DataFrame:
    """
    Apply business rules to forecast DataFrame.
    
    Args:
        forecast_df: DataFrame with columns [sku, forecast]
        stock_df: DataFrame with columns [sku, current_stock]
        rules_config: Path to business rules YAML
        
    Returns:
        DataFrame with [sku, forecast_qty, final_order_qty, rule_explanation]
    """
    engine = BusinessRuleEngine(rules_config)
    
    # Merge forecast with stock
    merged = forecast_df.merge(stock_df, on='sku', how='left')
    merged['current_stock'] = merged['current_stock'].fillna(0)
    
    results = []
    for _, row in merged.iterrows():
        result = engine.apply_rules(
            forecast_qty=row['forecast'],
            current_stock=row['current_stock'],
            sku=row['sku']
        )
        results.append({
            'sku': row['sku'],
            'forecast_qty': result['forecast_qty'],
            'final_order_qty': result['final_qty'],
            'rule_explanation': result['explanation']
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Test business rules
    config_path = Path(__file__).parent.parent / "config" / "business_rules.yml"
    
    engine = BusinessRuleEngine(str(config_path))
    
    # Test applying rules
    result = engine.apply_rules(
        forecast_qty=200,
        current_stock=50,
        sku="MILK_1L",
        daily_demand=45
    )
    
    print(f"\nBusiness Rule Application:")
    print(f"  Forecast: {result['forecast_qty']}")
    print(f"  Final Order: {result['final_qty']}")
    print(f"  Explanation: {result['explanation']}")
    
    print("\nAll Rules:")
    print(engine.get_rules_df())
