from forecasting import ForecastingEngine
import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import sys
from pathlib import Path

st.set_page_config(page_title="M5 Demand Forecasting", page_icon="", layout="wide")

# BASE PATHS
BASE_PATH = Path(__file__).parent.parent
CLEANED_PATH = BASE_PATH / "data_cleaned"
MODELS_PATH = BASE_PATH / "models1"
CONFIG_PATH = BASE_PATH / "config1"

# Session state defaults
for key in ('models_trained', 'drift_run', 'rules_run'):
    st.session_state.setdefault(key, False)


FEATURES_CACHE = BASE_PATH / "data_cleaned" / "parquet_cache" / "features_cache.parquet"


@st.cache_data
def load_data():
    from run_pipeline import step_clean_and_load, step_features
    from feature_engineering import get_feature_columns

    pos_df, cal, sales, prices = step_clean_and_load(None)

    # Use a parquet cache for feature-engineered data  the groupby/rolling
    # on 4.7M rows is expensive and shouldn't re-run on every page refresh.
    if FEATURES_CACHE.exists():
        import pandas as _pd
        features_df = _pd.read_parquet(FEATURES_CACHE)
    else:
        features_df, _ = step_features(pos_df, None)
        FEATURES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_parquet(FEATURES_CACHE, index=False)

    feature_cols = get_feature_columns()
    return pos_df, cal, sales, prices, features_df, feature_cols


def _train_single_sku(sku, features_df, feature_cols):
    """Train one SKU model  used as a worker in the thread pool."""
    from forecasting import ForecastModel
    sku_df = features_df[features_df['sku'] == sku].copy()

    if len(sku_df) < 50:
        return None
    try:
        import warnings
        warnings.filterwarnings('ignore')
        model = ForecastModel(sku)
        model.train(sku_df, feature_cols, use_time_series_cv=False)
        model.save_model(str(MODELS_PATH / f"{sku}_model.joblib"))
        preds = model.predict(sku_df)
        return {
            'sku': sku,
            'model': model,
            'preds': {
                'actual': sku_df['units_sold'].values,
                'predicted': preds,
                'dates': sku_df['date'].values,
            },
            'metrics': model.metrics,
            'samples': len(sku_df),
        }
    except Exception as e:
        return {'sku': sku, 'error': str(e)}


#  Pages 

def show_data_explorer():
    st.header(" Data Explorer & Features")
    with st.spinner("Loading cleaned M5 datasets..."):
        pos_df, cal, sales, prices, features_df, feature_cols = load_data()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(pos_df):,}")
    col2.metric("Total SKUs", f"{pos_df['sku'].nunique():,}")
    col3.metric("Date Range", f"{pos_df['date'].min().date()} to {pos_df['date'].max().date()}")

    st.subheader("Data Sample")
    st.dataframe(pos_df.head(100), use_container_width=True)

    st.markdown("---")
    st.subheader(" Exploratory Data Analytics")

    c1, c2 = st.columns(2)
    import plotly.express as px
    if 'cat_id' in pos_df.columns:
            cat_counts = pos_df['cat_id'].value_counts().reset_index()
            cat_counts.columns = ['Category', 'Count']
            fig = px.pie(cat_counts, values='Count', names='Category',
                         title="Records by Category", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        dow_sales = pos_df.groupby(pos_df['date'].dt.dayofweek)['units_sold'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        fig2 = px.bar(x=days, y=dow_sales.values,
                      title="Average Units Sold by Day of Week",
                      labels={'x': 'Day', 'y': 'Avg Units'})
        fig2.update_traces(marker_color='#9C27B0')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Total Daily Sales Volume over Time**")
    daily_sales = pos_df.groupby('date')['units_sold'].sum().reset_index()
    fig3 = px.line(daily_sales, x='date', y='units_sold', render_mode='webgl')
    fig3.update_traces(line_color="#2196F3", line_width=1)
    st.plotly_chart(fig3, use_container_width=True)


def show_model_training():
    st.header(" Model Training")
    with st.spinner("Loading features..."):
        pos_df, cal, sales, prices, features_df, feature_cols = load_data()

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("""
        **Train XGBoost Models**
        Configure the number of SKUs you want to process. Training uses
        multicore parallelization. Note: navigating away during active
        training will interrupt the thread.
        """)
    with c2:
        sku_limit = st.number_input(
            "SKUs to Train", min_value=1,
            max_value=int(pos_df['sku'].nunique()), value=50)

    st.write("")
    if st.session_state['models_trained']:
        st.success(" Models have already been trained for this session!")
        run_training = st.button(" Force Override & Retrain", type="secondary")
    else:
        run_training = st.button(" Start Training Sweep", type="primary")

    if run_training:
        st.warning(" Training in progress. Please do not navigate to other tabs until complete.")
        MODELS_PATH.mkdir(parents=True, exist_ok=True)
        from forecasting import ForecastingEngine
        engine = ForecastingEngine(str(MODELS_PATH))

        skus = features_df['sku'].unique()[:sku_limit]

        import concurrent.futures
        progress_bar = st.progress(0, text="Initializing training sweep...")
        all_predictions = {}
        training_results = []
        max_workers = os.cpu_count() or 4
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_train_single_sku, sku, features_df, feature_cols): sku
                for sku in skus
            }
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                progress_bar.progress(completed / len(skus),
                                      text=f"Trained {completed}/{len(skus)} SKUs")
                res = future.result()
                if res is None or 'error' in res:
                    continue

                sku = res['sku']
                engine.models[sku] = res['model']
                all_predictions[sku] = res['preds']
                training_results.append({
                    'sku': sku,
                    'mae': res['metrics'].get('mae', 0),
                    'rmse': res['metrics'].get('rmse', 0),
                    'r2': res['metrics'].get('r2', 0),
                    'samples': res['samples'],
                })

        elapsed = time.time() - start_time
        st.success(f"Trained {len(training_results)} models in {elapsed:.1f} seconds!")

        st.session_state['training_results'] = pd.DataFrame(training_results)
        st.session_state['all_predictions'] = all_predictions
        st.session_state['models_trained'] = True

    # Show results if training has been done
    if st.session_state['models_trained']:
        st.markdown("---")
        st.subheader("Training Metrics Summary")
        tr_df = st.session_state['training_results']

        m1, m2, m3 = st.columns(3)
        m1.metric("Average MAE", f"{tr_df['mae'].mean():.2f}")
        m2.metric("Average RMSE", f"{tr_df['rmse'].mean():.2f}")
        m3.metric("Average R", f"{tr_df['r2'].mean():.3f}")

        with st.expander("View Full Metrics Table"):
            st.dataframe(tr_df, use_container_width=True)

        st.markdown("---")
        st.subheader(" Live SKU Forecasting Interactive Graph")
        all_preds = st.session_state['all_predictions']
        trained_skus = list(all_preds.keys())

        if trained_skus:
            selected_sku = st.selectbox(
                "Select a SKU to visualize its forecast vs actuals:", trained_skus)

            import plotly.graph_objects as go
            data = all_preds[selected_sku]
            dates = data['dates'][-90:]
            actual = data['actual'][-90:]
            predicted = data['predicted'][-90:]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=actual, mode='lines', name='Actual Sales',
                line=dict(color='#2196F3', width=2)))
            fig.add_trace(go.Scatter(
                x=dates, y=predicted, mode='lines', name='XGBoost Forecast',
                line=dict(color='#FF9800', width=2, dash='dot')))
            fig.update_layout(
                title=f"90-Day Forecast History for {selected_sku}",
                xaxis_title="Date", yaxis_title="Units",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)


def show_drift_detection():
    st.header(" Drift Detection & Diagnostics")

    if not st.session_state['models_trained']:
        st.warning("Please run Model Training first to generate predictions for drift detection.")
        return

    if st.session_state.get('drift_run'):
        st.success(" Drift Diagnostics already generated for this session.")
        run_drift = st.button(" Force Re-run Drift Analysis", type="secondary")
    else:
        run_drift = st.button(" Run Drift Analysis", type="primary")

    if run_drift:
        all_predictions = st.session_state['all_predictions']
        from drift_detection import DriftMonitor
        monitor = DriftMonitor()
        results_summary = []
        progress_bar = st.progress(0, text="Analyzing residuals...")

        skus = list(all_predictions.keys())
        for i, sku in enumerate(skus):
            data = all_predictions[sku]
            actual = data['actual']
            predicted = data['predicted']

            monitor.get_detector(sku)
            last_result = None
            for a, p in zip(actual[-100:], predicted[-100:]):
                last_result = monitor.update(sku, float(a), float(p))

            if last_result is None:
                continue

            residuals = actual[-100:] - predicted[-100:]
            results_summary.append({
                'sku': sku,
                'drift': last_result.get('drift_detected', False),
                'severity': last_result.get('severity', 'none'),
                'mean_residual': np.mean(residuals),
                'std_residual': np.std(residuals),
                'detectors': ", ".join(last_result.get('detectors_triggered', [])) or "",
            })
            if i % 10 == 0:
                progress_bar.progress(i / len(skus), text=f"Analyzed {i}/{len(skus)} SKUs")

        progress_bar.progress(1.0, text="Analysis Complete!")
        st.session_state['drift_results'] = pd.DataFrame(results_summary)
        st.session_state['drift_run'] = True

    if st.session_state.get('drift_run'):
        dr_df = st.session_state['drift_results']
        drift_count = dr_df['drift'].sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Drift Detected", f"{drift_count} SKUs", delta_color="inverse")
        c2.metric("Average Mean Residual", f"{dr_df['mean_residual'].mean():+.2f}")
        c3.metric("Average Std Residual", f"{dr_df['std_residual'].mean():.2f}")

        st.subheader("Detailed Drift Log")
        st.dataframe(
            dr_df.style.apply(
                lambda x: ['background: #ffebee' if v else '' for v in x],
                subset=['drift']),
            use_container_width=True)

        if drift_count > 0:
            st.markdown("---")
            if st.button(" Improve Drifted Models (Retrain)", type="primary"):
                st.info(f"Retraining {drift_count} models to resolve drift...")
                from forecasting import ForecastModel
                features_df = load_data()[4]
                feature_cols = load_data()[5]
                drifted_skus = dr_df[dr_df['drift']]['sku'].tolist()

                progress = st.progress(0, text="Retraining models...")
                for idx, sku in enumerate(drifted_skus):
                    sku_df = features_df[features_df['sku'] == sku].copy()
                    model_path = MODELS_PATH / f"{sku}_model.joblib"
                    try:
                        # Load existing, and incremental train
                        model = ForecastModel.load_model(str(model_path))
                        # We use simple train here, incremental logic would be if we had new data
                        model.train(sku_df, feature_cols, use_time_series_cv=False)
                        model.save_model(str(model_path))
                        
                        # Update session state predictions so that re-running drift is fresh
                        st.session_state['all_predictions'][sku]['predicted'] = model.predict(sku_df)
                        progress.progress((idx + 1) / len(drifted_skus), text=f"Retrained {sku}")
                    except Exception as e:
                        st.error(f"Failed to retrain {sku}: {str(e)}")
                
                st.success(" Drifted models have been retrained and improved.")
                st.session_state['drift_run'] = False # require re-running the diagnostic
                st.rerun()


def show_business_rules():
    st.header(" Business Rules Application")
    with st.spinner("Loading data for rules engine..."):
        pos_df, cal, sales, prices, features_df, feature_cols = load_data()

    st.markdown("### 1. Configure Supply Chain Constraints")
    c1, c2 = st.columns([1, 2])
    with c1:
        global_budget = st.number_input(
            "Global Purchasing Budget ($)", min_value=0, value=25000, step=1000)
    with c2:
        st.markdown("**Editable SKU Configurations:** Modify cost, price, and stock below.")

    if st.session_state['models_trained']:
        skus = list(st.session_state['all_predictions'].keys())
    else:
        skus = pos_df['sku'].unique()[:50]
        st.warning("Training wasn't executed. Using a 50 SKU sample for the Business Rules.")

    # Build or reuse the editable SKU config table
    if 'sku_config_df' not in st.session_state or len(st.session_state['sku_config_df']) != len(skus):
        np.random.seed(42)
        base_cost = np.random.uniform(5, 50, len(skus)).round(2)
        st.session_state['sku_config_df'] = pd.DataFrame({
            'sku': skus,
            'unit_cost': base_cost,
            'selling_price': (base_cost * np.random.uniform(1.2, 2.5, len(skus))).round(2),
            'current_stock': np.random.randint(10, 100, len(skus)),
        })

    edited_config = st.data_editor(
        st.session_state['sku_config_df'],
        use_container_width=True, hide_index=True,
        column_config={
            "sku": st.column_config.TextColumn("SKU", disabled=True),
            "unit_cost": st.column_config.NumberColumn("Unit Cost ($)", min_value=0.01, format="$%.2f"),
            "selling_price": st.column_config.NumberColumn("Selling Price ($)", min_value=0.01, format="$%.2f"),
            "current_stock": st.column_config.NumberColumn("Current Stock", min_value=0, step=1),
        })
    st.session_state['sku_config_df'] = edited_config

    st.markdown("### 2. Execute Supply Chain Logic")
    if st.session_state.get('rules_run'):
        st.success(" Supply chain rules already applied!")
        run_rules = st.button(" Re-apply Supply Chain Rules", type="secondary")
    else:
        run_rules = st.button(" Apply Supply Chain Rules", type="primary")

    if run_rules:
        import yaml

        from run_pipeline import generate_business_rules
        from business_rules import BusinessRuleEngine

        CONFIG_PATH.mkdir(parents=True, exist_ok=True)
        rules_path = CONFIG_PATH / "business_rules.yml"

        np.random.seed(42)
        rules = generate_business_rules(pos_df)
        with open(rules_path, 'w') as f:
            yaml.dump(rules, f, default_flow_style=False)

        engine = BusinessRuleEngine(str(rules_path))
        grouped = pos_df.groupby('sku')

        # Compute weekly forecasts from recent 14-day averages
        forecasts = {}
        for sku in skus:
            recent_avg = grouped.get_group(sku).tail(14)['units_sold'].mean()
            forecasts[sku] = int(recent_avg * 7)

        # Step 1: Apply per-SKU perishability / capacity rules
        unconstrained = []
        for sku in skus:
            forecast = forecasts.get(sku, 0)
            cfg = edited_config[edited_config['sku'] == sku].iloc[0]
            stock = int(cfg['current_stock'])
            cost = float(cfg['unit_cost'])
            price = float(cfg['selling_price'])
            daily_demand = grouped.get_group(sku)['units_sold'].mean()

            res = engine.apply_rules(
                forecast_qty=float(forecast),
                current_stock=float(stock),
                sku=sku,
                daily_demand=float(daily_demand))

            unconstrained.append({
                'sku': sku, 'forecast': forecast, 'stock': stock,
                'base_rule_order': res['final_qty'],
                'unit_cost': cost, 'margin': price - cost,
                'total_cost': res['final_qty'] * cost,
                'rule_applied': res['explanation'],
            })

        df_unc = pd.DataFrame(unconstrained).sort_values('margin', ascending=False)

        # Step 2: Global budget optimisation (highest margin first)
        final_orders = []
        spent = 0.0
        for _, row in df_unc.iterrows():
            qty = row['base_rule_order']
            cost = row['unit_cost']
            affordable = int((global_budget - spent) // cost)
            final_qty = min(qty, affordable)

            rule_note = row['rule_applied']
            if final_qty < qty:
                rule_note += f" |  Reduced from {qty} to {final_qty} due to Budget Limit."

            order_cost = final_qty * cost
            spent += order_cost
            final_orders.append({
                'sku': row['sku'], 'forecast': row['forecast'],
                'stock': row['stock'], 'margin': row['margin'],
                'final_order_qty': final_qty,
                'total_sku_cost': order_cost,
                'rule_applied': rule_note,
            })

        total_forecast = df_unc['forecast'].sum()
        total_order = sum(r['final_order_qty'] for r in final_orders)
        reduction = ((total_forecast - total_order) / total_forecast * 100) if total_forecast > 0 else 0

        st.session_state.update({
            'rules_results': final_orders,
            'rules_reduction': reduction,
            'rules_total_forecast': total_forecast,
            'rules_total_order': total_order,
            'rules_total_spent': spent,
            'global_budget': global_budget,
            'rules_run': True,
        })

    if st.session_state.get('rules_run'):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Unconstrained Forecast", f"{st.session_state['rules_total_forecast']:,}")
        c2.metric("Budget-Constrained Order", f"{st.session_state['rules_total_order']:,}")
        c3.metric("Total Order Cost", f"${st.session_state['rules_total_spent']:,.2f}")
        remaining = st.session_state['global_budget'] - st.session_state['rules_total_spent']
        c4.metric("Remaining Budget", f"${remaining:,.2f}")

        st.subheader("Financial & Supply Chain Order Log")
        st.dataframe(pd.DataFrame(st.session_state['rules_results']),
                     use_container_width=True)


#  Navigation 

def main():
    st.sidebar.title(" M5 Pipeline")
    page = st.sidebar.radio(
        "Navigation",
        ["Data & Features", "Model Training", "Drift & Diagnostics", "Business Rules"])

    pages = {
        "Data & Features": show_data_explorer,
        "Model Training": show_model_training,
        "Drift & Diagnostics": show_drift_detection,
        "Business Rules": show_business_rules,
    }
    pages[page]()


if __name__ == "__main__":
    main()
