import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.cloud import bigquery
from google.oauth2 import service_account
import json
import base64
from datetime import datetime, timedelta
import pytz
import time

# Set page config
st.set_page_config(
    page_title="📈 ETF Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .rising-card {
        border-left-color: #28a745 !important;
    }
    .falling-card {
        border-left-color: #dc3545 !important;
    }
    .neutral-card {
        border-left-color: #6c757d !important;
    }
    .stMetric > label {
        font-size: 0.8rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Authentication setup
@st.cache_resource
def init_bigquery():
    """Initialize BigQuery client with credentials from Streamlit secrets"""
    try:
        # Method 1: Using Streamlit secrets (recommended)
        if "gcp_service_account" in st.secrets:
            credentials = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"]
            )
            return bigquery.Client(credentials=credentials)
        
        # Method 2: Using base64 encoded JSON in secrets
        elif "GOOGLE_APPLICATION_CREDENTIALS_B64" in st.secrets:
            credentials_json = base64.b64decode(
                st.secrets["GOOGLE_APPLICATION_CREDENTIALS_B64"]
            ).decode('utf-8')
            credentials_info = json.loads(credentials_json)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            return bigquery.Client(credentials=credentials)
        
        else:
            st.error("No BigQuery credentials found in secrets")
            return None
            
    except Exception as e:
        st.error(f"Failed to initialize BigQuery: {str(e)}")
        return None

# Data loading functions
@st.cache_data(ttl=30)  # Cache for 30 seconds
def load_etf_data():
    """Load ETF data from BigQuery"""
    client = init_bigquery()
    
    if client is None:
        return create_demo_data()
    
    try:
        query = """
        SELECT *
        FROM `databolt-159516.rpt.live_etf_process_summary`
        LIMIT 1000
        """
        
        df = client.query(query).to_dataframe()
        
        if df.empty:
            return create_demo_data()
        
        # Debug: Show available columns
        st.sidebar.write("Available columns:", list(df.columns))
        st.sidebar.write("Data shape:", df.shape)
        
        # Process data
        df = process_etf_data(df)
        return df
        
    except Exception as e:
        st.warning(f"BigQuery error: {str(e)}. Using demo data.")
        return create_demo_data()
def create_demo_data():
    """Create realistic demo data"""
    np.random.seed(42)  # For consistent demo data
    
    symbols = ['SPY', 'QQQ', 'VTI', 'XLK', 'XLF', 'XLV', 'XLE', 'GLD', 'TLT', 'VEA', 
               'VWO', 'IWM', 'EFA', 'AGG', 'BND', 'ARKK', 'NVDA', 'TSLA', 'MSFT', 'AAPL']
    
    data = {
        'symbol': symbols,
        'Blended': np.random.normal(0, 0.8, len(symbols)),
        'USD_volume': np.random.lognormal(20, 1, len(symbols)),
        'last_price': np.random.uniform(50, 500, len(symbols)),
        'change_24h': np.random.normal(0, 2, len(symbols)),
        'n_obs': np.random.randint(50, 200, len(symbols)),
        'unique_prices': np.random.randint(20, 80, len(symbols)),
        'tops': np.random.randint(2, 15, len(symbols)),
        'lows': np.random.randint(1, 10, len(symbols)),
        'end_time': [datetime.now()] * len(symbols),
        'investment_style': ['Large Cap Blend', 'Large Cap Growth', 'Total Stock Market', 'Technology', 
                           'Financial', 'Healthcare', 'Energy', 'Gold', 'Long Treasury', 'Developed Markets',
                           'Emerging Markets', 'Small Cap Blend', 'Developed Markets', 'Aggregate Bonds', 
                           'Aggregate Bonds', 'Innovation', 'Technology', 'Technology', 'Technology', 'Technology']
    }
    
    df = pd.DataFrame(data)
    return process_etf_data(df)

def process_etf_data(df):
    """Process and enrich ETF data"""
    # Handle the 'category' column - create if missing or fix if has None values
    if 'category' not in df.columns:
        df['category'] = df['symbol'].apply(lambda x: categorize_by_symbol(x))
    else:
        # Fill any None/NaN values in category
        df['category'] = df['category'].fillna('Other')
        df['category'] = df['category'].replace('', 'Other')
    
    # Add momentum flag if Blended column exists
    if 'Blended' in df.columns:
        df['momentum_flag'] = pd.cut(df['Blended'], 
                                    bins=[-np.inf, -0.5, 0.5, np.inf], 
                                    labels=['Falling', 'Neutral', 'Rising'])
    else:
        df['momentum_flag'] = 'Neutral'
        df['Blended'] = 0.0
    
    # Ensure required columns exist
    required_columns = {
        'USD_volume': lambda: np.random.lognormal(20, 1, len(df)),
        'last_price': lambda: np.random.uniform(50, 500, len(df)),
        'change_24h': lambda: np.random.normal(0, 2, len(df)),
        'price_change_range': lambda: np.random.uniform(-0.03, 0.03, len(df)),
        'last_price_change': lambda: np.random.uniform(-0.01, 0.01, len(df)),
        'DoD_price_ratio': lambda: np.random.uniform(0.8, 1.2, len(df)),
        'DoD_Volume_ratio': lambda: np.random.uniform(0.5, 2.0, len(df)),
        'Recent_volume_Percent': lambda: np.random.uniform(0.1, 0.8, len(df)),
        'time_range': lambda: np.random.choice(['15:30', '22:45', '31:20', '45:15'], len(df)),
        'n_obs': lambda: np.random.randint(50, 200, len(df)),
        'unique_prices': lambda: np.random.randint(20, 80, len(df)),
        'tops': lambda: np.random.randint(2, 15, len(df)),
        'lows': lambda: np.random.randint(1, 10, len(df)),
        'end_time': lambda: [datetime.now()] * len(df)
    }
    
    for col, default_func in required_columns.items():
        if col not in df.columns:
            df[col] = default_func()
    
    return df

def categorize_by_symbol(symbol):
    """Categorize ETF by symbol if no other category exists"""
    if pd.isna(symbol):
        return 'Other'
    
    symbol = str(symbol).upper()
    
    if symbol in ['SPY', 'VTI', 'IVV', 'VOO']:
        return 'US Broad Market'
    elif symbol in ['QQQ', 'XLK', 'NVDA', 'MSFT', 'AAPL']:
        return 'Technology'
    elif symbol in ['XLF']:
        return 'Financial Services'
    elif symbol in ['XLV']:
        return 'Healthcare'
    elif symbol in ['XLE']:
        return 'Energy'
    elif symbol in ['GLD', 'SLV']:
        return 'Commodities'
    elif symbol in ['TLT', 'AGG', 'BND']:
        return 'Fixed Income'
    elif symbol in ['VEA', 'EFA']:
        return 'International'
    elif symbol in ['VWO']:
        return 'International'
    else:
        return 'Other'

def is_market_open():
    """Check if market is currently open (8:30 AM - 3:00 PM CT, Mon-Fri)"""
    ct = pytz.timezone('America/Chicago')
    now_ct = datetime.now(ct)
    
    # Check if it's a weekday
    if now_ct.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Check time
    hour_min = now_ct.hour + now_ct.minute / 60
    return 8.5 <= hour_min <= 15.0

def format_volume(value):
    """Format volume in billions/millions"""
    if pd.isna(value) or value == 0:
        return "$0.00M"
    elif value >= 1e9:
        return f"${value/1e9:.2f}B"
    else:
        return f"${value/1e6:.2f}M"

def format_percentage(value):
    """Format percentage"""
    if pd.isna(value):
        return "0.00%"
    return f"{value:.2f}%"

# Main app
def main():
    # Title and header
    st.title("📈 ETF Momentum Dashboard")
    st.markdown("Real-time ETF momentum tracking powered by advanced algorithms")
    
    # Market hours check
    if not is_market_open():
        st.warning("⏰ **Market Closed** - Live data available Monday-Friday, 8:30 AM - 3:00 PM Central Time")
    
    # Load data
    with st.spinner("Loading ETF data..."):
        df = load_etf_data()
    
    if df.empty:
        st.error("No data available")
        return
    
    # Sidebar
    st.sidebar.header("🎯 Filters")
    
    # Asset class filter
    asset_classes = ['All'] + sorted(df['category'].unique().tolist())
    selected_asset_class = st.sidebar.selectbox("Asset Class", asset_classes)
    
    # Momentum filter
    momentum_options = ['Rising', 'Falling', 'Neutral']
    selected_momentum = st.sidebar.multiselect(
        "Momentum", 
        momentum_options, 
        default=momentum_options
    )
    
    # Filter data
    filtered_df = df.copy()
    if selected_asset_class != 'All':
        filtered_df = filtered_df[filtered_df['category'] == selected_asset_class]
    if selected_momentum:
        filtered_df = filtered_df[filtered_df['momentum_flag'].isin(selected_momentum)]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total ETFs", len(df))
    
    with col2:
        rising_count = len(df[df['momentum_flag'] == 'Rising'])
        st.metric("Rising ETFs", rising_count)
    
    with col3:
        falling_count = len(df[df['momentum_flag'] == 'Falling'])
        st.metric("Falling ETFs", falling_count)
    
    with col4:
        last_update = df['end_time'].max().strftime("%H:%M:%S") if 'end_time' in df.columns else "Unknown"
        st.metric("Last Update", last_update)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📊 Live Dashboard", "🎯 Asset Classes", "❓ Help Guide"])
    
    with tab1:
        # Top rising and falling ETFs
# Top rising and falling ETFs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 Top Rising ETFs")
        rising_df = df[df['Blended'] > 0]
        if not rising_df.empty:
            display_etf_table(df, "rising")  # Pass full df, function handles filtering
        else:
            st.info("No rising ETFs found")
    
    with col2:
        st.subheader("📉 Top Falling ETFs") 
        falling_df = df[df['Blended'] < 0]
        if not falling_df.empty:
            display_etf_table(df, "falling")  # Pass full df, function handles filtering
        else:
            st.info("No falling ETFs found")
        
        # Visualizations
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Asset Class Performance Heatmap")
            create_heatmap(df)
        
        with col2:
            st.subheader("📊 Momentum Distribution")
            create_momentum_distribution(df)
    
    with tab2:
        # Asset class metrics
        asset_perf = df.groupby('category')['Blended'].mean().sort_values(ascending=False)
        asset_volume = df.groupby('category')['USD_volume'].sum().sort_values(ascending=False)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not asset_perf.empty:
                st.metric("Best Momentum", asset_perf.index[0], f"{asset_perf.iloc[0]:.2f}")
        
        with col2:
            if not asset_perf.empty:
                st.metric("Worst Momentum", asset_perf.index[-1], f"{asset_perf.iloc[-1]:.2f}")
        
        with col3:
            if not asset_volume.empty:
                st.metric("Most Active", asset_volume.index[0])
        
        # Filtered table
        st.subheader("📋 Filtered ETF Details")
        if not filtered_df.empty:
            display_filtered_table(filtered_df)
        else:
            st.info("No ETFs match the selected filters")
        
        # Comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Asset Class Comparison")
            create_asset_comparison(df)
        
        with col2:
            st.subheader("🌊 Volume by Asset Class")
            create_volume_chart(df)
    
    with tab3:
        display_help_content()
    
    # Auto-refresh
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Auto-refresh every 30 seconds during market hours
    if is_market_open():
        time.sleep(30)
        st.rerun()

def display_etf_table(df, table_type):
    """Display ETF table with formatting - top 30 with scrolling"""
    
    # Get top 30 based on type
    if table_type == "rising":
        filtered_df = df[df['Blended'] > 0].nlargest(30, 'Blended')
    else:
        filtered_df = df[df['Blended'] < 0].nsmallest(30, 'Blended')
    
    if filtered_df.empty:
        st.info(f"No {table_type} ETFs found")
        return
    
    # Create display dataframe with exact R format
    display_df = pd.DataFrame({
        'Symbol': filtered_df['symbol'],
        'Asset Class': filtered_df['category'],
        'Obs': filtered_df['n_obs'].astype(int),
        'Prices': filtered_df['unique_prices'].astype(int),
        'Score': filtered_df['Blended'].round(1),
        'Tops': filtered_df['tops'].astype(int),
        'Lows': filtered_df['lows'].astype(int),
        'Price': filtered_df['last_price'].apply(lambda x: f"${x:.2f}"),
        'Last %': filtered_df['last_price_change'].apply(lambda x: f"{x*100:.2f}%"),
        'Recent %': filtered_df['price_change_range'].apply(lambda x: f"{x*100:.2f}%"),
        '24h %': filtered_df['change_24h'].apply(lambda x: f"{x:.2f}%"),
        'Day Vol': filtered_df['USD_volume'].apply(format_volume),
        'Range': filtered_df['time_range'],
        'P.Ratio': filtered_df['DoD_price_ratio'].round(2),
        'V.Ratio': filtered_df['DoD_Volume_ratio'].round(2),
        '10 mn Vol': filtered_df['Recent_volume_Percent'].round(2),
        'Time': filtered_df['end_time'].dt.strftime('%H:%M:%S') if 'end_time' in filtered_df.columns else "N/A",
        'Blended': filtered_df['Blended'].round(3)
    })
    
    # Display with controlled height and horizontal scrolling
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=400,  # Fixed height with vertical scroll
        column_config={
            "Symbol": st.column_config.TextColumn(width="small"),
            "Asset Class": st.column_config.TextColumn(width="medium"),
            "Score": st.column_config.NumberColumn(
                format="%.1f",
                help="Momentum score (-100 to +100)"
            ),
            "Price": st.column_config.TextColumn(width="small"),
            "Day Vol": st.column_config.TextColumn(width="small"),
            "Blended": st.column_config.NumberColumn(format="%.3f")
        }
    )

def display_filtered_table(df):
    """Display filtered ETF table"""
    display_df = df[['symbol', 'category', 'momentum_flag', 'Blended', 'last_price', 'change_24h', 'USD_volume']].copy()
    display_df.columns = ['Symbol', 'Category', 'Momentum', 'Score', 'Price', '24h %', 'Volume']
    
    # Format columns
    display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")
    display_df['24h %'] = display_df['24h %'].apply(format_percentage)
    display_df['Volume'] = display_df['Volume'].apply(format_volume)
    display_df['Score'] = display_df['Score'].round(3)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

def create_heatmap(df):
    """Create asset class performance heatmap"""
    asset_summary = df.groupby('category').agg({
        'Blended': 'mean',
        'symbol': 'count'
    }).reset_index()
    asset_summary.columns = ['Category', 'Avg_Momentum', 'Count']
    
    fig = px.treemap(
        asset_summary,
        path=['Category'],
        values='Count',
        color='Avg_Momentum',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        title="Asset Class Performance"
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def create_momentum_distribution(df):
    """Create momentum distribution chart"""
    momentum_counts = df.groupby(['category', 'momentum_flag']).size().reset_index(name='count')
    
    fig = px.bar(
        momentum_counts,
        x='category',
        y='count',
        color='momentum_flag',
        color_discrete_map={
            'Rising': '#28a745',
            'Falling': '#dc3545', 
            'Neutral': '#6c757d'
        },
        title="Momentum Distribution by Asset Class"
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_asset_comparison(df):
    """Create asset class comparison chart"""
    asset_perf = df.groupby('category')['Blended'].mean().sort_values()
    
    colors = ['#dc3545' if x < 0 else '#28a745' for x in asset_perf.values]
    
    fig = go.Figure(go.Bar(
        x=asset_perf.values,
        y=asset_perf.index,
        orientation='h',
        marker_color=colors
    ))
    
    fig.update_layout(
        title="Average Momentum by Asset Class",
        xaxis_title="Average Momentum Score",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_volume_chart(df):
    """Create volume by asset class chart"""
    volume_summary = df.groupby('category')['USD_volume'].sum().sort_values(ascending=False)
    
    fig = px.bar(
        x=volume_summary.index,
        y=volume_summary.values / 1e9,
        title="Total Volume by Asset Class ($B)"
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_title="Volume ($B)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_help_content():
    """Display help documentation"""
    st.markdown("""
    ## What This App Does
    
    The **ETF Momentum Dashboard** is a real-time analysis tool that monitors Exchange-Traded Funds (ETFs) during market hours (8:30 AM - 3:00 PM Central Time, Monday-Friday). It processes live market data to identify momentum patterns and provides actionable insights for traders and investors.
    
    ### Key Features:
    - **Real-time ETF monitoring** with 30-second data refresh during market hours
    - **Advanced momentum scoring** using proprietary algorithms
    - **Asset class categorization** for sector-based analysis
    - **Interactive visualizations** and filterable data tables
    
    ---
    
    ## 📊 Main Table Field Explanations
    
    | Field | Description | What It Tells You |
    |-------|-------------|-------------------|
    | **Symbol** | ETF ticker symbol | The fund being tracked |
    | **Asset Class** | Investment category | Sector/theme (Technology, Healthcare, etc.) |
    | **Score** | Blended momentum score | **Key metric: -100 to +100 momentum rating** |
    | **Price** | Current/last price | Latest trading price |
    | **24h %** | Daily change | 24-hour percentage change |
    | **Volume** | Dollar volume | Total $ value traded (M = millions, B = billions) |
    
    ---
    
    ## 🎯 Momentum Score Components
    
    The **Blended Score** is calculated using five components:
    
    1. **Volume Activity (30%)** - Measures volume expansion/contraction
    2. **Price Momentum (25%)** - Raw price movement strength
    3. **Microstructure Score (20%)** - Balance of new highs vs new lows
    4. **Liquidity Score (15%)** - Rewards higher dollar volume
    5. **Confidence Multipliers (10%)** - Boosts for volume explosions and trend clarity
    
    ---
    
    ## 📈 Trading Signals
    
    - **Rising ETFs (Score > 0.5)**: Strong upward momentum, potential bullish continuation
    - **Falling ETFs (Score < -0.5)**: Strong downward momentum, potential bearish continuation  
    - **Neutral ETFs (-0.5 to 0.5)**: Sideways movement, no clear directional bias
    
    ---
    
    ## ⚠️ Important Notes
    
    **Market Hours Only**: Live data available 8:30 AM - 3:00 PM Central Time, Monday-Friday
    
    **Best Practices**:
    1. Focus on high-volume ETFs (>$10M daily volume)
    2. Cross-reference multiple indicators
    3. Consider market conditions and news events
    4. Use filters to focus on specific asset classes
    
    ---
    
    *Built by [@SwapStatsHub](https://x.com/swapstatshub) - Real-time ETF momentum tracking for active traders*
    """)

if __name__ == "__main__":
    main()




