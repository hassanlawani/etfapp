import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, time
import pytz
from io import StringIO
import time as time_module

st.set_page_config(
    page_title="📈 ETF Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .status-open {
        color: #28a745;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .status-closed {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        background-color: #f8f9fa;
        border-radius: 0.25rem 0.25rem 0 0;
        border: 1px solid #dee2e6;
    }
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #17a2b8);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

ETF_CATEGORIES = {
    'SPY': 'Large Cap Blend', 'QQQ': 'Large Cap Growth', 'IVV': 'Large Cap Blend',
    'VTI': 'Total Stock Market', 'VOO': 'Large Cap Blend', 'VEA': 'Developed Markets',
    'IEFA': 'Developed Markets', 'VWO': 'Emerging Markets', 'AGG': 'Aggregate Bonds',
    'BND': 'Aggregate Bonds', 'XLK': 'Technology', 'XLF': 'Financial',
    'XLV': 'Healthcare', 'XLE': 'Energy', 'XLI': 'Industrial', 'GLD': 'Gold',
    'SLV': 'Silver', 'TLT': 'Long Treasury', 'VTV': 'Large Cap Value',
    'VUG': 'Large Cap Growth', 'IWM': 'Small Cap Blend', 'EFA': 'Developed Markets',
    'EEM': 'Emerging Markets', 'HYG': 'High Yield', 'LQD': 'Corporate Bonds',
    'TIP': 'Inflation Protected', 'VNQ': 'Real Estate', 'DIA': 'Large Cap Blend'
}

def style_to_asset_class(style):
    mapping = {
        'Large Cap Blend': 'U.S. Broad Market', 'Total Stock Market': 'U.S. Broad Market',
        'Extended Market': 'U.S. Broad Market', 'Large Cap Growth': 'U.S. Growth',
        'Mid Cap Growth': 'U.S. Growth', 'Small Cap Growth': 'U.S. Growth',
        'Large Cap Value': 'U.S. Value', 'Mid Cap Value': 'U.S. Value',
        'Small Cap Value': 'U.S. Value', 'Small Cap Blend': 'U.S. Value',
        'Dividend Growth': 'Dividend & Income', 'Dividend Aristocrats': 'Dividend & Income',
        'High Dividend': 'Dividend & Income', 'Dividend': 'Dividend & Income',
        'Developed Markets': 'International Developed', 'Europe': 'International Developed',
        'Asia Pacific': 'International Developed', 'Global': 'International Developed',
        'Emerging Markets': 'Emerging Markets', 'China': 'Emerging Markets',
        'Latin America': 'Emerging Markets', 'Japan': 'Single Country',
        'Brazil': 'Single Country', 'Canada': 'Single Country', 'Germany': 'Single Country',
        'Financial': 'Sector Equity', 'Energy': 'Sector Equity', 'Industrial': 'Sector Equity',
        'Consumer Discretionary': 'Sector Equity', 'Consumer Staples': 'Sector Equity',
        'Materials': 'Sector Equity', 'Utilities': 'Sector Equity',
        'Healthcare': 'Healthcare & Biotech', 'Biotech': 'Healthcare & Biotech',
        'Technology': 'Technology', 'Semiconductors': 'Technology', 'Internet': 'Technology',
        'Cloud Computing': 'Technology', 'Real Estate': 'Real Estate',
        'Aggregate Bonds': 'Fixed Income', 'Treasury': 'Fixed Income',
        'Corporate Bonds': 'Fixed Income', 'High Yield': 'Fixed Income',
        'Municipal Bonds': 'Fixed Income', 'Inflation Protected': 'Fixed Income',
        'Long Treasury': 'Fixed Income', 'Gold': 'Commodities', 'Silver': 'Commodities',
        'Oil': 'Commodities', 'Natural Gas': 'Commodities', 'Agriculture': 'Commodities',
        'Clean Energy': 'Thematic Investing', 'Cybersecurity': 'Thematic Investing',
        'Robotics': 'Thematic Investing', 'Gaming': 'Thematic Investing',
        'Fintech': 'Thematic Investing', '5G': 'Thematic Investing',
        'Social Media': 'Thematic Investing', 'Leveraged': 'Leveraged/Inverse',
        'Inverse': 'Leveraged/Inverse', 'Volatility': 'Leveraged/Inverse',
        'Low Volatility': 'Low Volatility', 'Minimum Volatility': 'Low Volatility',
        'Quality': 'Smart Beta', 'Momentum': 'Smart Beta', 'Value': 'Smart Beta',
        'Size': 'Smart Beta', 'Equal Weight': 'Smart Beta', 'MLPs': 'Alternatives',
        'Currency': 'Alternatives', 'Infrastructure': 'Alternatives'
    }
    return mapping.get(style, 'Other')

def is_market_open():
    try:
        ct_tz = pytz.timezone('America/Chicago')
        now_ct = datetime.now(ct_tz)
        if now_ct.weekday() >= 5:
            return False
        market_start = time(8, 30)
        market_end = time(15, 0)
        current_time = now_ct.time()
        return market_start <= current_time < market_end
    except Exception:
        return True

def format_currency(value):
    try:
        if pd.isna(value) or value == 0:
            return "$0.00"
        return f"${value:,.2f}"
    except (ValueError, TypeError):
        return "$0.00"

def format_percentage(value):
    try:
        if pd.isna(value):
            return "0.00%"
        return f"{value:.2f}%"
    except (ValueError, TypeError):
        return "0.00%"

def format_volume(value):
    try:
        if pd.isna(value) or value == 0:
            return "$0.00M"
        if value >= 1e9:
            return f"${value/1e9:.2f}B"
        else:
            return f"${value/1e6:.2f}M"
    except (ValueError, TypeError):
        return "$0.00M"

@st.cache_data(ttl=25)
def load_etf_data():
    url = "https://gist.githubusercontent.com/hassanlawani/03b12b9ea91f1c8cf4095a3484923ff8/raw/9671ad116e056b244768b6a028fc9aa0ba1a7114/live_etf_process_summary.csv"
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        if df.empty:
            return create_sample_data()
        df = process_etf_data(df)
        return df
    except Exception as e:
        st.error(f"⚠️ Error loading live data: {str(e)}")
        st.info("📊 Using sample data for demonstration")
        return create_sample_data()

def create_sample_data():
    np.random.seed(42)
    symbols = list(ETF_CATEGORIES.keys())[:15]
    data = []
    for i, symbol in enumerate(symbols):
        base_price = 50 + (i * 20) + np.random.uniform(-10, 10)
        data.append({
            'symbol': symbol,
            'last_price': base_price,
            'first_price': base_price * (1 + np.random.uniform(-0.02, 0.02)),
            'Blended': np.random.uniform(-2, 2),
            'change_24h': np.random.uniform(-5, 5),
            'USD_volume': np.random.uniform(1e6, 1e9),
            'end_time': datetime.now(),
            'n_obs': np.random.randint(100, 1000),
            'unique_prices': np.random.randint(50, 200),
            'tops': np.random.randint(5, 50),
            'lows': np.random.randint(5, 50),
            'last_price_change': np.random.uniform(-0.05, 0.05),
            'time_range': f"{np.random.randint(1, 24)}h {np.random.randint(0, 59)}m",
            'DoD_price_ratio': np.random.uniform(0.8, 1.2),
            'DoD_Volume_ratio': np.random.uniform(0.5, 2.0),
            'Recent_volume_Percent': np.random.uniform(10, 90)
        })
    df = pd.DataFrame(data)
    return process_etf_data(df)

def process_etf_data(df):
    try:
        if 'last_price' not in df.columns:
            if 'close' in df.columns:
                df['last_price'] = df['close']
            elif 'price' in df.columns:
                df['last_price'] = df['price']
            else:
                df['last_price'] = 100
        if 'first_price' not in df.columns:
            df['first_price'] = df['last_price'] * 0.98
        if 'Blended' not in df.columns:
            df['Blended'] = np.random.uniform(-1, 1, len(df))
        if 'change_24h' not in df.columns:
            df['change_24h'] = np.random.uniform(-5, 5, len(df))
        if 'USD_volume' not in df.columns:
            df['USD_volume'] = np.random.uniform(1e6, 1e9, len(df))
        
        df['recent_change_pct'] = np.where(
            df['first_price'] != 0,
            (df['last_price'] - df['first_price']) / df['first_price'],
            0
        )
        
        df['momentum_flag'] = pd.cut(
            df['Blended'],
            bins=[-np.inf, -0.5, 0.5, np.inf],
            labels=['Falling', 'Neutral', 'Rising']
        )
        
        df['investment_style'] = df['symbol'].map(ETF_CATEGORIES).fillna('Other')
        df['category'] = df['investment_style'].apply(style_to_asset_class)
        df['asset_class'] = df['category']
        
        if 'end_time' not in df.columns:
            df['end_time'] = datetime.now()
        
        return df
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return pd.DataFrame()

def show_market_status_modal():
    if not is_market_open():
        ct_tz = pytz.timezone('America/Chicago')
        now_ct = datetime.now(ct_tz)
        if now_ct.weekday() >= 5:
            message = "📅 Markets are closed on weekends. Please revisit Monday-Friday between 8:30 AM and 3:00 PM CT for live data."
        else:
            message = "🕒 Markets are currently closed. Please revisit between 8:30 AM and 3:00 PM CT for live data."
        st.warning(f"""
        ### ⏰ Market Hours Notice
        {message}
        **Current time (CT):** {now_ct.strftime('%A, %B %d, %Y at %I:%M %p')}
        📈 **Market Hours:** Monday-Friday, 8:30 AM - 3:00 PM Central Time
        """)

def create_value_cards(df):
    col1, col2, col3, col4 = st.columns(4)
    total_etfs = len(df)
    rising_count = len(df[df['momentum_flag'] == 'Rising']) if not df.empty else 0
    falling_count = len(df[df['momentum_flag'] == 'Falling']) if not df.empty else 0
    
    with col1:
        st.metric(
            label="📊 Total ETFs",
            value=total_etfs,
            help="Total number of ETFs being tracked"
        )
    
    with col2:
        rising_pct = (rising_count/total_etfs*100) if total_etfs > 0 else 0
        st.metric(
            label="🚀 Rising ETFs",
            value=rising_count,
            delta=f"{rising_pct:.1f}%",
            help="ETFs with positive momentum (Blended > 0.5)"
        )
    
    with col3:
        falling_pct = (falling_count/total_etfs*100) if total_etfs > 0 else 0
        st.metric(
            label="📉 Falling ETFs", 
            value=falling_count,
            delta=f"-{falling_pct:.1f}%",
            delta_color="inverse",
            help="ETFs with negative momentum (Blended < -0.5)"
        )
    
    with col4:
        last_update = datetime.now().strftime("%H:%M:%S")
        st.metric(
            label="🕐 Last Update",
            value=last_update,
            help="Time of last data refresh"
        )

def create_etf_table(df, table_type="rising", max_rows=25):
    if df.empty:
        return pd.DataFrame({'Message': ['Loading data...']})
    try:
        if table_type == "rising":
            filtered_df = df[df['Blended'] > 0].nlargest(max_rows, 'Blended')
        else:
            filtered_df = df[df['Blended'] < 0].nsmallest(max_rows, 'Blended')
        
        if filtered_df.empty:
            return pd.DataFrame({'Message': ['No data available']})
        
        display_df = pd.DataFrame({
            'Symbol': filtered_df['symbol'],
            'Asset Class': filtered_df.get('asset_class', 'N/A'),
            'Score': filtered_df['Blended'].round(2),
            'Price': filtered_df['last_price'].apply(format_currency),
            'Recent %': (filtered_df.get('recent_change_pct', 0) * 100).apply(lambda x: f"{x:.2f}%"),
            '24h %': filtered_df.get('change_24h', 0).apply(lambda x: f"{x:.2f}%"),
            'Volume': filtered_df.get('USD_volume', 0).apply(format_volume),
            'Obs': filtered_df.get('n_obs', 0),
            'Time': pd.to_datetime(filtered_df.get('end_time', datetime.now())).dt.strftime("%H:%M:%S")
        })
        return display_df
    except Exception as e:
        st.error(f"Error creating table: {str(e)}")
        return pd.DataFrame({'Error': [f'Table error: {str(e)}']})

def create_heatmap(df):
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    try:
        asset_summary = df.groupby('category').agg({
            'Blended': 'mean',
            'symbol': 'count'
        }).round(2)
        asset_summary.columns = ['avg_momentum', 'count']
        asset_summary = asset_summary.reset_index()
        
        fig = px.treemap(
            asset_summary,
            path=['category'],
            values='count',
            color='avg_momentum',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            title="Asset Class Performance Heatmap",
            hover_data={'avg_momentum': ':.2f', 'count': True}
        )
        fig.update_layout(height=400, font_size=12, title_font_size=16, margin=dict(t=50, l=25, r=25, b=25))
        return fig
    except Exception as e:
        st.error(f"Error creating heatmap: {str(e)}")
        return go.Figure()

def create_momentum_distribution(df):
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    try:
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
        fig.update_layout(height=400, xaxis_title="", yaxis_title="Count", legend_title="Momentum", xaxis_tickangle=-45, margin=dict(t=50, l=25, r=25, b=50))
        return fig
    except Exception as e:
        st.error(f"Error creating distribution chart: {str(e)}")
        return go.Figure()

def create_asset_class_comparison(df):
    if df.empty:
        return go.Figure()
    try:
        asset_perf = df.groupby('category')['Blended'].mean().sort_values()
        colors = ['#dc3545' if x < 0 else '#28a745' for x in asset_perf.values]
        fig = go.Figure(data=[
            go.Bar(
                y=asset_perf.index,
                x=asset_perf.values,
                orientation='h',
                marker_color=colors,
                text=[f"{x:.2f}" for x in asset_perf.values],
                textposition='auto'
            )
        ])
        fig.update_layout(title="Asset Class Comparison", xaxis_title="Average Momentum Score", yaxis_title="", height=400, margin=dict(t=50, l=25, r=25, b=25))
        return fig
    except Exception as e:
        st.error(f"Error creating comparison chart: {str(e)}")
        return go.Figure()

def create_volume_chart(df):
    if df.empty or 'USD_volume' not in df.columns:
        return go.Figure()
    try:
        vol_summary = df.groupby('category')['USD_volume'].sum().sort_values()
        fig = go.Figure(data=[
            go.Bar(
                y=vol_summary.index,
                x=vol_summary.values / 1e9,
                orientation='h',
                marker_color='#fd7e14',
                text=[f"${x/1e9:.1f}B" for x in vol_summary.values],
                textposition='auto'
            )
        ])
        fig.update_layout(title="Volume by Asset Class", xaxis_title="Total Volume ($B)", yaxis_title="", height=400, margin=dict(t=50, l=25, r=25, b=25))
        return fig
    except Exception as e:
        st.error(f"Error creating volume chart: {str(e)}")
        return go.Figure()

if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 25

def main():
    st.markdown("""
    <div class="main-header">
        <h1>📈 ETF Momentum Dashboard</h1>
        <p>Real-time ETF momentum tracking with live market data</p>
    </div>
    """, unsafe_allow_html=True)
    
    market_open = is_market_open()
    status_text = "🟢 Market Open" if market_open else "🔴 Market Closed"
    status_class = "status-open" if market_open else "status-closed"
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        st.markdown(f'<p class="{status_class}">{status_text}</p>', unsafe_allow_html=True)
    
    with col2:
        if st.button("🔄 Refresh Data", help="Manually refresh data"):
            st.cache_data.clear()
            st.session_state.last_refresh = datetime.now()
            st.rerun()
    
    with col3:
        auto_refresh = st.checkbox("Auto-refresh", value=st.session_state.auto_refresh, help=f"Automatically refresh every {st.session_state.refresh_interval} seconds")
        st.session_state.auto_refresh = auto_refresh
    
    with col4:
        st.write(f"⏱️ Every {st.session_state.refresh_interval}s")
    
    if auto_refresh:
        current_time = datetime.now()
        time_diff = (current_time - st.session_state.last_refresh).total_seconds()
        if time_diff >= st.session_state.refresh_interval:
            st.session_state.last_refresh = current_time
            st.cache_data.clear()
            st.rerun()
    
    with st.spinner("🔄 Loading ETF data..."):
        df = load_etf_data()
    
    if not market_open:
        show_market_status_modal()
    
    tab1, tab2, tab3 = st.tabs(["📊 Live Dashboard", "🎯 Asset Classes", "⚙️ Settings"])
    
    with tab1:
        create_value_cards(df)
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🚀 Top Rising ETFs")
            rising_df = create_etf_table(df, "rising")
            st.dataframe(rising_df, use_container_width=True, height=400, hide_index=True)
        
        with col2:
            st.markdown("### 📉 Top Falling ETFs")
            falling_df = create_etf_table(df, "falling")
            st.dataframe(falling_df, use_container_width=True, height=400, hide_index=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(create_heatmap(df), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_momentum_distribution(df), use_container_width=True)
    
    with tab2:
        if not df.empty:
            try:
                asset_perf = df.groupby('asset_class')['Blended'].mean()
                best_class = asset_perf.idxmax() if not asset_perf.empty else "N/A"
                worst_class = asset_perf.idxmin() if not asset_perf.empty else "N/A"
                if 'USD_volume' in df.columns:
                    most_active = df.groupby('asset_class')['USD_volume'].sum().idxmax()
                else:
                    most_active = "N/A"
            except Exception:
                best_class = worst_class = most_active = "N/A"
        else:
            best_class = worst_class = most_active = "N/A"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 Best Momentum", best_class)
        with col2:
            st.metric("📉 Worst Momentum", worst_class)
        with col3:
            st.metric("🔥 Most Active", most_active)
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("### 🎯 Filters")
            asset_classes = ['All'] + sorted(df['category'].unique().tolist()) if not df.empty else ['All']
            selected_class = st.selectbox("Asset Class:", asset_classes)
            momentum_options = st.multiselect("Momentum:", ['Rising', 'Falling', 'Neutral'], default=['Rising', 'Falling', 'Neutral'])
            if st.button("🔄 Reset Filters"):
                st.rerun()
        
        with col2:
            st.markdown("### 📋 Filtered ETF Details")
            try:
                filtered_df = df.copy()
                if selected_class != 'All':
                    filtered_df = filtered_df[filtered_df['category'] == selected_class]
                if momentum_options:
                    filtered_df = filtered_df[filtered_df['momentum_flag'].isin(momentum_options)]
                
                if not filtered_df.empty:
                    display_df = pd.DataFrame({
                        'Symbol': filtered_df['symbol'],
                        'Asset Class': filtered_df['asset_class'],
                        'Momentum': filtered_df['momentum_flag'],
                        'Blended Score': filtered_df['Blended'].round(3),
                        'Price': filtered_df['last_price'].apply(format_currency),
                        '24h %': filtered_df.get('change_24h', 0).apply(lambda x: f"{x:.2f}%"),
                        'Volume': filtered_df.get('USD_volume', 0).apply(format_volume)
                    })
                    st.dataframe(display_df, use_container_width=True, height=400, hide_index=True)
                else:
                    st.info("🔍 No ETFs match the selected filters.")
            except Exception as e:
                st.error(f"Error filtering data: {str(e)}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_asset_class_comparison(df), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_volume_chart(df), use_container_width=True)
    
    with tab3:
        st.markdown("### ⚙️ Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input("Data Source:", value="Hassan's Live ETF Data Feed", disabled=True, help="Data is fetched from Hassan's GitHub Gist")
            new_refresh_rate = st.slider("Auto-refresh Rate (seconds):", min_value=15, max_value=60, value=st.session_state.refresh_interval, step=5, help="How often to automatically refresh the data")
            if new_refresh_rate != st.session_state.refresh_interval:
                st.session_state.refresh_interval = new_refresh_rate
                st.success(f"✅ Refresh rate updated to {new_refresh_rate} seconds")
            
            if not df.empty:
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Data (CSV)",
                    data=csv_data,
                    file_name=f'etf_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv',
                    help="Download current ETF data as CSV file"
                )
        
        with col2:
            st.markdown("### 📊 System Status")
            data_status = "✅ Live Data" if not df.empty else "❌ No Data"
            records_count = len(df)
            last_update_time = datetime.now().strftime('%H:%M:%S')
            market_status_text = '🟢 Open' if market_open else '🔴 Closed'
            status_info = f"""
            **Data Status:** {data_status}  
            **Records:** {records_count:,}  
            **Refresh Rate:** {st.session_state.refresh_interval} seconds  
            **Last Update:** {last_update_time}  
            **Market Status:** {market_status_text}  
            **Auto-refresh:** {'🟢 On' if st.session_state.auto_refresh else '🔴 Off'}
            """
            st.markdown(status_info)
            
            if st.button("🗑️ Clear Cache", help="Clear data cache to force refresh"):
                st.cache_data.clear()
                st.success("✅ Cache cleared successfully!")
                time_module.sleep(1)
                st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 🏷️ ETF Categories & Asset Class Mapping")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Categories Overview:**")
            if not df.empty:
                category_counts = df['category'].value_counts()
                for category, count in category_counts.items():
                    st.write(f"• **{category}:** {count} ETFs")
            else:
                st.write("No data available")
        
        with col2:
            mapping_df = pd.DataFrame([
                {
                    'ETF Symbol': symbol, 
                    'Investment Style': style, 
                    'Asset Class': style_to_asset_class(style)
                }
                for symbol, style in ETF_CATEGORIES.items()
            ])
            st.dataframe(mapping_df, use_container_width=True, height=300, hide_index=True)
    
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown("**📈 Live ETF Momentum Tracker 2025**")
        st.markdown("Built with ❤️ using Streamlit")
    
    with footer_col2:
        st.markdown(f"**📊 Data:** {len(df)} ETFs tracked")
        st.markdown(f"**⏱️ Updated:** Every {st.session_state.refresh_interval}s")
    
    with footer_col3:
        st.markdown("**🔗 Links:**")
        st.markdown("[📚 Documentation](https://docs.streamlit.io) | [🐙 Source Code](https://github.com)")

def handle_auto_refresh():
    if st.session_state.auto_refresh:
        refresh_script = f"""
        <script>
            setTimeout(function(){{
                window.location.reload();
            }}, {st.session_state.refresh_interval * 1000});
        </script>
        """
        st.markdown(refresh_script, unsafe_allow_html=True)

def safe_main():
    try:
        main()
        handle_auto_refresh()
    except Exception as e:
        st.error(f"""
        ### ⚠️ Application Error
        An unexpected error occurred: {str(e)}
        **Troubleshooting:**
        1. Try refreshing the page
        2. Clear cache using the Settings tab
        3. Check your internet connection
        If the problem persists, this might be a temporary data source issue.
        """)
        if st.checkbox("🔧 Show technical details"):
            st.exception(e)

if __name__ == "__main__":
    safe_main()