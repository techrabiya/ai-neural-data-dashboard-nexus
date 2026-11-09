import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Neural Data Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: #38bdf8;
    }
    .stSidebar {
        background-color: #1e293b;
    }
    h1, h2, h3 {
        color: #38bdf8;
        font-family: 'Courier New', monospace;
    }
    .dashboard-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid #38bdf8;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.15);
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Advanced AI Multi-Feature Neural Data Dashboard")
st.markdown("---")

st.sidebar.header("🎛️ Neural Control Matrix")
selected_tab = st.sidebar.radio("Navigation Matrix", ["Live Data Metrics", "Graphical Visualizer", "Data Filter Matrix"])
data_source = st.sidebar.selectbox("Data Stream Source", ["Synthetic Neural Stream", "Custom CSV Upload"])

try:
    if data_source == "Custom CSV Upload":
        uploaded_file = st.sidebar.file_uploader("Upload Data Matrix File (CSV)", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("Custom CSV stream successfully loaded!")
        else:
            st.info("Smart Warning: Please upload a CSV file via the sidebar to initialize custom data streams.")
            np.random.seed(42)
            df = pd.DataFrame({
                "Timestamp": pd.date_range(start="2026-01-01", periods=100, freq="H"),
                "Neural_Load": np.random.uniform(10.0, 100.0, 100),
                "Signal_Efficiency": np.random.uniform(50.0, 99.9, 100),
                "Category": np.random.choice(["Alpha", "Beta", "Gamma"], 100)
            })
    else:
        np.random.seed(42)
        df = pd.DataFrame({
            "Timestamp": pd.date_range(start="2026-01-01", periods=100, freq="H"),
            "Neural_Load": np.random.uniform(10.0, 100.0, 100),
            "Signal_Efficiency": np.random.uniform(50.0, 99.9, 100),
            "Category": np.random.choice(["Alpha", "Beta", "Gamma"], 100)
        })

    if selected_tab == "Live Data Metrics":
        st.subheader("📊 Live Neural Metrics & Data Matrix")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="⚡ Avg Neural Load", value=f"{df['Neural_Load'].mean():.2f}%", delta="+2.4%")
        with col2:
            st.metric(label="📈 Avg Signal Efficiency", value=f"{df['Signal_Efficiency'].mean():.2f}%", delta="+1.1%")
        with col3:
            st.metric(label="📁 Total Data Points", value=len(df), delta="Active")
            
        st.markdown("---")
        st.dataframe(df.head(15), use_container_width=True)

    elif selected_tab == "Graphical Visualizer":
        st.subheader("📈 Dynamic Graphical Visualizer")
        
        col_a, col_b = st.columns(2)
        with col_a:
            chart_type = st.selectbox("Select Visualizer Type", ["Line Graph", "Scatter Plot", "Histogram"])
        with col_b:
            target_metric = st.selectbox("Target Matrix Column", ["Neural_Load", "Signal_Efficiency"])
            
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        
        if chart_type == "Line Graph":
            ax.plot(df[target_metric], color='#38bdf8', linewidth=2)
            ax.set_title(f"Neural Trend Line: {target_metric}", color='#38bdf8')
        elif chart_type == "Scatter Plot":
            ax.scatter(df.index, df[target_metric], color='#38bdf8', alpha=0.7)
            ax.set_title(f"Neural Scatter Matrix: {target_metric}", color='#38bdf8')
        else:
            ax.hist(df[target_metric], bins=15, color='#38bdf8', edgecolor='#0f172a')
            ax.set_title(f"Neural Frequency Distribution: {target_metric}", color='#38bdf8')
            
        ax.tick_params(colors='#38bdf8')
        ax.xaxis.label.set_color('#38bdf8')
        ax.yaxis.label.set_color('#38bdf8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#38bdf8')
            
        st.pyplot(fig)

    else:
        st.subheader("🔍 Custom Text & Number Filtering Matrix")
        search_query = st.text_input("Search Category or Text Parameter", "")
        min_load, max_load = st.slider("Filter Neural Load Range", float(df['Neural_Load'].min()), float(df['Neural_Load'].max()), (float(df['Neural_Load'].min()), float(df['Neural_Load'].max())))
        
        filtered_df = df[(df['Neural_Load'] >= min_load) & (df['Neural_Load'] <= max_load)]
        if search_query:
            filtered_df = filtered_df[filtered_df['Category'].str.contains(search_query, case=False, na=False)]
            
        st.success(f"Filter Matrix Applied: {len(filtered_df)} records matched criteria.")
        st.dataframe(filtered_df, use_container_width=True)

except Exception as e:
    st.error(f"Smart Warning: An anomaly was caught by the zero-crash error handler: {e}")
    st.info("The application remains online and stable. Please check your data stream format or input parameters.")
