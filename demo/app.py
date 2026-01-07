"""Streamlit demo application for earnings call analysis."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from earnings_analysis.models import EarningsAnalyzer
from earnings_analysis.utils import Config, setup_logging, set_seed

# Page configuration
st.set_page_config(
    page_title="Earnings Call Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None


@st.cache_data
def load_sample_data():
    """Load sample data."""
    sample_path = Path("data/sample_transcripts.json")
    if sample_path.exists():
        with open(sample_path, 'r') as f:
            return json.load(f)
    return None


@st.cache_resource
def initialize_analyzer(sentiment_model: str, topic_method: str, num_topics: int):
    """Initialize analyzer with caching."""
    config = Config()
    return EarningsAnalyzer(
        config=config,
        sentiment_model=sentiment_model,
        topic_method=topic_method,
        num_topics=num_topics
    )


def display_disclaimer():
    """Display disclaimer."""
    st.markdown("""
    <div class="disclaimer">
        <h4>⚠️ IMPORTANT DISCLAIMER</h4>
        <p><strong>This tool is for research and educational purposes only.</strong></p>
        <ul>
            <li>This is NOT investment advice</li>
            <li>Analysis results may be inaccurate</li>
            <li>Do NOT use for investment decisions</li>
            <li>Consult qualified financial advisors</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def create_sentiment_chart(sentiment_data: Dict[str, Any]) -> go.Figure:
    """Create sentiment visualization."""
    fig = go.Figure()
    
    # Overall sentiment
    sentiment = sentiment_data["overall"]
    label = sentiment["label"]
    score = sentiment["score"]
    
    # Color mapping
    color_map = {
        "POSITIVE": "#28a745",
        "NEGATIVE": "#dc3545", 
        "NEUTRAL": "#ffc107"
    }
    
    fig.add_trace(go.Bar(
        x=[label],
        y=[score],
        marker_color=color_map.get(label, "#6c757d"),
        text=[f"{score:.3f}"],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Overall Sentiment Analysis",
        xaxis_title="Sentiment",
        yaxis_title="Confidence Score",
        height=400
    )
    
    return fig


def create_topic_chart(topics: List[Dict[str, Any]]) -> go.Figure:
    """Create topic visualization."""
    if not topics:
        return go.Figure()
    
    fig = go.Figure()
    
    for topic in topics:
        topic_id = topic["topic_id"]
        words = topic["words"]
        
        if words:
            word_text = ", ".join([word for word, _ in words[:5]])
            fig.add_trace(go.Bar(
                x=[f"Topic {topic_id}"],
                y=[len(words)],
                text=[word_text],
                textposition='auto',
                name=f"Topic {topic_id}"
            ))
    
    fig.update_layout(
        title="Topic Distribution",
        xaxis_title="Topics",
        yaxis_title="Number of Words",
        height=400
    )
    
    return fig


def create_key_phrases_chart(key_phrases: Dict[str, List]) -> go.Figure:
    """Create key phrases visualization."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Financial Metrics", "Growth Terms", "Monetary Values", "Time Periods"),
        specs=[[{"type": "pie"}, {"type": "pie"}],
               [{"type": "pie"}, {"type": "pie"}]]
    )
    
    # Financial metrics
    if key_phrases.get("financial_metrics"):
        fig.add_trace(
            go.Pie(
                labels=key_phrases["financial_metrics"][:10],
                values=[1] * len(key_phrases["financial_metrics"][:10]),
                name="Financial Metrics"
            ),
            row=1, col=1
        )
    
    # Growth terms
    if key_phrases.get("growth_terms"):
        fig.add_trace(
            go.Pie(
                labels=key_phrases["growth_terms"][:10],
                values=[1] * len(key_phrases["growth_terms"][:10]),
                name="Growth Terms"
            ),
            row=1, col=2
        )
    
    # Monetary values
    if key_phrases.get("monetary_values"):
        monetary_labels = [item["value"] for item in key_phrases["monetary_values"][:10]]
        fig.add_trace(
            go.Pie(
                labels=monetary_labels,
                values=[1] * len(monetary_labels),
                name="Monetary Values"
            ),
            row=2, col=1
        )
    
    # Time periods
    if key_phrases.get("time_periods"):
        fig.add_trace(
            go.Pie(
                labels=key_phrases["time_periods"][:10],
                values=[1] * len(key_phrases["time_periods"][:10]),
                name="Time Periods"
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title="Key Phrases Distribution",
        height=600,
        showlegend=False
    )
    
    return fig


def main():
    """Main application."""
    # Header
    st.markdown('<h1 class="main-header">📊 Earnings Call Analysis</h1>', unsafe_allow_html=True)
    
    # Display disclaimer
    display_disclaimer()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    sentiment_model = st.sidebar.selectbox(
        "Sentiment Model",
        ["finbert", "textblob"],
        help="Choose the sentiment analysis model"
    )
    
    topic_method = st.sidebar.selectbox(
        "Topic Modeling Method",
        ["lda", "tfidf_lda", "bert_clustering"],
        help="Choose the topic modeling method"
    )
    
    num_topics = st.sidebar.slider(
        "Number of Topics",
        min_value=5,
        max_value=20,
        value=10,
        help="Number of topics to extract"
    )
    
    # Initialize analyzer
    if st.sidebar.button("Initialize Analyzer"):
        with st.spinner("Initializing analyzer..."):
            st.session_state.analyzer = initialize_analyzer(
                sentiment_model, topic_method, num_topics
            )
        st.success("Analyzer initialized successfully!")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Single Analysis", "Batch Analysis", "Sample Data", "About"])
    
    with tab1:
        st.header("Single Transcript Analysis")
        
        # Text input
        transcript_text = st.text_area(
            "Enter earnings call transcript:",
            height=300,
            placeholder="Paste your earnings call transcript here..."
        )
        
        # Metadata input
        col1, col2 = st.columns(2)
        with col1:
            company = st.text_input("Company", value="Sample Company")
            quarter = st.text_input("Quarter", value="Q1 2024")
        with col2:
            date = st.text_input("Date", value="2024-01-15")
            participants = st.text_input("Participants (comma-separated)", value="CEO, CFO")
        
        if st.button("Analyze Transcript") and transcript_text:
            if st.session_state.analyzer is None:
                st.error("Please initialize the analyzer first!")
            else:
                with st.spinner("Analyzing transcript..."):
                    metadata = {
                        "company": company,
                        "quarter": quarter,
                        "date": date,
                        "participants": [p.strip() for p in participants.split(",")]
                    }
                    
                    try:
                        result = st.session_state.analyzer.analyze(transcript_text, metadata)
                        st.session_state.analysis_results = result
                        
                        st.success("Analysis completed successfully!")
                        
                        # Display results
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Word Count", result["word_count"])
                        with col2:
                            st.metric("Character Count", result["char_count"])
                        with col3:
                            st.metric("Sentiment", result["sentiment"]["overall"]["label"])
                        with col4:
                            st.metric("Confidence", f"{result['sentiment']['overall']['score']:.3f}")
                        
                        # Visualizations
                        st.subheader("Sentiment Analysis")
                        sentiment_chart = create_sentiment_chart(result["sentiment"])
                        st.plotly_chart(sentiment_chart, use_container_width=True)
                        
                        st.subheader("Key Phrases")
                        phrases_chart = create_key_phrases_chart(result["key_phrases"])
                        st.plotly_chart(phrases_chart, use_container_width=True)
                        
                        # Detailed results
                        with st.expander("Detailed Results"):
                            st.json(result)
                            
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
    
    with tab2:
        st.header("Batch Analysis")
        
        # Load sample data
        if st.button("Load Sample Data"):
            sample_data = load_sample_data()
            if sample_data:
                st.session_state.sample_data = sample_data
                st.success(f"Loaded {len(sample_data)} sample transcripts")
            else:
                st.error("Sample data not found. Please run create_sample_data.py first.")
        
        if st.session_state.sample_data and st.button("Run Batch Analysis"):
            if st.session_state.analyzer is None:
                st.error("Please initialize the analyzer first!")
            else:
                with st.spinner("Running batch analysis..."):
                    try:
                        results = st.session_state.analyzer.analyze_batch(st.session_state.sample_data)
                        
                        # Create summary
                        df_results = pd.DataFrame([
                            {
                                "Company": r["company"],
                                "Quarter": r["quarter"],
                                "Sentiment": r["sentiment"]["overall"]["label"],
                                "Confidence": r["sentiment"]["overall"]["score"],
                                "Word Count": r["word_count"]
                            }
                            for r in results
                        ])
                        
                        st.subheader("Batch Analysis Results")
                        st.dataframe(df_results, use_container_width=True)
                        
                        # Summary statistics
                        st.subheader("Summary Statistics")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            sentiment_counts = df_results["Sentiment"].value_counts()
                            fig_sentiment = px.pie(
                                values=sentiment_counts.values,
                                names=sentiment_counts.index,
                                title="Sentiment Distribution"
                            )
                            st.plotly_chart(fig_sentiment, use_container_width=True)
                        
                        with col2:
                            fig_confidence = px.histogram(
                                df_results,
                                x="Confidence",
                                title="Confidence Score Distribution"
                            )
                            st.plotly_chart(fig_confidence, use_container_width=True)
                        
                        # Save results
                        if st.button("Save Results"):
                            results_path = Path("assets/batch_results.json")
                            results_path.parent.mkdir(exist_ok=True)
                            
                            with open(results_path, 'w') as f:
                                json.dump(results, f, indent=2, default=str)
                            
                            st.success(f"Results saved to {results_path}")
                        
                    except Exception as e:
                        st.error(f"Batch analysis failed: {str(e)}")
    
    with tab3:
        st.header("Sample Data")
        
        sample_data = load_sample_data()
        if sample_data:
            st.success(f"Found {len(sample_data)} sample transcripts")
            
            # Display sample data
            for i, transcript in enumerate(sample_data):
                with st.expander(f"{transcript['company']} - {transcript['quarter']}"):
                    st.write(f"**Date:** {transcript['date']}")
                    st.write(f"**Participants:** {', '.join(transcript['participants'])}")
                    st.write(f"**Transcript:**")
                    st.text(transcript['transcript'][:500] + "..." if len(transcript['transcript']) > 500 else transcript['transcript'])
        else:
            st.error("Sample data not found. Please run create_sample_data.py first.")
            st.code("python scripts/create_sample_data.py")
    
    with tab4:
        st.header("About This Application")
        
        st.markdown("""
        ## Earnings Call Analysis Tool
        
        This application provides comprehensive analysis of earnings call transcripts using advanced NLP techniques.
        
        ### Features:
        - **Sentiment Analysis**: Using FinBERT or TextBlob
        - **Topic Modeling**: LDA, TF-IDF LDA, or BERT clustering
        - **Key Phrase Extraction**: Financial metrics, growth terms, monetary values
        - **Batch Processing**: Analyze multiple transcripts at once
        - **Interactive Visualizations**: Charts and graphs for insights
        
        ### Models Used:
        - **FinBERT**: Financial domain-specific BERT model for sentiment
        - **LDA**: Latent Dirichlet Allocation for topic modeling
        - **TF-IDF**: Term frequency-inverse document frequency
        - **BERT**: Bidirectional Encoder Representations from Transformers
        
        ### Usage:
        1. Configure the models in the sidebar
        2. Initialize the analyzer
        3. Choose single analysis or batch analysis
        4. View results and visualizations
        
        ### Technical Details:
        - Built with Streamlit for the frontend
        - Uses Transformers library for NLP models
        - Plotly for interactive visualizations
        - Pandas for data manipulation
        """)
        
        st.markdown("""
        ### Disclaimer:
        This tool is for research and educational purposes only. 
        It is NOT investment advice and should not be used for making investment decisions.
        """)


if __name__ == "__main__":
    main()
