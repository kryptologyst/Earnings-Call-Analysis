#!/usr/bin/env python3
"""Quick start script for earnings call analysis."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from earnings_analysis.models import EarningsAnalyzer
from earnings_analysis.utils import Config, setup_logging, set_seed


def main():
    """Quick start demonstration."""
    print("=" * 60)
    print("EARNINGS CALL ANALYSIS - QUICK START")
    print("=" * 60)
    print("DISCLAIMER: This is for research/education only. NOT investment advice.")
    print("=" * 60)
    
    # Setup
    setup_logging()
    set_seed(42)
    
    # Initialize analyzer
    print("Initializing analyzer...")
    config = Config()
    analyzer = EarningsAnalyzer(
        config=config,
        sentiment_model="textblob",  # Use TextBlob for quick demo
        topic_method="lda",
        num_topics=5
    )
    
    # Sample transcript
    transcript = """
    CEO: Good morning, everyone. We're excited to report that for Q1, the company has achieved a revenue growth of 12% year-over-year.
    We have successfully launched our new product line, which has exceeded initial projections by 20%.
    Our operating income has grown by 15%, and we are on track to meet our 2023 targets.
    
    CFO: The company has maintained strong liquidity with over $2 billion in cash reserves. Our expenses were well-controlled, and margins improved.
    We are investing heavily in R&D and new product development. Our team remains focused on driving sustainable growth.
    
    CEO: We're optimistic about the future, and we believe that our recent acquisitions will position us for even greater success in the coming quarters.
    """
    
    metadata = {
        "company": "Sample Corp",
        "quarter": "Q1 2023",
        "date": "2023-01-15",
        "participants": ["CEO", "CFO"]
    }
    
    # Analyze transcript
    print("Analyzing transcript...")
    results = analyzer.analyze(transcript, metadata)
    
    # Display results
    print(f"\nAnalysis Results:")
    print(f"Company: {results['company']}")
    print(f"Quarter: {results['quarter']}")
    print(f"Sentiment: {results['sentiment']['overall']['label']}")
    print(f"Confidence: {results['sentiment']['overall']['score']:.3f}")
    print(f"Word Count: {results['word_count']}")
    
    print(f"\nFinancial Terms Found:")
    for category, terms in results['financial_terms'].items():
        if terms:
            print(f"  {category}: {terms}")
    
    print(f"\nKey Phrases:")
    for category, phrases in results['key_phrases'].items():
        if phrases:
            print(f"  {category}: {phrases[:3]}...")  # Show first 3
    
    print(f"\n" + "=" * 60)
    print("QUICK START COMPLETED!")
    print("=" * 60)
    print("Next steps:")
    print("1. Run the Streamlit demo: streamlit run demo/app.py")
    print("2. Create sample data: python scripts/create_sample_data.py")
    print("3. Run evaluation: python scripts/evaluate_model.py")
    print("4. Run tests: pytest tests/ -v")


if __name__ == "__main__":
    main()
