#!/usr/bin/env python3
"""
Simple Earnings Call Analysis Example

This is a basic example script demonstrating earnings call analysis.
For the full-featured version, use the modernized package in src/earnings_analysis/

DISCLAIMER: This is for research and educational purposes only. NOT investment advice.
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
import re
import matplotlib.pyplot as plt


def analyze_earnings_call_simple(transcript: str) -> dict:
    """Simple earnings call analysis using basic NLP techniques.
    
    Args:
        transcript: Earnings call transcript text
        
    Returns:
        Dictionary with analysis results
    """
    # Sentiment analysis
    blob = TextBlob(transcript)
    sentiment_polarity = blob.sentiment.polarity
    sentiment_label = "Positive" if sentiment_polarity > 0.1 else "Negative" if sentiment_polarity < -0.1 else "Neutral"
    
    # Extract financial terms
    financial_terms = re.findall(r"\$[0-9,]+(?:\.[0-9]{1,2})?[BMK]?", transcript)
    growth_terms = re.findall(r"(revenue growth|operating income|net income|margins|cash reserves)", transcript, re.IGNORECASE)
    percentages = re.findall(r"\b\d+(?:\.\d+)?%\b", transcript)
    
    return {
        "sentiment": {
            "label": sentiment_label,
            "polarity": sentiment_polarity,
            "subjectivity": blob.sentiment.subjectivity
        },
        "financial_terms": {
            "monetary_values": financial_terms,
            "growth_metrics": growth_terms,
            "percentages": percentages
        },
        "text_stats": {
            "word_count": len(transcript.split()),
            "char_count": len(transcript)
        }
    }


def main():
    """Main function demonstrating simple earnings call analysis."""
    print("=" * 60)
    print("SIMPLE EARNINGS CALL ANALYSIS EXAMPLE")
    print("=" * 60)
    print("DISCLAIMER: This is for research/education only. NOT investment advice.")
    print("=" * 60)
    
    # Sample earnings call transcript
    earnings_call_transcript = """
    CEO: Good morning, everyone. We're excited to report that for Q1, the company has achieved a revenue growth of 12% year-over-year.
    We have successfully launched our new product line, which has exceeded initial projections by 20%.
    Our operating income has grown by 15%, and we are on track to meet our 2023 targets.
    
    CFO: The company has maintained strong liquidity with over $2 billion in cash reserves. Our expenses were well-controlled, and margins improved.
    We are investing heavily in R&D and new product development. Our team remains focused on driving sustainable growth.
    
    CEO: We're optimistic about the future, and we believe that our recent acquisitions will position us for even greater success in the coming quarters.
    """
    
    # Analyze the transcript
    results = analyze_earnings_call_simple(earnings_call_transcript)
    
    # Display results
    print(f"\nOverall Sentiment: {results['sentiment']['label']} (Polarity: {results['sentiment']['polarity']:.2f})")
    print(f"Subjectivity: {results['sentiment']['subjectivity']:.2f}")
    print(f"Word Count: {results['text_stats']['word_count']}")
    print(f"Character Count: {results['text_stats']['char_count']}")
    
    print(f"\nFinancial Terms Found:")
    print(f"  Monetary Values: {results['financial_terms']['monetary_values']}")
    print(f"  Growth Metrics: {results['financial_terms']['growth_metrics']}")
    print(f"  Percentages: {results['financial_terms']['percentages']}")
    
    # Simple visualization
    plt.figure(figsize=(10, 6))
    
    # Sentiment bar chart
    plt.subplot(1, 2, 1)
    colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'orange'}
    plt.bar([results['sentiment']['label']], [results['sentiment']['polarity']], 
            color=colors[results['sentiment']['label']])
    plt.title("Earnings Call Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Polarity Score")
    plt.ylim(-1, 1)
    
    # Financial terms count
    plt.subplot(1, 2, 2)
    term_counts = {
        'Monetary': len(results['financial_terms']['monetary_values']),
        'Growth': len(results['financial_terms']['growth_metrics']),
        'Percentages': len(results['financial_terms']['percentages'])
    }
    plt.bar(term_counts.keys(), term_counts.values(), color=['blue', 'purple', 'orange'])
    plt.title("Financial Terms Count")
    plt.xlabel("Term Type")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFor advanced analysis, use the full package:")
    print(f"  from src.earnings_analysis.models import EarningsAnalyzer")
    print(f"  analyzer = EarningsAnalyzer()")
    print(f"  results = analyzer.analyze(transcript)")


if __name__ == "__main__":
    main()



