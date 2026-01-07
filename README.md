# Earnings Call Analysis - Research & Education Only

**DISCLAIMER: This project is for research and educational purposes only. It is NOT investment advice. The analysis may be inaccurate and should not be used for making investment decisions. Past performance does not guarantee future results.**

## Overview

This project provides advanced NLP analysis of earnings call transcripts for research and educational purposes. It includes sentiment analysis, topic modeling, key phrase extraction, and interactive visualization capabilities.

## Features

- **Sentiment Analysis**: Using FinBERT and traditional NLP methods
- **Topic Modeling**: LDA and BERT-based topic extraction
- **Key Phrase Extraction**: Financial metrics and insights extraction
- **Interactive Demo**: Streamlit-based web interface
- **Reproducible Research**: Deterministic seeding and proper evaluation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Earnings-Call-Analysis.git
cd Earnings-Call-Analysis

# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Basic Usage

```python
from src.earnings_analysis.models import EarningsAnalyzer

# Initialize analyzer
analyzer = EarningsAnalyzer()

# Analyze transcript
transcript = "Your earnings call transcript here..."
results = analyzer.analyze(transcript)

print(f"Sentiment: {results['sentiment']}")
print(f"Key Topics: {results['topics']}")
print(f"Financial Metrics: {results['metrics']}")
```

### Run Demo

```bash
streamlit run demo/app.py
```

## Project Structure

```
├── src/earnings_analysis/     # Main source code
│   ├── data/                 # Data processing modules
│   ├── features/             # Feature extraction
│   ├── models/               # NLP models and analysis
│   └── utils/                # Utility functions
├── configs/                  # Configuration files
├── scripts/                  # Training and evaluation scripts
├── notebooks/                # Jupyter notebooks for exploration
├── tests/                    # Unit tests
├── assets/                   # Generated plots and results
├── demo/                     # Streamlit demo application
└── data/                     # Sample data and datasets
```

## Dataset Schema

The project expects earnings call transcripts in the following format:

```json
{
  "transcript_id": "unique_identifier",
  "company": "Company Name",
  "quarter": "Q1 2026",
  "date": "2026-01-07",
  "transcript": "Full earnings call transcript text...",
  "participants": ["CEO", "CFO", "Analyst"],
  "metadata": {
    "duration": 3600,
    "language": "en"
  }
}
```

## Evaluation Metrics

- **Sentiment Analysis**: Accuracy, F1-score, AUROC
- **Topic Modeling**: Coherence score, perplexity
- **Key Phrase Extraction**: Precision, Recall, F1-score
- **Overall**: Cross-validation with time-based splits

## Configuration

Configuration files are located in `configs/` and use OmegaConf format:

```yaml
model:
  sentiment_model: "finbert"
  topic_model: "lda"
  num_topics: 10

data:
  max_length: 512
  batch_size: 16

evaluation:
  cv_folds: 5
  test_size: 0.2
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run linting: `black . && ruff check .`
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Research Citation

If you use this project in your research, please cite:

```bibtex
@software{earnings_call_analysis,
  title={Earnings Call Analysis using NLP},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Earnings-Call-Analysis}
}
```
# Earnings-Call-Analysis
