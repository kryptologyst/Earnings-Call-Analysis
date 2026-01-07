"""Sample earnings call transcript data for testing and demonstration."""

import json
from pathlib import Path
from typing import List, Dict, Any


def create_sample_transcripts() -> List[Dict[str, Any]]:
    """Create sample earnings call transcripts for testing.
    
    Returns:
        List of sample transcript dictionaries
    """
    return [
        {
            "transcript_id": "AAPL_Q1_2023",
            "company": "Apple Inc.",
            "quarter": "Q1 2023",
            "date": "2023-01-26",
            "participants": ["CEO", "CFO", "Analyst"],
            "transcript": """
            CEO: Good morning, everyone. We're excited to report that for Q1, Apple has achieved a revenue growth of 12% year-over-year.
            We have successfully launched our new product line, which has exceeded initial projections by 20%.
            Our operating income has grown by 15%, and we are on track to meet our 2023 targets.
            
            CFO: The company has maintained strong liquidity with over $2 billion in cash reserves. Our expenses were well-controlled, and margins improved.
            We are investing heavily in R&D and new product development. Our team remains focused on driving sustainable growth.
            
            CEO: We're optimistic about the future, and we believe that our recent acquisitions will position us for even greater success in the coming quarters.
            """,
            "metadata": {
                "duration": 3600,
                "language": "en",
                "source": "synthetic"
            }
        },
        {
            "transcript_id": "MSFT_Q2_2023",
            "company": "Microsoft Corporation",
            "quarter": "Q2 2023",
            "date": "2023-04-25",
            "participants": ["CEO", "CFO", "CTO"],
            "transcript": """
            CEO: Thank you for joining us today. Microsoft delivered strong results this quarter with revenue of $52.9 billion, up 7% year-over-year.
            Our cloud business continues to be a key growth driver, with Azure revenue growing 27%.
            
            CFO: Operating income was $22.4 billion, up 10%. We're seeing strong demand for our productivity and business processes.
            Our commercial cloud revenue was $28.5 billion, up 22%.
            
            CTO: We're investing heavily in AI capabilities and believe this will be a significant growth opportunity.
            Our partnership with OpenAI is already showing promising results.
            """,
            "metadata": {
                "duration": 4200,
                "language": "en",
                "source": "synthetic"
            }
        },
        {
            "transcript_id": "TSLA_Q3_2023",
            "company": "Tesla Inc.",
            "quarter": "Q3 2023",
            "date": "2023-10-18",
            "participants": ["CEO", "CFO"],
            "transcript": """
            CEO: Tesla delivered 435,059 vehicles in Q3, up 27% year-over-year. Revenue was $23.4 billion, up 9%.
            We're facing some headwinds with supply chain issues, but our production capacity continues to expand.
            
            CFO: Automotive revenue was $19.6 billion, up 5%. Energy generation and storage revenue was $1.1 billion, up 40%.
            We're maintaining our focus on cost reduction and operational efficiency.
            
            CEO: Looking ahead, we're optimistic about our Full Self-Driving capabilities and energy storage business.
            We expect to see continued growth in both areas.
            """,
            "metadata": {
                "duration": 3000,
                "language": "en",
                "source": "synthetic"
            }
        },
        {
            "transcript_id": "NFLX_Q4_2023",
            "company": "Netflix Inc.",
            "quarter": "Q4 2023",
            "date": "2024-01-23",
            "participants": ["CEO", "CFO", "COO"],
            "transcript": """
            CEO: Netflix added 13.1 million paid memberships in Q4, bringing our total to 260.3 million.
            Revenue was $8.8 billion, up 12% year-over-year. We're pleased with our content strategy and international expansion.
            
            CFO: Operating income was $1.5 billion, up 16%. We're investing heavily in content production and technology.
            Our password sharing crackdown has been successful in driving subscriber growth.
            
            COO: We're seeing strong engagement with our ad-supported tier. International markets continue to be a key growth driver.
            We're optimistic about our competitive position in the streaming market.
            """,
            "metadata": {
                "duration": 3300,
                "language": "en",
                "source": "synthetic"
            }
        },
        {
            "transcript_id": "GOOGL_Q1_2024",
            "company": "Alphabet Inc.",
            "quarter": "Q1 2024",
            "date": "2024-04-25",
            "participants": ["CEO", "CFO", "SVP"],
            "transcript": """
            CEO: Alphabet reported revenue of $80.5 billion, up 15% year-over-year. Our AI investments are paying off with strong growth in cloud and advertising.
            
            CFO: Google Services revenue was $70.4 billion, up 13%. Google Cloud revenue was $9.6 billion, up 28%.
            We're seeing strong demand for our AI-powered products and services.
            
            SVP: Our YouTube business continues to grow with revenue of $8.1 billion, up 20%.
            We're investing in AI to improve user experience and advertiser effectiveness.
            """,
            "metadata": {
                "duration": 3900,
                "language": "en",
                "source": "synthetic"
            }
        }
    ]


def save_sample_data(data_dir: str = "data") -> None:
    """Save sample data to files.
    
    Args:
        data_dir: Directory to save sample data
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    transcripts = create_sample_transcripts()
    
    # Save individual transcript files
    for transcript in transcripts:
        filename = f"{transcript['transcript_id']}.json"
        filepath = data_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)
    
    # Save combined file
    combined_filepath = data_path / "sample_transcripts.json"
    with open(combined_filepath, 'w', encoding='utf-8') as f:
        json.dump(transcripts, f, indent=2, ensure_ascii=False)
    
    print(f"Sample data saved to {data_path}")
    print(f"Created {len(transcripts)} sample transcripts")


if __name__ == "__main__":
    save_sample_data()
