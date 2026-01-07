"""Configuration management for earnings call analysis."""

from typing import Any, Dict, Optional
from pathlib import Path
import yaml
from omegaconf import OmegaConf


class Config:
    """Configuration manager for earnings call analysis.
    
    Handles loading and validation of configuration files.
    """
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file or use defaults."""
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        else:
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "model": {
                "sentiment_model": "finbert",
                "topic_model": "lda", 
                "num_topics": 10,
                "max_length": 512,
                "batch_size": 16
            },
            "data": {
                "min_transcript_length": 100,
                "max_transcript_length": 10000,
                "language": "en"
            },
            "evaluation": {
                "cv_folds": 5,
                "test_size": 0.2,
                "random_state": 42
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return OmegaConf.select(self._config, key, default=default)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        self._config = OmegaConf.merge(self._config, updates)
    
    def save(self, path: str) -> None:
        """Save configuration to file.
        
        Args:
            path: Path to save configuration
        """
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration dictionary."""
        return self._config
