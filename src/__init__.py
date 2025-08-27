from .base import ChordExtractor
from .btc.model import LargeBTCExtractor, SmallBTCExtractor


def get_extractor(name: str) -> ChordExtractor:
    """
    Factory function to get a chord extractor by name.

    Args:
        name (str): The name of the chord extractor. Options are "small_btc" and "large_btc".

    Returns:
        ChordExtractor: An instance of the requested chord extractor.

    Raises:
        ValueError: If the provided name does not match any known extractors.
    """
    mapping = {
        "small_btc": lambda: SmallBTCExtractor(),
        "large_btc": lambda: LargeBTCExtractor(),
    }
    if name not in mapping:
        raise ValueError(f"Unknown extractor name: {name}. Available options are: {list(mapping.keys())}")
    return mapping[name]()
