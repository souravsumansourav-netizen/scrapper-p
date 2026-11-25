from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


STOCK_IN_STOCK_ONLINE = "in_stock_online"
STOCK_IN_STOCK_STORE_ONLY = "in_stock_store_only"
STOCK_OUT_OF_STOCK = "out_of_stock"
STOCK_UNKNOWN = "unknown"

VALID_STOCK_STATUSES = {
    STOCK_IN_STOCK_ONLINE,
    STOCK_IN_STOCK_STORE_ONLY,
    STOCK_OUT_OF_STOCK,
    STOCK_UNKNOWN,
}


@dataclass
class ProductResult:
    """Standardized product extraction result."""

    url: str
    title: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    stock_status: str = STOCK_UNKNOWN
    tier: Optional[str] = None
    error: Optional[str] = None

    def is_complete(self) -> bool:
        """Return True if core fields (price and resolved stock_status) are present."""
        return (
            self.price is not None
            and self.stock_status != STOCK_UNKNOWN
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dictionary for JSON serialization."""
        return asdict(self)

