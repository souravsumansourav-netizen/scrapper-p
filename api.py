import asyncio
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl

from scraper import extract_product

app = FastAPI(title="Product Extractor API", version="1.0.0")

# Limit concurrent extractions (especially Playwright fallbacks) to protect memory.
_EXTRACT_SEMAPHORE = asyncio.Semaphore(3)


class ExtractRequest(BaseModel):
    url: HttpUrl


class ExtractResponse(BaseModel):
    url: str
    title: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    stock_status: str
    tier: Optional[str] = None
    error: Optional[str] = None


@app.post("/extract", response_model=ExtractResponse)
async def extract_endpoint(payload: ExtractRequest) -> ExtractResponse:
    """Extract product details from a given e-commerce URL."""
    async with _EXTRACT_SEMAPHORE:
        data = await asyncio.to_thread(extract_product, str(payload.url))
    return ExtractResponse(**data)
