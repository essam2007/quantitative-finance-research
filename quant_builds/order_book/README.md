# Order Book Simulator

Simplified limit order book for high-frequency style experiments.

## Features

- **Bid/ask queues**: price levels with aggregate quantity
- **Limit orders**: add/cancel at price
- **Market orders**: walk the book; optional **price impact** (cost ∝ √qty or linear)
- **Order flow**: Poisson arrivals; mix of limit and market orders

## Usage

```python
from quant_builds.order_book import OrderBook, simulate_order_flow

book = OrderBook(tick_size=0.01)
book.add_limit("BID", 99.99, 100)
book.add_limit("ASK", 100.01, 50)
filled, avg_price, impact = book.market_order("BUY", 30, impact_factor=0.01)

book, events = simulate_order_flow(n_events=500, lambda_arrival=10, seed=42)
```

Run: `python quant_builds/order_book/engine.py`
