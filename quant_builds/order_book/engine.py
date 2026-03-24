"""
Simplified Limit Order Book (LOB) Simulator.

- Bid/ask queues (price levels)
- Market orders, limit orders
- Price impact
- Optional: Poisson order arrivals, market impact models
"""

import numpy as np
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# -----------------------------------------------------------------------------
# Order book (price → quantity; separate bid/ask)
# -----------------------------------------------------------------------------

class OrderBook:
    """Simplified LOB: bids and asks as dicts price -> total qty at level."""

    def __init__(self, tick_size=0.01):
        self.tick = tick_size
        self.bids = defaultdict(float)  # price -> qty
        self.asks = defaultdict(float)
        self.mid = None
        self.last_trade = None

    def best_bid(self):
        return max(self.bids.keys()) if self.bids else None

    def best_ask(self):
        return min(self.asks.keys()) if self.asks else None

    def spread(self):
        b, a = self.best_bid(), self.best_ask()
        if b is not None and a is not None:
            return a - b
        return None

    def add_limit(self, side, price, qty):
        price = round(price / self.tick) * self.tick
        if side.upper() == "BID":
            self.bids[price] += qty
        else:
            self.asks[price] += qty
        self._update_mid()

    def cancel(self, side, price, qty):
        price = round(price / self.tick) * self.tick
        if side.upper() == "BID":
            self.bids[price] = max(0, self.bids[price] - qty)
            if self.bids[price] == 0:
                del self.bids[price]
        else:
            self.asks[price] = max(0, self.asks[price] - qty)
            if self.asks[price] == 0:
                del self.asks[price]
        self._update_mid()

    def _update_mid(self):
        b, a = self.best_bid(), self.best_ask()
        if b is not None and a is not None:
            self.mid = (b + a) / 2
        elif b is not None:
            self.mid = b
        elif a is not None:
            self.mid = a

    def market_order(self, side, qty, impact_factor=0.01):
        """
        Execute market order: walk the book. impact_factor adds cost per unit filled.
        Returns (filled_qty, avg_price, impact_cost).
        """
        filled = 0
        cost = 0.0
        if side.upper() == "SELL":
            levels = sorted(self.bids.keys(), reverse=True)
            book = self.bids
        else:
            levels = sorted(self.asks.keys())
            book = self.asks
        remaining = qty
        for p in levels:
            if remaining <= 0:
                break
            available = book[p]
            take = min(remaining, available)
            filled += take
            cost += take * p
            remaining -= take
            book[p] -= take
            if book[p] <= 0:
                del book[p]
        self._update_mid()
        if filled > 0:
            avg_price = cost / filled
            impact_cost = impact_factor * filled * (avg_price ** 0.5)
            self.last_trade = avg_price
            return filled, avg_price, impact_cost
        return 0, 0.0, 0.0


# -----------------------------------------------------------------------------
# Simulator: Poisson arrivals, simple impact
# -----------------------------------------------------------------------------

def simulate_order_flow(n_events=500, lambda_arrival=10.0, init_mid=100.0, tick=0.01, seed=42):
    """
    Simulate order flow: Poisson arrivals; each event is limit or market with prob.
    Returns (book, list of (t, side, type, price, qty, fill_info)).
    """
    if seed is not None:
        np.random.seed(seed)
    book = OrderBook(tick_size=tick)
    # Seed book
    for i in range(5):
        book.add_limit("BID", init_mid - (i + 1) * tick, 100)
        book.add_limit("ASK", init_mid + (i + 1) * tick, 100)
    book._update_mid()
    events = []
    t = 0.0
    for _ in range(n_events):
        dt = np.random.exponential(1.0 / lambda_arrival)
        t += dt
        side = "BID" if np.random.rand() < 0.5 else "ASK"
        is_market = np.random.rand() < 0.3
        qty = int(np.random.lognormal(2, 1)) + 1
        if is_market:
            filled, avg_p, impact = book.market_order("SELL" if side == "BID" else "BUY", qty, impact_factor=0.005)
            events.append((t, side, "MARKET", None, qty, (filled, avg_p, impact)))
        else:
            offset = np.random.uniform(1, 10) * tick * (1 if side == "BID" else -1)
            price = (book.mid or init_mid) + offset
            book.add_limit(side, price, qty)
            events.append((t, side, "LIMIT", price, qty, None))
    return book, events


# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------

def run_demo(n_events=300, seed=42):
    book, events = simulate_order_flow(n_events=n_events, seed=seed)
    market_evts = [e for e in events if e[2] == "MARKET" and e[5] is not None and e[5][0] > 0]
    print("Order book simulator demo")
    print("  Best bid:", book.best_bid(), " Best ask:", book.best_ask(), " Mid:", book.mid)
    print("  Events:", len(events), " Market fills:", len(market_evts))
    if market_evts:
        impacts = [e[5][2] for e in market_evts]
        print("  Total impact cost (sample):", sum(impacts))
    return book, events


if __name__ == "__main__":
    run_demo(n_events=300, seed=42)
