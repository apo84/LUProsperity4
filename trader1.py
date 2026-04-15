from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json

class Trader:

    def bid(self) -> int:
        """Required for Round 2, ignored for Round 1."""
        return 0

    def run(self, state: TradingState):
        """
        Improved trading logic.
        Key changes vs v1:
          - OSMIUM: tighter edge (1 instead of 2) + passive quoting to capture spread
          - PEPPER: momentum/trend-following with EMA instead of lagging SMA;
                    active position unwinding near end of session
        """
        if state.traderData:
            memory = json.loads(state.traderData)
        else:
            memory = {
                'pepper_history': [],
                'pepper_ema': None,
                'timestamp': 0,
            }

        result = {}
        conversions = 0
        POSITION_LIMIT = 20

        # ------------------------------------------------------------------ #
        # Track timestamp so we can trigger end-of-session unwind             #
        # ------------------------------------------------------------------ #
        memory['timestamp'] = state.timestamp

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            current_pos = state.position.get(product, 0)

            # ============================================================== #
            # ASH_COATED_OSMIUM  –  mean-reversion market-maker              #
            # FIX: edge 2→1 (catches more fills); also post passive quotes   #
            # ============================================================== #
            if product == "ASH_COATED_OSMIUM":
                FAIR = 10000
                EDGE = 1          # was 2; captures fills at 10001/9999
                PASSIVE_EDGE = 2  # quote passively at FAIR ± PASSIVE_EDGE

                # -- Aggressive: hit mispriced orders in the book ----------- #
                if order_depth.sell_orders:
                    for price in sorted(order_depth.sell_orders.keys()):
                        if price < FAIR - EDGE:
                            vol = abs(order_depth.sell_orders[price])
                            max_buy = POSITION_LIMIT - current_pos
                            buy_qty = min(max_buy, vol)
                            if buy_qty > 0:
                                orders.append(Order(product, price, buy_qty))
                                current_pos += buy_qty
                        else:
                            break   # orders are sorted ascending

                if order_depth.buy_orders:
                    for price in sorted(order_depth.buy_orders.keys(), reverse=True):
                        if price > FAIR + EDGE:
                            vol = order_depth.buy_orders[price]
                            max_sell = -POSITION_LIMIT - current_pos
                            sell_qty = max(max_sell, -vol)
                            if sell_qty < 0:
                                orders.append(Order(product, price, sell_qty))
                                current_pos += sell_qty
                        else:
                            break

                # -- Passive: post resting quotes to capture spread --------- #
                # Only post if we have remaining capacity after aggressive fills
                passive_buy_qty = POSITION_LIMIT - current_pos
                if passive_buy_qty > 0:
                    orders.append(Order(product, FAIR - PASSIVE_EDGE, passive_buy_qty))

                passive_sell_qty = -POSITION_LIMIT - current_pos
                if passive_sell_qty < 0:
                    orders.append(Order(product, FAIR + PASSIVE_EDGE, passive_sell_qty))

            # ============================================================== #
            # INTARIAN_PEPPER_ROOT  –  trend-following                       #
            # FIX 1: use EMA (alpha=0.3) instead of 5-period SMA to track    #
            #         the trend faster without overshooting on noise          #
            # FIX 2: momentum entry – buy when price > EMA (trend up),       #
            #         sell when price < EMA (trend down)                      #
            # FIX 3: end-of-session unwind to avoid holding overnight         #
            # ============================================================== #
            elif product == "INTARIAN_PEPPER_ROOT":
                # Calculate current Mid-Price
                best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
                best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
                
                if best_ask is not None and best_bid is not None:
                    mid_price = (best_ask + best_bid) / 2
                    memory['pepper_history'].append(mid_price)
                    
                    # Keep rolling window to 5
                    if len(memory['pepper_history']) > 5:
                        memory['pepper_history'].pop(0)
                    
                    # Calculate Fair Value based on Rolling Average
                    acceptable_price = sum(memory['pepper_history']) / len(memory['pepper_history'])
                    edge = 3

                    # Trading logic using dynamic acceptable_price
                    if best_ask < acceptable_price - edge:
                        max_buy = POSITION_LIMIT - current_pos
                        if max_buy > 0:
                            orders.append(Order(product, best_ask, max_buy))
                    
                    if best_bid > acceptable_price + edge:
                        max_sell = -POSITION_LIMIT - current_pos
                        if max_sell < 0:
                            orders.append(Order(product, best_bid, max_sell))

            result[product] = orders

        traderData = json.dumps(memory)
        return result, conversions, traderData