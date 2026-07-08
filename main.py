#!/usr/bin/env python3
"""
Polymarket Arbitrage Trading Bot
=================================

Main entry point for the trading bot.

Usage:
    python main.py                      # Run in dry-run mode (default)
    python main.py --live               # Run in live mode
    python main.py --backtest           # Run backtest
    python main.py --config my.yaml     # Use custom config file
"""

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # must run before importing modules that read env vars at import time (e.g. telegram_notifier)

from polymarket_client import PolymarketClient
from kalshi_client import KalshiClient
from core.data_feed import DataFeed
from core.arb_engine import ArbEngine, ArbConfig
from core.execution import ExecutionEngine, ExecutionConfig
from core.risk_manager import RiskManager, RiskConfig
from core.portfolio import Portfolio
from core.cross_platform_arb import CrossPlatformArbEngine, match_markets_sync
from utils.config_loader import load_config, BotConfig
from utils.logging_utils import setup_logging, performance_logger

try:
    from utils.telegram_notifier import notify_cross_platform_opportunity
except Exception:  # pragma: no cover - telegram is optional
    async def notify_cross_platform_opportunity(*args, **kwargs):
        return None


logger = logging.getLogger(__name__)


class TradingBot:
    """
    Main trading bot orchestrator.
    
    Coordinates all components and manages the trading lifecycle.
    """
    
    def __init__(self, config: BotConfig):
        self.config = config
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Components (initialized in start())
        self.client: Optional[PolymarketClient] = None
        self.data_feed: Optional[DataFeed] = None
        self.arb_engine: Optional[ArbEngine] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.risk_manager: Optional[RiskManager] = None
        self.portfolio: Optional[Portfolio] = None

        # Components - Kalshi (cross-platform arbitrage)
        self.kalshi_client: Optional[KalshiClient] = None
        self.cross_platform_engine: Optional[CrossPlatformArbEngine] = None
        self.market_matcher = None
        self._kalshi_markets: list = []
        self._matched_pairs: list = []

        # Statistics
        self._start_time: Optional[datetime] = None
        self._update_count = 0
        self._signal_count = 0
        self._cross_platform_count = 0
    
    async def start(self) -> None:
        """Initialize and start all components."""
        logger.info("=" * 60)
        logger.info("Polymarket Arbitrage Bot Starting")
        logger.info("=" * 60)
        logger.info(f"Mode: {'DRY RUN' if self.config.is_dry_run else 'LIVE'}")
        logger.info(f"Markets: {self.config.trading.markets or 'Auto-discover'}")
        
        self._start_time = datetime.utcnow()
        self._running = True
        
        # Initialize API client
        self.client = PolymarketClient(
            rest_url=self.config.api.polymarket_rest_url,
            ws_url=self.config.api.polymarket_ws_url,
            gamma_url=self.config.api.gamma_api_url,
            api_key=self.config.api.api_key,
            api_secret=self.config.api.api_secret,
            private_key=self.config.api.private_key,
            timeout=self.config.api.timeout_seconds,
            max_retries=self.config.api.max_retries,
            retry_delay=self.config.api.retry_delay_seconds,
            dry_run=self.config.is_dry_run,
        )
        await self.client.connect()

        # Initialize Kalshi client + cross-platform arbitrage engine
        if self.config.mode.cross_platform_enabled and self.config.mode.kalshi_enabled:
            logger.info("Cross-platform arbitrage ENABLED (Polymarket <-> Kalshi)")
            self.kalshi_client = KalshiClient(
                timeout=self.config.api.timeout_seconds,
                max_retries=self.config.api.max_retries,
                dry_run=self.config.is_dry_run,
            )
            # User requirement: only surface opportunities with >2% net edge.
            self.cross_platform_engine = CrossPlatformArbEngine(
                min_edge=0.02,
            )
            self.market_matcher = self.cross_platform_engine.matcher
            logger.info(
                f"Cross-platform min edge: {self.cross_platform_engine.min_edge:.1%}"
            )
        else:
            logger.info("Cross-platform arbitrage DISABLED")

        # Initialize portfolio
        initial_balance = (
            self.config.mode.dry_run_initial_balance 
            if self.config.is_dry_run 
            else 0.0
        )
        self.portfolio = Portfolio(initial_balance=initial_balance)
        
        # Initialize risk manager
        self.risk_manager = RiskManager(RiskConfig(
            max_position_per_market=self.config.risk.max_position_per_market,
            max_global_exposure=self.config.risk.max_global_exposure,
            max_daily_loss=self.config.risk.max_daily_loss,
            max_drawdown_pct=self.config.risk.max_drawdown_pct,
            trade_only_high_volume=self.config.risk.trade_only_high_volume,
            min_24h_volume=self.config.risk.min_24h_volume,
            whitelist=self.config.risk.whitelist,
            blacklist=self.config.risk.blacklist,
            kill_switch_enabled=self.config.risk.kill_switch_enabled,
            auto_unwind_on_breach=self.config.risk.auto_unwind_on_breach,
        ))
        
        # Initialize execution engine
        self.execution_engine = ExecutionEngine(
            client=self.client,
            risk_manager=self.risk_manager,
            portfolio=self.portfolio,
            config=ExecutionConfig(
                slippage_tolerance=self.config.trading.slippage_tolerance,
                order_timeout_seconds=self.config.trading.order_timeout_seconds,
                dry_run=self.config.is_dry_run,
            ),
        )
        await self.execution_engine.start()
        
        # Initialize arbitrage engine
        self.arb_engine = ArbEngine(ArbConfig(
            min_edge=self.config.trading.min_edge,
            bundle_arb_enabled=self.config.trading.bundle_arb_enabled,
            min_spread=self.config.trading.min_spread,
            mm_enabled=self.config.trading.mm_enabled,
            tick_size=self.config.trading.tick_size,
            default_order_size=self.config.trading.default_order_size,
            min_order_size=self.config.trading.min_order_size,
            max_order_size=self.config.trading.max_order_size,
        ))
        
        # Initialize data feed
        market_ids = self.config.trading.markets.copy()
        self.data_feed = DataFeed(
            client=self.client,
            market_ids=market_ids,
            position_refresh_interval=5.0,
            on_update=self._on_market_update,
            config=self.config,
        )
        await self.data_feed.start()
        
        # Wait for initial data
        logger.info("Waiting for market data...")
        if not await self.data_feed.wait_for_data(timeout=30.0):
            logger.warning("Timeout waiting for initial data, proceeding anyway")
        
        logger.info("Bot started successfully!")
        logger.info("-" * 60)
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())

        # Start cross-platform (Polymarket <-> Kalshi) arbitrage monitoring
        if self.kalshi_client and self.cross_platform_engine:
            asyncio.create_task(self._start_kalshi_monitoring())

        # Start fill simulation for dry run
        if self.config.is_dry_run and self.config.mode.simulate_fills:
            asyncio.create_task(self._simulate_fills())
    
    def _on_market_update(self, market_id: str, market_state) -> None:
        """Callback for market state updates."""
        self._update_count += 1
        
        # Check risk limits
        if not self.risk_manager.within_global_limits():
            logger.warning("Risk limits exceeded, skipping analysis")
            return
        
        # Analyze for opportunities
        signals = self.arb_engine.analyze(market_state)
        
        for signal in signals:
            self._signal_count += 1
            # Submit signal asynchronously
            asyncio.create_task(self.execution_engine.submit_signal(signal))
    
    async def _monitoring_loop(self) -> None:
        """Periodic monitoring and logging."""
        interval = self.config.monitoring.snapshot_interval
        
        while self._running:
            try:
                await asyncio.sleep(interval)
                
                # Log portfolio snapshot
                pnl = self.portfolio.get_pnl()
                exposure = self.portfolio.get_total_exposure()
                positions = len(self.portfolio.get_all_positions())
                open_orders = self.execution_engine.open_order_count
                
                performance_logger.log_snapshot(pnl, exposure, positions, open_orders)
                
                # Update risk manager
                self.risk_manager.update_pnl(
                    pnl["realized_pnl"],
                    pnl["unrealized_pnl"]
                )
                
                # Log statistics
                arb_stats = self.arb_engine.get_stats()
                exec_stats = self.execution_engine.get_stats()
                risk_summary = self.risk_manager.get_summary()
                
                xplat_info = ""
                if self.cross_platform_engine:
                    xplat_info = (
                        f" | XPlatform: {len(self._matched_pairs)} pairs, "
                        f"{self._cross_platform_count} arbs"
                    )

                logger.info(
                    f"Stats | Updates: {self._update_count} | "
                    f"Signals: {self._signal_count} | "
                    f"Orders: {exec_stats.orders_placed} placed, {exec_stats.orders_filled} filled | "
                    f"PnL: ${pnl['total_pnl']:.2f}"
                    f"{xplat_info}"
                )
                
                if risk_summary["kill_switch_triggered"]:
                    logger.critical("KILL SWITCH ACTIVE - Trading halted")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    async def _start_kalshi_monitoring(self) -> None:
        """
        Load Kalshi markets, match them against Polymarket markets, then
        continuously watch matched pairs for cross-platform arbitrage.
        """
        if not self.kalshi_client:
            return

        # Scale caps: matching is O(poly x kalshi). Runs in a separate process
        # (see _run_matching) so it no longer blocks the orderbook stream,
        # which lets us cover more markets than the original conservative cap.
        KALSHI_MAX = 4000
        POLY_MATCH_CAP = 1200
        REMATCH_INTERVAL = 7200.0  # refresh matches every 2h to catch new markets

        try:
            async with self.kalshi_client:
                # Fetch Kalshi markets
                logger.info("Fetching Kalshi markets...")
                self._kalshi_markets = await self.kalshi_client.list_all_markets(
                    status="open",
                    max_markets=KALSHI_MAX,
                )
                logger.info(f"Loaded {len(self._kalshi_markets)} Kalshi markets")

                if not self._kalshi_markets:
                    logger.warning("No Kalshi markets loaded; cross-platform arbitrage disabled")
                    return

                # Wait for Polymarket markets to be available
                logger.info("Waiting for Polymarket markets before matching...")
                for i in range(30):
                    await asyncio.sleep(1)
                    poly_count = len(self.data_feed._markets) if self.data_feed else 0
                    if poly_count >= 50:
                        logger.info(f"Got {poly_count} Polymarket markets - starting matching!")
                        break
                    if i % 5 == 0:
                        logger.info(f"Polymarket: {poly_count} markets loaded...")

                if not self.data_feed or not self.data_feed._markets:
                    logger.warning("No Polymarket markets available; skipping matching")
                    return

                # Match markets between platforms in a separate process so the
                # CPU-bound matching (hundreds of thousands of comparisons)
                # runs at full speed without starving the main event loop.
                polymarket_markets = list(self.data_feed._markets.values())
                if len(polymarket_markets) > POLY_MATCH_CAP:
                    logger.info(
                        f"Capping Polymarket markets for matching: "
                        f"{len(polymarket_markets)} -> {POLY_MATCH_CAP}"
                    )
                    polymarket_markets = polymarket_markets[:POLY_MATCH_CAP]
                logger.info(
                    f"Matching {len(polymarket_markets)} Polymarket x "
                    f"{len(self._kalshi_markets)} Kalshi markets (separate process)..."
                )
                self._matched_pairs = await self._run_matching(polymarket_markets)
                logger.info(f"Matching complete! Found {len(self._matched_pairs)} pairs")

                if not self._matched_pairs:
                    logger.warning("No matched pairs found; nothing to watch for cross-platform arb")
                    return

                # Watch matched pairs for price divergence, concurrently with
                # periodic re-matching (to pick up newly listed markets).
                # Both run inside the `async with` so the Kalshi client stays
                # open for the lifetime of the bot.
                logger.info("Starting cross-platform arbitrage watcher...")
                watcher_task = asyncio.create_task(self._watch_cross_platform_arbitrage())
                try:
                    await self._periodic_rematch_loop(
                        interval=REMATCH_INTERVAL,
                        kalshi_max=KALSHI_MAX,
                        poly_cap=POLY_MATCH_CAP,
                    )
                finally:
                    watcher_task.cancel()

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(f"Kalshi monitoring error: {e}")

    async def _periodic_rematch_loop(
        self, interval: float, kalshi_max: int, poly_cap: int
    ) -> None:
        """
        Periodically refresh the Kalshi market list and re-run matching so
        newly listed markets on either platform get picked up over time.
        Replaces the matcher's pair cache on each successful refresh.
        """
        while self._running:
            await asyncio.sleep(interval)
            try:
                logger.info("Refreshing Kalshi markets for re-matching...")
                kalshi_markets = await self.kalshi_client.list_all_markets(
                    status="open",
                    max_markets=kalshi_max,
                )
                if not kalshi_markets:
                    logger.warning("Re-match: no Kalshi markets returned, keeping existing pairs")
                    continue
                self._kalshi_markets = kalshi_markets

                polymarket_markets = list(self.data_feed._markets.values()) if self.data_feed else []
                if len(polymarket_markets) > poly_cap:
                    polymarket_markets = polymarket_markets[:poly_cap]

                logger.info(
                    f"Re-matching {len(polymarket_markets)} Polymarket x "
                    f"{len(self._kalshi_markets)} Kalshi markets..."
                )
                new_pairs = await self._run_matching(polymarket_markets)
                self._matched_pairs = new_pairs
                logger.info(f"Re-match complete! Now watching {len(new_pairs)} pairs")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Re-match failed, keeping existing pairs: {e}")

    async def _run_matching(self, polymarket_markets: list) -> list:
        """
        Run the CPU-bound market matching in a separate process so it runs
        at full speed on its own core, free of GIL contention with the main
        event loop (which is busy streaming Polymarket orderbooks).

        The matched pairs are copied back into this process's matcher cache
        so the arbitrage watcher can see them via get_cached_pairs().
        """
        import concurrent.futures

        loop = asyncio.get_event_loop()

        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                pairs = await loop.run_in_executor(
                    executor,
                    match_markets_sync,
                    polymarket_markets,
                    self._kalshi_markets,
                    self.market_matcher.min_similarity,
                )
        except Exception as e:
            # If the process pool is unavailable for any reason, fall back to
            # in-process matching so the bot still works (just slower).
            logger.warning(f"Process-pool matching failed ({e}); falling back in-process")
            pairs = await self.market_matcher.find_matches(
                polymarket_markets, self._kalshi_markets
            )

        # Replace the local matcher cache (the subprocess had its own) so a
        # re-match doesn't leave stale pairs from markets that no longer match.
        self.market_matcher._matched_pairs = {pair.pair_id: pair for pair in pairs}

        return pairs

    async def _check_pair_for_arbitrage(
        self,
        pair,
        notified_recently: dict,
        semaphore: asyncio.Semaphore,
        renotify_cooldown: float,
    ) -> None:
        """Fetch both order books for a pair and notify on a >min_edge opportunity."""
        async with semaphore:
            try:
                poly_ob = await self.client.get_orderbook(pair.polymarket_id)
                kalshi_ob = await self.kalshi_client.get_orderbook_unified(pair.kalshi_ticker)

                if not poly_ob or not kalshi_ob:
                    return

                opp = self.cross_platform_engine.check_arbitrage(
                    market_pair=pair,
                    polymarket_ob=poly_ob,
                    kalshi_ob=kalshi_ob,
                )

                if opp is None:
                    return

                now = asyncio.get_event_loop().time()
                last_notified = notified_recently.get(pair.pair_id, 0)
                if now - last_notified < renotify_cooldown:
                    return
                notified_recently[pair.pair_id] = now

                self._cross_platform_count += 1
                logger.info(
                    f"CROSS-PLATFORM ARB FOUND: {opp.token} | "
                    f"Buy {opp.buy_platform} @ {opp.buy_price:.3f} | "
                    f"Sell {opp.sell_platform} @ {opp.sell_price:.3f} | "
                    f"Net edge: {opp.edge_pct:.2%}"
                )

                asyncio.create_task(notify_cross_platform_opportunity(
                    question=pair.polymarket_question,
                    token=opp.token,
                    buy_platform=opp.buy_platform,
                    buy_price=opp.buy_price,
                    sell_platform=opp.sell_platform,
                    sell_price=opp.sell_price,
                    edge_pct=opp.edge_pct,
                    suggested_size=opp.suggested_size,
                ))

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Error checking pair {pair.pair_id}: {e}")

    async def _watch_cross_platform_arbitrage(
        self, poll_interval: float = 15.0, max_concurrent: int = 10
    ) -> None:
        """
        Continuously check matched market pairs for real cross-platform
        arbitrage (price divergence between Polymarket and Kalshi).

        Pairs are checked concurrently (bounded by max_concurrent) so a
        growing pair count doesn't stretch a poll cycle past poll_interval.
        Notifies Telegram whenever a net edge >= the engine's min_edge is
        found, de-duped so the same pair isn't spammed every poll.
        """
        notified_recently: dict[str, float] = {}
        RENOTIFY_COOLDOWN = 300  # seconds
        semaphore = asyncio.Semaphore(max_concurrent)

        while self._running:
            try:
                pairs = self.market_matcher.get_cached_pairs()
                if not pairs:
                    await asyncio.sleep(poll_interval)
                    continue

                await asyncio.gather(*(
                    self._check_pair_for_arbitrage(pair, notified_recently, semaphore, RENOTIFY_COOLDOWN)
                    for pair in pairs
                ))

                await asyncio.sleep(poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cross-platform arbitrage watcher: {e}")
                await asyncio.sleep(poll_interval)

    async def _simulate_fills(self) -> None:
        """Simulate order fills in dry run mode."""
        import random
        
        while self._running:
            try:
                await asyncio.sleep(2.0)  # Check every 2 seconds
                
                # Get open orders
                orders = self.execution_engine.get_open_orders()
                
                for order in orders:
                    # Random chance of fill
                    if random.random() < self.config.mode.fill_probability:
                        trade = self.client.simulate_fill(order.order_id)
                        if trade:
                            self.execution_engine.handle_fill(trade)
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fill simulation error: {e}")
    
    async def stop(self) -> None:
        """Stop all components gracefully."""
        logger.info("Shutting down...")
        self._running = False
        
        if self.data_feed:
            await self.data_feed.stop()
        
        if self.execution_engine:
            await self.execution_engine.stop()
        
        if self.client:
            await self.client.disconnect()
        
        # Final summary
        if self.portfolio:
            summary = self.portfolio.get_summary()
            logger.info("=" * 60)
            logger.info("Final Portfolio Summary")
            logger.info("=" * 60)
            logger.info(f"Total PnL: ${summary['pnl']['total_pnl']:.2f}")
            logger.info(f"  Realized: ${summary['pnl']['realized_pnl']:.2f}")
            logger.info(f"  Unrealized: ${summary['pnl']['unrealized_pnl']:.2f}")
            logger.info(f"Total Trades: {summary['total_trades']}")
            logger.info(f"Win Rate: {summary['win_rate']:.1%}")
            logger.info(f"Total Volume: ${summary['total_volume']:.2f}")
        
        if self.arb_engine:
            stats = self.arb_engine.get_stats()
            logger.info("-" * 60)
            logger.info(f"Bundle Opportunities: {stats.bundle_opportunities_detected}")
            logger.info(f"MM Opportunities: {stats.mm_opportunities_detected}")
            logger.info(f"Signals Generated: {stats.signals_generated}")
        
        logger.info("=" * 60)
        logger.info("Bot stopped")
        
        self._shutdown_event.set()
    
    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()


async def run_backtest(config: BotConfig, duration: float = 300.0) -> None:
    """Run a backtest simulation."""
    from utils.backtest import BacktestConfig, BacktestEngine, run_backtest as _run_backtest
    
    logger.info("Starting backtest mode...")
    
    # Create components
    portfolio = Portfolio(initial_balance=config.mode.dry_run_initial_balance)
    
    risk_manager = RiskManager(RiskConfig(
        max_position_per_market=config.risk.max_position_per_market,
        max_global_exposure=config.risk.max_global_exposure,
        max_daily_loss=config.risk.max_daily_loss,
        max_drawdown_pct=config.risk.max_drawdown_pct,
    ))
    
    arb_engine = ArbEngine(ArbConfig(
        min_edge=config.trading.min_edge,
        bundle_arb_enabled=config.trading.bundle_arb_enabled,
        min_spread=config.trading.min_spread,
        mm_enabled=config.trading.mm_enabled,
        tick_size=config.trading.tick_size,
        default_order_size=config.trading.default_order_size,
    ))
    
    # Use placeholder client for execution
    client = PolymarketClient(dry_run=True)
    await client.connect()
    
    execution_engine = ExecutionEngine(
        client=client,
        risk_manager=risk_manager,
        portfolio=portfolio,
        config=ExecutionConfig(dry_run=True),
    )
    await execution_engine.start()
    
    # Run backtest
    backtest_config = BacktestConfig(
        initial_balance=config.mode.dry_run_initial_balance,
        simulate_fills=True,
        fill_probability=config.mode.fill_probability,
    )
    
    # Generate market IDs
    market_ids = config.trading.markets or [f"market_{i}" for i in range(3)]
    
    result = await _run_backtest(
        config=backtest_config,
        market_ids=market_ids,
        arb_engine=arb_engine,
        execution_engine=execution_engine,
        risk_manager=risk_manager,
        portfolio=portfolio,
        duration_seconds=duration,
    )
    
    await execution_engine.stop()
    await client.disconnect()
    
    return result


async def main_async(args: argparse.Namespace) -> None:
    """Async main function."""
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
    
    # Override mode from command line
    if args.live:
        config.mode.trading_mode = "live"
    elif args.dry_run:
        config.mode.trading_mode = "dry_run"
    
    # Run backtest if requested
    if args.backtest:
        await run_backtest(config, duration=args.backtest_duration)
        return
    
    # Create and run the bot
    bot = TradingBot(config)
    
    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(bot.stop())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass
    
    try:
        await bot.start()
        await bot.wait_for_shutdown()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        await bot.stop()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        await bot.stop()
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Polymarket Arbitrage Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Run in dry-run mode
  python main.py --live             Run in live trading mode
  python main.py --backtest         Run backtest simulation
  python main.py -c custom.yaml     Use custom config file
        """
    )
    
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live trading mode"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Run in dry-run mode (default)"
    )
    
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest simulation"
    )
    
    parser.add_argument(
        "--backtest-duration",
        type=float,
        default=300.0,
        help="Backtest duration in simulated seconds (default: 300)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(console_level=log_level)
    
    # Run the async main
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nShutdown complete.")


if __name__ == "__main__":
    main()

