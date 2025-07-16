#!/usr/bin/env python
"""
Main script for running the FinSight AI system.
"""

import argparse
import logging
import sys
from datetime import datetime

from finsight.pipeline import PipelineManager
from finsight.config import DEFAULT_NEWS_QUERIES, DEFAULT_TICKERS, DEFAULT_DAYS_BACK

# Setup logging
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FinSight AI - Financial insights system")

    parser.add_argument(
        "--mode",
        choices=["collect", "analyze", "full"],
        default="full",
        help="Pipeline mode to run",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS_BACK,
        help=f"Number of days to look back (default: {DEFAULT_DAYS_BACK})",
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Ticker symbols to collect data for",
    )

    parser.add_argument(
        "--queries",
        nargs="+",
        default=DEFAULT_NEWS_QUERIES,
        help="News query terms",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main():
    """Run the FinSight AI system."""
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    start_time = datetime.now()
    logger.info(f"Starting FinSight AI at {start_time}")
    logger.info(f"Mode: {args.mode}")

    # Initialize pipeline
    pipeline = PipelineManager()

    # Run specified pipeline mode
    if args.mode == "collect":
        result = pipeline.run_data_collection_pipeline(
            news_queries=args.queries,
            tickers=args.tickers,
            days_back=args.days,
        )
        logger.info(f"Collected {len(result['news_data'])} news articles")
        logger.info(f"Collected market data for {len(result['market_data'])} tickers")

    elif args.mode == "analyze":
        logger.info("Analysis-only mode requires pre-collected data, which is not implemented yet")
        # TODO: Implement analysis-only mode

    elif args.mode == "full":
        result = pipeline.run_full_pipeline(
            news_queries=args.queries,
            tickers=args.tickers,
            days_back=args.days,
        )
        logger.info(f"Processed {len(result['news_data'])} news articles")
        logger.info(f"Processed market data for {len(result['market_data'])} tickers")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"FinSight AI completed in {duration} seconds")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running FinSight AI: {str(e)}", exc_info=True)
        sys.exit(1)