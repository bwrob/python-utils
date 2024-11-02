"""ExitStack mock script."""

from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from enum import StrEnum
from logging import getLogger
from uuid import uuid4

import pandas as pd

logger = getLogger(__name__)

PORTFOLIO = "portfolio_1"
PORTFOLIOS = ["portfolio_1", "portfolio_2", "portfolio_3"]


class MockObject(StrEnum):
    """Types of mock objects."""

    ANALYSIS = "analysis"
    BENCHMARK = "benchmark"
    OTC_PRODUCTS = "otc_products"
    PORTFOLIO = "portfolio"


def mock_object(object_type: MockObject) -> str:
    """Mock a UUID for a given object type.

    Args:
    ----
        object_type: Type of object.

    """
    return f"{object_type}_{uuid4()}"


def mock_preparation(object_type: MockObject, **kwargs: str) -> None:
    """Mock preparation of an object.

    Args:
    ----
        object_type: Type of object.
        **kwargs: Keyword arguments for the preparation.

    """
    msg = f"Preparing {object_type}" + (f" using {kwargs}" if kwargs else ".")
    logger.info(msg)


def mock_clean_up(object_uuid: str) -> None:
    """Mock clean up of an object.

    Args:
    ----
        object_uuid: Uuid of the object.

    """
    msg = f"Cleaning up after {object_uuid}."
    logger.info(msg)


def print_title(title: str) -> None:
    """Print a title padded, surrounded by dashes and empty lines."""
    msg = "\n" + title.center(60, "-") + "\n"
    logger.info(msg)


@contextmanager
def analysis(
    *,
    benchmark_uuid: str,
) -> Generator[str, None, None]:
    """Mock definition of an analysis.

    Example: equity delta and correlation with benchmark.

    Args:
    ----
        benchmark_uuid: Uuid of the benchmark.

    """
    mock_preparation(
        MockObject.ANALYSIS,
        benchmark_name=benchmark_uuid,
    )
    analysis_uuid = mock_object(MockObject.ANALYSIS)
    yield analysis_uuid
    mock_clean_up(analysis_uuid)


@contextmanager
def benchmark() -> Generator[str, None, None]:
    """Mock definition of a benchmark.

    Args:
    ----
        otc_products_uuid: Uuid of the otc products.

    """
    mock_preparation(
        MockObject.BENCHMARK,
    )
    benchmark_uuid = mock_object(MockObject.BENCHMARK)
    yield benchmark_uuid
    mock_clean_up(benchmark_uuid)


@contextmanager
def otc_products() -> Generator[str, None, None]:
    """Mock definition of an otc products.

    Args:
    ----
        otc_products_uuid: Uuid of the otc products.

    """
    mock_preparation(MockObject.OTC_PRODUCTS)
    otcs_uuid = mock_object(MockObject.OTC_PRODUCTS)
    yield otcs_uuid
    mock_clean_up(otcs_uuid)


@contextmanager
def portfolio(
    *,
    portfolio_name: str,
    otc_products_uuid: str,
) -> Generator[str, None, None]:
    """Mock definition of a portfolio.

    Args:
    ----
        portfolio_name: Name of the portfolio.
        otc_products_uuid: Uuid of the otc products.

    """
    mock_preparation(
        MockObject.PORTFOLIO,
        portfolio_name=portfolio_name,
        otc_products_uuid=otc_products_uuid,
    )
    portfolio_uuid = mock_object(MockObject.PORTFOLIO)
    yield portfolio_uuid
    mock_clean_up(portfolio_uuid)


def analysis_results(
    *,
    analysis_uuid: str,
    portfolio_uuid: str,
) -> pd.DataFrame:
    """Mock running the analysis on given portfolio.

    Returns empty dataframe.

    Args:
    ----
        analysis_uuid: Uuid of the analysis.
        portfolio_uuid: Uuid of the portfolio.

    """
    msg = f"Running analysis {analysis_uuid} on portfolio {portfolio_uuid}."
    logger.info(msg)
    return pd.DataFrame()


def run_analysis() -> pd.DataFrame:
    """Mock running the analysis using with clauses."""
    with (
        otc_products() as otc_uuid,
        benchmark() as benchmark_uuid,
        portfolio(
            portfolio_name=PORTFOLIO,
            otc_products_uuid=otc_uuid,
        ) as portfolio_uuid,
        analysis(
            benchmark_uuid=benchmark_uuid,
        ) as analysis_uuid,
    ):
        return analysis_results(
            analysis_uuid=analysis_uuid,
            portfolio_uuid=portfolio_uuid,
        )


def run_analysis_with_exit_stack() -> pd.DataFrame:
    """Mock running the analysis using exit stack."""
    with ExitStack() as stack:
        otc_uuid = stack.enter_context(otc_products())
        benchmark_uuid = stack.enter_context(benchmark())
        portfolio_uuid = stack.enter_context(
            portfolio(
                portfolio_name=PORTFOLIO,
                otc_products_uuid=otc_uuid,
            ),
        )
        analysis_uuid = stack.enter_context(
            analysis(
                benchmark_uuid=benchmark_uuid,
            ),
        )
        return analysis_results(
            analysis_uuid=analysis_uuid,
            portfolio_uuid=portfolio_uuid,
        )


def run_analysis_with_exit_stack_2(
    *,
    clean_up: bool = True,
) -> pd.DataFrame:
    """Mock running the analysis using exit stack.

    Args:
    ----
        clean_up: Whether to clean up after the objects.

    """
    with ExitStack() as stack:
        otc_uuid = stack.enter_context(otc_products())
        benchmark_uuid = stack.enter_context(benchmark())
        portfolio_uuid = stack.enter_context(
            portfolio(
                portfolio_name=PORTFOLIO,
                otc_products_uuid=otc_uuid,
            ),
        )
        analysis_uuid = stack.enter_context(
            analysis(
                benchmark_uuid=benchmark_uuid,
            ),
        )
        results = analysis_results(
            analysis_uuid=analysis_uuid,
            portfolio_uuid=portfolio_uuid,
        )

        if not clean_up:
            _ = stack.pop_all()
    return results


def run_analysis_with_exit_stack_3(
    *,
    clean_up: bool = True,
) -> pd.DataFrame:
    """Mock running the analysis for multiple portfolios using exit stack.

    Args:
    ----
        clean_up: Whether to clean up after the objects.

    """
    with ExitStack() as stack:
        otc_uuid = stack.enter_context(otc_products())
        benchmark_uuid = stack.enter_context(benchmark())
        portfolio_uuids = [
            stack.enter_context(
                portfolio(
                    portfolio_name=portfolio_name,
                    otc_products_uuid=otc_uuid,
                ),
            )
            for portfolio_name in PORTFOLIOS
        ]
        analysis_uuid = stack.enter_context(
            analysis(
                benchmark_uuid=benchmark_uuid,
            ),
        )
        result_parts = [
            analysis_results(
                analysis_uuid=analysis_uuid,
                portfolio_uuid=portfolio_uuid,
            )
            for portfolio_uuid in portfolio_uuids
        ]
        results = pd.concat(result_parts)

        if not clean_up:
            _ = stack.pop_all()
    return results


if __name__ == "__main__":
    print_title("Running analysis.")
    _ = run_analysis()

    print_title("Running analysis with exit stack.")
    _ = run_analysis_with_exit_stack()

    print_title("Running analysis with exit stack and no clean up.")
    _ = run_analysis_with_exit_stack_2(clean_up=False)

    print_title("Running analysis with exit stack on multiple portfolios.")
    _ = run_analysis_with_exit_stack_3(clean_up=True)
