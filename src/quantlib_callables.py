"""QuantLib callables example."""

import multiprocessing as mp
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import batched, chain, count, takewhile

import QuantLib as ql  # noqa: N813
import tqdm
from tqdm.contrib.concurrent import process_map


@dataclass
class Bond:
    """A representation of a bond with an interest rate and an identifier.

    Attributes
    ----------
    rate : float
        The interest rate of the bond.
    id : int
        The unique identifier for the bond.

    """

    rate: float
    id: int


@dataclass
class InputData:
    """Input data for pricing callable bonds.

    Attributes
    ----------
    bonds : list[Bond]
        A list of Bond objects to be priced.

    """

    bonds: list[Bond]


def create_callable_bond(coupon_rate: float) -> ql.CallableFixedRateBond:
    """Create a callable fixed rate bond.

    This example is from the QuantLib examples/callablebond.cpp file.

    The example creates a callable fixed rate bond with the following
    characteristics:

    - Issue date: September 16, 2004
    - Maturity date: September 15, 2012
    - Tenor: Quarterly
    - Accrual convention: Unadjusted
    - Settlement days: 3
    - Face amount: 100
    - Accrual daycount: Actual/Actual (Bond)
    - Coupon rate: 2.5%
    - Callability schedule: 24 call dates with call price of 100.0

    The example then returns the callable fixed rate bond object.

    """
    callability_schedule = ql.CallabilitySchedule()
    call_price = 100.0
    call_date = ql.Date(15, ql.September, 2006)
    null_calendar = ql.NullCalendar()

    # Number of call dates is 24
    for _ in range(24):
        callability_price = ql.BondPrice(call_price, ql.BondPrice.Clean)
        callability_schedule.append(
            ql.Callability(callability_price, ql.Callability.Call, call_date),
        )
        call_date = null_calendar.advance(call_date, 3, ql.Months)

    issue_date = ql.Date(16, ql.September, 2004)
    maturity_date = ql.Date(15, ql.September, 2012)
    bond_calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    bond_tenor = ql.Period(ql.Quarterly)
    accrual_convention = ql.Unadjusted
    bond_schedule = ql.Schedule(
        issue_date,
        maturity_date,
        bond_tenor,
        bond_calendar,
        accrual_convention,
        accrual_convention,
        ql.DateGeneration.Backward,
        False,  # noqa: FBT003
    )

    settlement_days = 3
    face_amount = 100
    accrual_day_count = ql.ActualActual(ql.ActualActual.Bond)
    return ql.CallableFixedRateBond(
        settlement_days,
        face_amount,
        bond_schedule,
        [coupon_rate],
        accrual_day_count,
        ql.Following,
        face_amount,
        issue_date,
        callability_schedule,
    )


def yield_curve(calc_date: ql.Date) -> ql.YieldTermStructureHandle:
    """Construct a yield term structure handle.

    Parameters
    ----------
    calc_date : ql.Date
        The calculation date for the yield curve.

    Returns
    -------
    ql.YieldTermStructureHandle
        A handle to the yield term structure.

    """
    day_count = ql.ActualActual(ql.ActualActual.Bond)
    rate = 0.06
    term_structure = ql.FlatForward(
        calc_date,
        rate,
        day_count,
        ql.Compounded,
        ql.Semiannual,
    )
    return ql.RelinkableYieldTermStructureHandle(term_structure)


def hull_white_model(
    a: float,
    s: float,
    term_structure_handle: ql.YieldTermStructureHandle,
) -> ql.HullWhite:
    """Construct a Hull-White model.

    Parameters
    ----------
    a : float
        The mean reversion parameter of the Hull-White model.
    s : float
        The volatility parameter of the Hull-White model.
    term_structure_handle : YieldTermStructureHandle
        The yield term structure handle used for the Hull-White model.

    Returns
    -------
    model : HullWhite
        The Hull-White model.

    """
    return ql.HullWhite(term_structure_handle, a, s)


def engine(model: ql.HullWhite, grid_points: int) -> ql.TreeCallableFixedRateBondEngine:
    """Construct a pricing engine.

    Hull-White model and a TreeCallableFixedRateBondEngine that
    prices the callable bond.

    Parameters
    ----------
    model : HullWhite
        The Hull-White model to use for pricing.
    grid_points : int
        The number of grid points to use in the finite difference method.

    Returns
    -------
    engine : TreeCallableFixedRateBondEngine
        The pricing engine.

    """
    return ql.TreeCallableFixedRateBondEngine(model, grid_points)


def price_bond(
    bond: ql.CallableFixedRateBond,
    engine: ql.TreeCallableFixedRateBondEngine,
) -> float:
    """Price a callable bond with a given pricing engine.

    Parameters
    ----------
    bond : CallableFixedRateBond
        The callable bond to be priced.
    engine : TreeCallableFixedRateBondEngine
        The pricing engine to use.

    Returns
    -------
    float
        The clean price of the callable bond.

    """
    bond.setPricingEngine(engine)
    return bond.cleanPrice()


def price_with_setup(
    input_data: InputData,
    *,
    multiprocess: bool = True,
) -> list[float]:
    """Price a list of callable bonds using a Hull-White model.

    This function sets up the necessary QuantLib objects to price a list of
    callable bonds provided in the input data. It uses a specified evaluation
    date and constructs a yield curve, a Hull-White model, and a pricing engine
    to calculate the clean prices of the bonds.

    Parameters
    ----------
    input_data : InputData
        The input data containing the bonds to be priced.

    multiprocess : bool, optional
        Whether to disable the progress bar in multiprocess mode (default: True).

    Returns
    -------
    list[float]
        A list of clean prices for the callable bonds, rounded to three decimal
        places.

    """
    calc_date = ql.Date(16, 8, 2006)
    ql.Settings.instance().evaluationDate = calc_date

    ql_bonds = [create_callable_bond(bond.rate) for bond in input_data.bonds]
    term_structure = yield_curve(calc_date)
    model = hull_white_model(0.5, 0.05, term_structure)
    tree_engine = engine(model, 500)
    return [
        round(price_bond(bond, tree_engine), 3)
        for bond in tqdm.tqdm(ql_bonds, disable=multiprocess)
    ]


def main(rates: Iterable[float], *, multiprocess: bool = False) -> None:
    """Price a list of callable bonds using a Hull-White model.

    This function takes an iterable of interest rates and prices a list of
    callable bonds with the corresponding rates. It uses a specified evaluation
    date and constructs a yield curve, a Hull-White model, and a pricing engine
    to calculate the clean prices of the bonds. The function can be run in
    either single-process mode or multiprocess mode.

    Parameters
    ----------
    rates : Iterable[float]
        An iterable of interest rates.
    multiprocess : bool, optional
        Whether to run the function in multiprocess mode (default: False).

    """
    bonds = [Bond(rate, i) for i, rate in enumerate(rates)]

    print("Multiprocess:", multiprocess)
    if not multiprocess:
        results = price_with_setup(InputData(bonds), multiprocess=False)
    else:
        batch_size = 100
        batches = [
            InputData(list(batch))
            for batch in list(batched(bonds, batch_size, strict=False))
        ]
        results = process_map(
            price_with_setup,
            batches,
            max_workers=min(32, mp.cpu_count() + 4),
        )
        results = list(chain.from_iterable(results))
    print(results[0], results[-1])


if __name__ == "__main__":
    MAX_RATE = 0.15

    main(
        rates=takewhile(
            lambda x: x < MAX_RATE,
            count(0.03, 0.00001),
        ),
        multiprocess=True,
    )
