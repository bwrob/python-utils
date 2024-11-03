"""QuantLib callables example"""

import multiprocessing as mp
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import batched, chain, count, takewhile

import QuantLib as ql
import tqdm
from tqdm.contrib.concurrent import process_map


@dataclass
class Bond:
    rate: float
    id: int


@dataclass
class InputData:
    bonds: list[Bond]


def main(rates: Iterable[float], *, multiprocess: bool = False):
    if not multiprocess:
        results = price_with_setup(rates)
    else:
        batch_size = 100
        batches = batched(rates, batch_size)
        results = process_map(price_with_setup, batches, total=7_0, max_workers=8)
        results = list(chain.from_iterable(results))
    print(results[0], results[-1])


def callable_bond(rate: float) -> ql.CallableFixedRateBond:
    """Callable fixed rate bond example.

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
    callabilitySchedule = ql.CallabilitySchedule()
    callPrice = 100.0
    callDate = ql.Date(15, ql.September, 2006)
    nc = ql.NullCalendar()

    # Number of calldates is 24
    for _i in range(24):
        callabilityPrice = ql.BondPrice(callPrice, ql.BondPrice.Clean)
        callabilitySchedule.append(
            ql.Callability(callabilityPrice, ql.Callability.Call, callDate),
        )
        callDate = nc.advance(callDate, 3, ql.Months)

    issueDate = ql.Date(16, ql.September, 2004)
    maturityDate = ql.Date(15, ql.September, 2012)
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    tenor = ql.Period(ql.Quarterly)
    accrualConvention = ql.Unadjusted
    schedule = ql.Schedule(
        issueDate,
        maturityDate,
        tenor,
        calendar,
        accrualConvention,
        accrualConvention,
        ql.DateGeneration.Backward,
        False,
    )

    settlement_days = 3
    faceAmount = 100
    accrual_daycount = ql.ActualActual(ql.ActualActual.Bond)
    coupon = rate
    return ql.CallableFixedRateBond(
        settlement_days,
        faceAmount,
        schedule,
        [coupon],
        accrual_daycount,
        ql.Following,
        faceAmount,
        issueDate,
        callabilitySchedule,
    )


def yield_curve(calcDate: ql.Date) -> ql.YieldTermStructureHandle:
    dayCount = ql.ActualActual(ql.ActualActual.Bond)
    rate = 0.06
    termStructure = ql.FlatForward(
        calcDate,
        rate,
        dayCount,
        ql.Compounded,
        ql.Semiannual,
    )
    return ql.RelinkableYieldTermStructureHandle(termStructure)


def hull_white_model(
    a: float,
    s: float,
    term_structure_handle: ql.YieldTermStructureHandle,
):
    """Construct a Hull-White model.

    Parameters
    ----------
    a : float
        The mean reversion parameter of the Hull-White model.
    s : float
        The volatility parameter of the Hull-White model.

    Returns
    -------
    model : HullWhite
        The Hull-White model.

    """
    return ql.HullWhite(term_structure_handle, a, s)


def engine(model: ql.HullWhite, grid_points: int):
    """Construct a pricing engine.

    Hull-White model and a TreeCallableFixedRateBondEngine that
    prices the callable bond.

    Parameters
    ----------
    a : float
        The mean reversion parameter of the Hull-White model.
    s : float
        The volatility parameter of the Hull-White model.
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

    Returns
    -------
    list[float]
        A list of clean prices for the callable bonds, rounded to three decimal
        places.

    """
    calcDate = ql.Date(16, 8, 2006)
    ql.Settings.instance().evaluationDate = calcDate

    ql_bonds = [callable_bond(bond.rate) for bond in input_data.bonds]
    term_structure = yield_curve(calcDate)
    model = hull_white_model(0.5, 0.05, term_structure)
    tree_engine = engine(model, 500)
    return [
        round(price_bond(bond, tree_engine), 3)
        for bond in tqdm.tqdm(ql_bonds, disable=multiprocess)
    ]


def main(rates: Iterable[float], *, multiprocess: bool = False):
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
        batches = [InputData(list(batch)) for batch in list(batched(bonds, batch_size))]
        results = process_map(
            price_with_setup,
            batches,
            max_workers=min(32, mp.cpu_count() + 4),
        )
        results = list(chain.from_iterable(results))
    print(results[0], results[-1])


if __name__ == "__main__":
    main(
        rates=takewhile(
            lambda x: x < 0.15,
            count(0.03, 0.00001),
        ),
        multiprocess=True,
    )
