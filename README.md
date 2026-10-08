# Single-name CDS pricing engine

A small, readable Python implementation of the ISDA standard model for single-name credit default swaps, together with a tutorial notebook that builds the engine one step at a time and explains why each step is done the way it is.

## What it does

| Layer | Module | Content |
|---|---|---|
| Dates | `cdsengine.dates` | 20th-of-quarter coupon dates, standard maturities, step-in and cash-settle dates, Act/360 and Act/365F |
| Curves | `cdsengine.curves` | Discount curve with piecewise-constant forwards; credit curve with piecewise-constant hazard rates |
| Pricing | `cdsengine.pricing` | Closed-form protection leg, coupon leg and accrual on default; par spread; clean and dirty value; upfront |
| Calibration | `cdsengine.calibration` | Hazard-curve bootstrap by root finding, with explicit errors for quotes that admit no arbitrage-free curve |
| Quotes | `cdsengine.quotes` | Conventional spread, upfront and price conversions |
| Engines | `cdsengine.engines` | `ConventionalSpreadEngine` (one quoted spread, flat curve) and `ParSpreadEngine` (par spread term structure), sharing one pricing core |
| Risk | `cdsengine.risk` | RPV01, parallel and bucketed CS01, IR01, Rec01, jump-to-default |
| Textbook model | `cdsengine.simple` | Annual-coupon bootstrap, kept as a reference point |

## Tutorial

[`notebooks/cds_engine_tutorial.ipynb`](notebooks/cds_engine_tutorial.ipynb) starts from the textbook annual model, shows where it departs from market practice, and then rebuilds every part of the engine inline. Each code cell is introduced by a **Step** (what the cell does) and a **Why** (the reason for the design choice), and each inline result is asserted equal to the packaged implementation.

## Quick start

```bash
pip install -e ".[dev]"
pytest -q
```

```python
from datetime import date
from cdsengine import CDS, ConventionalSpreadEngine, DiscountCurve, ParSpreadEngine

val = date(2020, 12, 14)
disc = DiscountCurve.flat(val, 0.0295)
cds = CDS(maturity=date(2025, 12, 20), coupon=0.01, recovery=0.40, notional=10_000_000)

# 1) You have the contract's own conventional spread
conv = ConventionalSpreadEngine(val, disc).price(cds, spread=0.0125)
print(conv.upfront, conv.price, conv.cash_settlement_amount, conv.cs01_amount)

# 2) You have the name's par spread curve
par_engine = ParSpreadEngine.from_tenors(val, disc, tenor_years=[1, 2, 3, 5],
                                         par_spreads=[0.0050, 0.0077, 0.0094, 0.0125],
                                         recovery=0.40)
par = par_engine.price(cds)
print(par.upfront, par.conventional_spread, par.cs01_bucket_amounts)
```

Use the conventional engine when a single spread is quoted for the contract itself; it is exact for that contract only. Use the par engine when tenor quotes are available; it values every contract on the name consistently and gives bucketed risk. The lower-level functions (`bootstrap_credit_curve`, `value_cds`, `risk_report`) remain available.

Per-unit results are from the protection buyer's point of view; fields ending in `_amount` apply the contract's notional and side.

## Validation

* **Textbook model**: reproduces the survival probabilities of the source course notebook to the printed digits, including the negative values for the distressed example, which the strict mode rejects.
* **Closed forms**: the protection leg and accrual on default are checked against brute-force numerical integration.
* **QuantLib**: on identical curves the legs, par spread and NPV match `IsdaCdsEngine` to machine precision when `quantlib_compat=True`. The default convention differs only in the survival observation date of the final coupon period, worth about 1e-9 of notional.
* **Market example**: the price and spread pairs in the Markit CDS Indices Primer trade example are reproduced to within a cent per 100 under a stated flat-rate assumption.

## Limitations

* Discount factors are inputs; the curve is not yet built from deposit and swap quotes.
* The calendar is weekends-only.
* Results have not been compared with the ISDA C library or a vendor calculator.
* Single-name contracts only.
