"""Single-name CDS pricing engine following the ISDA standard model."""
from .calibration import CalibrationError, bootstrap_credit_curve
from .curves import CreditCurve, DiscountCurve
from .dates import (CouponPeriod, act360, act365, add_months, cash_settle_date,
                    cds_maturity, cds_schedule, step_in_date)
from .pricing import (CDS, CDSValuation, premium_leg_annuities, protection_leg_pv,
                      signed, value_cds)
from .quotes import (clean_price, conventional_spread_from_upfront,
                     upfront_from_conventional_spread)
from .risk import RiskReport, risk_report
from .simple import bootstrap_simple, df_log_linear, hazards_from_survival

__version__ = "0.1.0"
