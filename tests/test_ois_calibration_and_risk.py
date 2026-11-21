"""Tests for the HW8 pieces: annuity and par rate, the par-rate gradient, the calibration Jacobian and the
state-to-quote risk transform, checked against hand sums, finite differences and bump-and-recalibrate.

    python tests/test_ois_calibration_and_risk.py          # plain runner
    python -m pytest tests -q                              # with pytest installed
"""
import os
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fixedincomelib import (  # noqa: E402
    qfCreateBuildMethod, qfCreateCashflowsReport, qfCreateData1D, qfCreateDataCollection, qfCreateDataGeneric,
    qfCreateModel, qfCreateModelBuildMethodCollection, qfCreateProductFromDataConvention,
    qfCreateValuationParameters, qfCreateValuationParametersCollection, qfCreateValueReport,
)
from fixedincomelib.date import accrued  # noqa: E402
from fixedincomelib.valuation import ValuationEngineProductRegistry, ValuationRequest  # noqa: E402

VALUE_DATE = "2026-10-09"
TENORS = ["1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "12Y", "15Y", "20Y", "25Y", "30Y", "40Y", "50Y"]
QUOTES_BP = [447.795, 465.672, 470.730, 472.032, 473.200, 474.988, 477.000, 479.203, 481.600, 484.300,
             489.725, 496.507, 501.255, 498.939, 493.568, 480.540, 466.540]
NOTIONAL = 1_000_000
# the professor's reference outputs shipped in the notebook
REFERENCE_PV = -24829.03
REFERENCE_RISK = [1.336135, 0.382009, 0.597304, 1.038225, 0.823435, 1.766412, 1.566545, 3.531423,
                  -355.339457, -401.432157, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
REFERENCE_HEDGE = [13666.089735, 2011.495956, 2150.660087, 2869.776317, 1863.365811, 3402.622933, 2647.663054,
                   5344.724260, -489162.044569, -508671.426672, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
REFERENCE_OWN_RISK = [-97.770067, -189.912728, -277.730519, -361.778939, -441.907361, -519.132398, -591.670793,
                      -660.730533, -726.424834, -789.177720, -905.487777, -1057.514068, -1264.029757,
                      -1425.446493, -1554.292696, -1747.511351, -1886.178011]


def market(bumps=None):
    quotes = np.array(QUOTES_BP) / 1e4
    if bumps:
        for tenor, size in bumps.items():
            quotes[TENORS.index(tenor)] += size
    return pd.DataFrame({"values": quotes}, index=pd.Index(TENORS, name="Tenor"))


def build(bumps=None):
    funding_parameters = qfCreateDataGeneric("DATA GENERIC", "USD-FUNDING", pd.DataFrame([
        ["OVERNIGHT INDEX SWAP", "USD-SOFR-OIS", "SOFR-1B-FLAT"],
        ["SPREAD ZERO RATE", "SOFR-1B-FLAT-OVER-SOFR-1B-ZERO-SPREAD", "SOFR-1B-FLAT"],
    ], columns=["DATA TYPE", "DATA CONVENTION", "FUNDING IDENTIFIER"]))
    build_methods = qfCreateModelBuildMethodCollection([
        qfCreateBuildMethod("YIELD_CURVE_COMMON", {"TARGET": "USD", "FUNDING PARAMETERS": "USD-FUNDING", "SOLVER": "BRENTQ"}),
        qfCreateBuildMethod("YIELD_CURVE_INDEX", {"TARGET": "SOFR-1B", "OVERNIGHT INDEX SWAP": "USD-SOFR-OIS",
                                                  "INTERPOLATION METHOD": "PIECEWISE_CONSTANT_LEFT_CONTINUOUS", "EXTRAPOLATION METHOD": "FLAT"}),
        qfCreateBuildMethod("YIELD_CURVE_FUNDING", {"TARGET": "SOFR-1B-FLAT", "SPREAD ZERO RATE": "SOFR-1B-FLAT-OVER-SOFR-1B-ZERO-SPREAD",
                                                    "INTERPOLATION METHOD": "PIECEWISE_CONSTANT_LEFT_CONTINUOUS", "EXTRAPOLATION METHOD": "FLAT"}),
    ])
    data = qfCreateDataCollection([
        qfCreateData1D("OVERNIGHT INDEX SWAP", "USD-SOFR-OIS", market(bumps)),
        qfCreateData1D("SPREAD ZERO RATE", "SOFR-1B-FLAT-OVER-SOFR-1B-ZERO-SPREAD", pd.DataFrame({"values": [0.0]}, index=["1Y"])),
        funding_parameters,
    ])
    return qfCreateModel(VALUE_DATE, "YIELD_CURVE", data, build_methods)


@lru_cache(maxsize=None)
def base_model():
    return build()


def params():
    return qfCreateValuationParametersCollection([qfCreateValuationParameters("FUNDING INDEX PARAMETER", {"FUNDING INDEX": "SOFR-1B-FLAT"})])


def swap(tenor, rate, pay_or_rec="receive", notional=NOTIONAL):
    return qfCreateProductFromDataConvention(VALUE_DATE, "USD-SOFR-OIS", tenor, rate, notional=notional, pay_or_rec=pay_or_rec)


def target():
    return swap("9Y6M", 0.045)


def engine(model, product):
    eng = ValuationEngineProductRegistry.new_valuation_engine(model, product, params(), ValuationRequest.PV_DETAILED)
    eng.calculate_value()
    return eng


def pv(model, product):
    return qfCreateValueReport(model, product, params(), "PV")[0][1]


def quote_risk(model, product):
    frame = qfCreateValueReport(model, product, params(), "FirstOrderRisk").display()
    ois = frame.loc[frame["DATA_TYPE"] == "OVERNIGHT INDEX SWAP", "VALUES"].to_numpy(dtype=float)
    spread = frame.loc[frame["DATA_TYPE"] == "SPREAD ZERO RATE", "VALUES"].to_numpy(dtype=float)
    return ois, spread


def leg_rows(model, product, leg):
    return [row for row in qfCreateCashflowsReport(model, product, params()).content if row["Leg"] == leg]


def assert_close(a, b, tol):
    assert abs(a - b) <= tol, (a, b, a - b)


# ---- calibration --------------------------------------------------------------------------------
def test_calibration_reprices_every_quote():
    model = base_model()
    swaps = model.retrieve_model_component("SOFR-1B").calibration_product
    assert len(swaps) == 17
    for product, quote in zip(swaps, np.array(QUOTES_BP) / 1e4):
        assert_close(qfCreateValueReport(model, product, params(), "ParRateOrSpread"), quote, 1e-12)
        assert abs(pv(model, product)) < 1e-4                     # a par swap on one million is worth nothing


def test_nodes_sit_at_the_last_payment_dates_in_tenor_order():
    sofr = base_model().retrieve_model_component("SOFR-1B")
    times = sofr.state_data[0]
    assert len(times) == 17 and all(np.diff(times) > 0)
    assert [round(t) for t in times] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50]
    assert sofr.calibration_product[0].last_date.ISO() == "2027-10-14"   # 1Y OIS pays two business days after 2027-10-12
    assert np.array_equal(times, [accrued(sofr.value_date, p.last_date) for p in sofr.calibration_product])


# ---- 2a, 2b: annuity and par rate ---------------------------------------------------------------
def test_annuity_is_the_hand_sum_of_accrual_times_discount_factor():
    model = base_model()
    product = target()
    eng = engine(model, product)
    hand = sum(row["Accrual"] * row["Discount Factor"] for row in leg_rows(model, product, "Fixed"))
    assert_close(eng.annuity_, hand, 1e-12)
    assert 0 < eng.annuity_ < 9.5                                 # nine and a half annual accruals, discounted


def test_par_rate_equals_floating_pv_over_annuity_and_prices_the_swap_to_zero():
    model = base_model()
    product = target()
    eng = engine(model, product)
    floating = sum(row["PV"] for row in leg_rows(model, product, "Floating"))   # already carries the pay sign
    assert_close(eng.par_rate_or_spread_, -floating / (NOTIONAL * eng.annuity_), 1e-12)
    par = qfCreateValueReport(model, product, params(), "ParRateOrSpread")
    assert abs(pv(model, swap("9Y6M", par))) < 1e-6
    assert 0.0481 < par < 0.0485                                  # between the 9Y and 10Y quotes


def test_par_rate_does_not_depend_on_direction_or_notional_and_pv_is_linear():
    model = base_model()
    r1 = qfCreateValueReport(model, swap("9Y6M", 0.045), params(), "ParRateOrSpread")
    r2 = qfCreateValueReport(model, swap("9Y6M", 0.045, "pay"), params(), "ParRateOrSpread")
    r3 = qfCreateValueReport(model, swap("9Y6M", 0.03, notional=250_000), params(), "ParRateOrSpread")
    assert_close(r1, r2, 1e-14)
    assert_close(r1, r3, 1e-14)
    assert_close(pv(model, swap("9Y6M", 0.045, "pay")), -pv(model, target()), 1e-9)
    assert_close(pv(model, swap("9Y6M", 0.045, notional=2 * NOTIONAL)), 2 * pv(model, target()), 1e-9)


def test_reference_present_value():
    assert_close(pv(base_model(), target()), REFERENCE_PV, 0.01)
    # receive 4.5% when par is 4.83%: the receiver is under water by (par - K) * N * annuity
    eng = engine(base_model(), target())
    assert_close(pv(base_model(), target()), (0.045 - eng.par_rate_or_spread_) * NOTIONAL * eng.annuity_, 1e-6)


# ---- 2c: par-rate gradient ----------------------------------------------------------------------
def par_rate_after_state_bump(product, component, position, size):
    model = build()
    model.perturb_model_parameter(component, position, size)
    return qfCreateValueReport(model, product, params(), "ParRateOrSpread")


def test_par_rate_gradient_matches_central_differences_in_the_state():
    model = base_model()
    product = target()
    grad = engine(model, product).grad_at_par()
    assert len(grad) == model.num_components and [len(g) for g in grad] == model.num_sub_components
    sofr_block = grad[model.retrieve_model_component_index("SOFR-1B")]
    h = 1e-6
    for position in (0, 5, 8, 9, 10, 16):
        fd = (par_rate_after_state_bump(product, "SOFR-1B", position, h)
              - par_rate_after_state_bump(product, "SOFR-1B", position, -h)) / (2 * h)
        assert_close(sofr_block[position], fd, 1e-6 * max(1.0, abs(fd)))
    assert sofr_block[9] > 0 and all(abs(v) < 1e-12 for v in sofr_block[10:])   # nothing beyond the swap's last node


def test_par_rate_gradient_is_the_quotient_rule_of_the_leg_gradients():
    model = base_model()
    eng = engine(model, target())
    fixed, floating, local = [], [], []
    for g in (fixed, floating, local):
        model.resize_gradient(g)
    eng.fixed_leg_engine_.calculate_first_order_risk(fixed)
    eng.floating_leg_engine_.calculate_first_order_risk(floating)
    expected = [eng.fixed_rate_ * (gf * eng.fixed_value_ - eng.float_value_ * gx) / eng.fixed_value_ ** 2
                for gf, gx in zip(floating, fixed)]
    for got, want in zip(eng.grad_at_par(), expected):
        assert np.allclose(got, want, rtol=1e-12, atol=1e-18)


# ---- 3: Jacobian --------------------------------------------------------------------------------
def test_jacobian_is_square_in_component_order_and_lower_triangular_for_the_bootstrap():
    model = base_model()
    model.calculate_model_jacobian()
    J = model.model_jacobian
    n = sum(model.num_sub_components)
    assert J.shape == (n, n) == (18, 18)
    start = int(np.cumsum([0] + model.num_sub_components)[model.retrieve_model_component_index("SOFR-1B")])
    ois = J[start:start + 17, start:start + 17]
    assert np.allclose(np.triu(ois, 1), 0.0, atol=1e-12)        # quote i depends only on nodes up to i
    assert all(np.diag(ois) > 0)                                  # a higher own forward means a higher par rate
    assert abs(np.linalg.det(J)) > 0


def test_jacobian_rows_match_finite_differences_of_the_quotes():
    model = base_model()
    model.calculate_model_jacobian()
    J = model.model_jacobian
    swaps = model.retrieve_model_component("SOFR-1B").calibration_product
    start = int(np.cumsum([0] + model.num_sub_components)[model.retrieve_model_component_index("SOFR-1B")])
    h = 1e-6
    for row, product in ((0, swaps[0]), (9, swaps[9]), (16, swaps[16])):
        for col in sorted({0, row}):
            fd = (par_rate_after_state_bump(product, "SOFR-1B", col, h)
                  - par_rate_after_state_bump(product, "SOFR-1B", col, -h)) / (2 * h)
            assert_close(J[start + row, start + col], fd, 1e-6 * max(1.0, abs(fd)))


# ---- 4: quote risk ------------------------------------------------------------------------------
def test_quote_risk_matches_the_reference_and_bump_and_recalibrate():
    model = base_model()
    product = target()
    ois, spread = quote_risk(model, product)
    assert np.allclose(ois, REFERENCE_RISK, atol=2e-6)
    assert len(spread) == 1 and abs(spread[0] - 12.2318) < 1e-3      # the funding-spread row: USD per +1 bp of the zero spread
    assert_close(quote_risk(model, swap("9Y6M", 0.045, "pay"))[1][0], -spread[0], 1e-9)
    base = pv(model, product)
    for tenor in ("1Y", "8Y", "9Y", "10Y", "12Y"):
        up = pv(build({tenor: 0.5e-4}), product)
        down = pv(build({tenor: -0.5e-4}), product)
        assert_close(ois[TENORS.index(tenor)], up - down, 2e-3 + 1e-4 * abs(up - down))   # USD per +1 bp
    assert_close(ois[8] + ois[9], pv(build({"9Y": 0.5e-4, "10Y": 0.5e-4}), product) - pv(build({"9Y": -0.5e-4, "10Y": -0.5e-4}), product), 5e-3)


def test_quote_risk_is_the_state_gradient_times_the_inverse_jacobian():
    model = base_model()
    product = target()
    eng = engine(model, product)
    gradient = []
    eng.calculate_first_order_risk(gradient)
    flat = np.concatenate(gradient)
    model.calculate_model_jacobian()
    by_inverse = flat @ np.linalg.inv(model.model_jacobian)
    ois, spread = quote_risk(model, product)
    start = int(np.cumsum([0] + model.num_sub_components)[model.retrieve_model_component_index("SOFR-1B")])
    assert np.allclose(by_inverse[start:start + 17] * 1e-4, ois, rtol=1e-10, atol=1e-12)
    assert np.allclose(by_inverse * 1e-4, np.concatenate([spread, ois]) if start == 1 else np.concatenate([ois, spread]), rtol=1e-10, atol=1e-12)


def test_quote_risk_signs_and_locality():
    ois, _ = quote_risk(base_model(), target())
    assert ois[8] < 0 and ois[9] < 0                              # receive fixed loses when the 9Y and 10Y quotes rise
    assert all(v > 0 for v in ois[:8])                            # small positive residuals before the swap's last node
    assert all(v == 0 for v in ois[10:])                          # quotes beyond the 10Y node cannot move the swap
    pay, _ = quote_risk(base_model(), swap("9Y6M", 0.045, "pay"))
    assert np.allclose(pay, -ois, atol=1e-9)


def test_gradient_and_quote_risk_hold_for_forward_starting_and_long_swaps():
    model = base_model()
    h = 1e-6
    for tenor, direction, notional in (("2Yx5Y", "receive", NOTIONAL), ("15Y", "pay", NOTIONAL), ("9Y6M", "receive", -NOTIONAL)):
        product = swap(tenor, 0.045, direction, notional)
        block = engine(model, product).grad_at_par()[model.retrieve_model_component_index("SOFR-1B")]
        for position in (1, 6, 9):
            fd = (par_rate_after_state_bump(product, "SOFR-1B", position, h)
                  - par_rate_after_state_bump(product, "SOFR-1B", position, -h)) / (2 * h)
            assert_close(block[position], fd, 1e-6 * max(1.0, abs(fd)))
        ois, _ = quote_risk(model, product)
        for probe in ("2Y", "7Y", "12Y", "15Y"):
            fd = pv(build({probe: 0.5e-4}), product) - pv(build({probe: -0.5e-4}), product)
            assert_close(ois[TENORS.index(probe)], fd, 2e-3 + 1e-4 * abs(fd))


def test_zero_coupon_swap_still_reports_an_annuity_and_par_rate():
    model = base_model()
    eng = engine(model, swap("9Y6M", 0.0))
    reference = engine(model, target())
    assert_close(eng.annuity_, reference.annuity_, 1e-12)
    assert_close(eng.par_rate_or_spread_, reference.par_rate_or_spread_, 1e-12)


def test_par_swap_own_quote_risk_is_minus_its_annuity_per_bp():
    model = base_model()
    for i, tenor in enumerate(("1Y", "10Y", "50Y")):
        product = swap(tenor, QUOTES_BP[TENORS.index(tenor)] / 1e4)
        ois, _ = quote_risk(model, product)
        own = ois[TENORS.index(tenor)]
        assert_close(own, REFERENCE_OWN_RISK[TENORS.index(tenor)], 2e-6)
        assert_close(own, -NOTIONAL * engine(model, product).annuity_ * 1e-4, 1e-6)
        assert all(abs(v) < 1e-9 for j, v in enumerate(ois) if j != TENORS.index(tenor))   # a par OIS is risk only to its own quote


def test_risk_report_rows_carry_the_market_data_labels():
    frame = qfCreateValueReport(base_model(), target(), params(), "FirstOrderRisk").display()
    ois = frame.loc[frame["DATA_TYPE"] == "OVERNIGHT INDEX SWAP"]
    assert list(ois["AXIS1"]) == TENORS
    assert np.allclose(ois["MARKET_QUOTE"].astype(float).to_numpy(), np.array(QUOTES_BP) / 1e4)
    assert (frame["DATA_TYPE"] == "SPREAD ZERO RATE").sum() == 1


# ---- hedge --------------------------------------------------------------------------------------
def hedge_table():
    model = base_model()
    risk, _ = quote_risk(model, target())
    hedges = [swap(tenor, q / 1e4) for tenor, q in zip(TENORS, QUOTES_BP)]
    B = np.array([quote_risk(model, h)[0] for h in hedges])
    weights = np.linalg.solve(B.T, -risk)
    return risk, B, weights


def test_hedge_reproduces_the_reference_and_kills_the_quote_risk():
    risk, B, weights = hedge_table()
    assert np.allclose(np.diag(B), REFERENCE_OWN_RISK, atol=2e-6)
    assert np.allclose(weights * NOTIONAL, REFERENCE_HEDGE, atol=1e-3)
    assert np.allclose(risk + weights @ B, 0.0, atol=1e-9)
    assert weights[8] < 0 and weights[9] < 0 and all(abs(w) < 1e-12 for w in weights[10:])


def test_hedge_notionals_split_the_target_between_the_9y_and_10y_swaps():
    _, _, weights = hedge_table()
    assert_close(weights[8] + weights[9], -0.9978, 2e-4)         # almost one million of pay-fixed in total
    assert 0.45 < weights[8] / (weights[8] + weights[9]) < 0.55   # a 9Y6M swap sits about half way between the two


def test_hedged_book_is_flat_under_a_real_quote_bump():
    _, _, weights = hedge_table()
    book = [(target(), 1.0)] + [(swap(t, q / 1e4), w) for t, q, w in zip(TENORS, QUOTES_BP, weights)]
    def value(model): return sum(w * pv(model, p) for p, w in book)
    base, base_target = value(base_model()), pv(base_model(), target())
    for tenor in ("9Y", "10Y"):
        bumped = build({tenor: 1e-4})
        unhedged = pv(bumped, target()) - base_target
        hedged = value(bumped) - base
        assert abs(unhedged) > 300                                # the naked swap moves by hundreds of dollars per bp
        assert abs(hedged) < 1.0 and abs(hedged) < 0.003 * abs(unhedged)   # the hedged book moves by cents (convexity only)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
