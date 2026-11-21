# FRE-GY-9743-Assignments-2026fall

## Assignment 8 submission: Raj Pawar (`Raj_Pawar_Branch`)

| Deliverable | Location |
|---|---|
| TODO 2a, 2b: `ValuationEngineProductRfrSwap.calculate_value` sets the fixed-leg annuity per unit notional (fixed-leg PV over notional times coupon) and the OIS par rate (floating-leg PV over notional times annuity) | `fixedincomelib/yield_curve/valuation_engine.py` |
| TODO 2c: `grad_at_par` accumulates the par-rate gradient with respect to the internal state by the quotient rule, calling each leg engine's gradient once with the matching scalar | `fixedincomelib/yield_curve/valuation_engine.py` |
| TODO 3: `YieldCurve.calculate_model_jacobian` appends each calibration instrument's par-rate gradient, flattened in model component order, as one row of `J` | `fixedincomelib/yield_curve/yield_curve_model.py` |
| TODO 4: `risk_calculation` converts the state gradient to quote risk by solving `J^T r^T = (grad V)^T`, i.e. `grad_M V = grad_I V J^{-1}` | `fixedincomelib/valuation/utilities.py` |
| Professor's notebook executed top to bottom with no cell changed; the repricing table, swap PV, quote risks and hedge notionals equal the reference outputs he shipped in it to the printed precision (the 1e-10 bp repricing errors differ in their last digits) | `hw8_sofr_ois_calibration_and_risk.ipynb` |
| Tests: repricing, node placement, annuity and par rate by hand, par-rate gradient and Jacobian against finite differences (also for forward-starting, 15Y and negative-notional swaps), quote risk against the reference and against bump-and-recalibrate, the funding-spread row, signs and locality, par OIS PV01, a zero-coupon swap, report labels, the hedge | `tests/test_ois_calibration_and_risk.py` |
| Write-up: the bootstrap, the five pieces with their formulas, and a reading of the risk and hedge tables | `writeup/Assignment8_Writeup_RajPawar.pdf` (source `.tex` alongside) |
| Project page, numbers and figure generated from the library | `docs/index.html` (built by `docs/build_page.py`) |

### How to run

```bash
pip install -r requirements.txt                        # Python 3.11, QuantLib 1.41, scipy, openpyxl; matplotlib is only for docs/build_page.py
jupyter notebook hw8_sofr_ois_calibration_and_risk.ipynb   # Run All
python tests/test_ois_calibration_and_risk.py          # or, with pytest installed: python -m pytest tests -q
```

Within `fixedincomelib/` only the three `TODO` files changed, and in them only the five `TODO` bodies (the template's line that stacks a component's rows into the Jacobian is kept as supplied).
