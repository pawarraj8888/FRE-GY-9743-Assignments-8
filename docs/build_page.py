"""Generate docs/index.html: the Assignment 8 project page (calibration, risk by OIS quote, hedge).

Run from the repository root with the QuantLib interpreter:  python docs/build_page.py
"""
import html, io, os, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests")); os.chdir(ROOT)
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from test_ois_calibration_and_risk import NOTIONAL, QUOTES_BP, TENORS, base_model, engine, hedge_table, params, pv, target  # noqa: E402
from fixedincomelib import qfCreateValueReport  # noqa: E402

model = base_model(); eng = engine(model, target())
sofr = model.retrieve_model_component("SOFR-1B")
swaps = sofr.calibration_product
model_bp = [1e4 * qfCreateValueReport(model, p, params(), "ParRateOrSpread") for p in swaps]
risk, B, w = hedge_table()
n_tests = int([ln for ln in subprocess.run([sys.executable, "tests/test_ois_calibration_and_risk.py"], capture_output=True, text=True, check=True)
               .stdout.splitlines() if ln.endswith("tests passed")][0].split()[0])

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
x = np.arange(len(TENORS))
axes[0].bar(x, risk, color=["C3" if v < 0 else "C0" for v in risk]); axes[0].set_xticks(x); axes[0].set_xticklabels(TENORS, rotation=60, fontsize=8)
axes[0].set_title("Receive-fixed 9Y6M OIS at 4.5%: risk to each OIS quote (USD per +1 bp)", fontsize=9); axes[0].grid(alpha=0.3, axis="y")
axes[1].bar(x, w * NOTIONAL / 1e3, color=["C3" if v < 0 else "C0" for v in w]); axes[1].set_xticks(x); axes[1].set_xticklabels(TENORS, rotation=60, fontsize=8)
axes[1].set_title("Hedge: receive-fixed par OIS notional (USD thousands; negative = pay fixed)", fontsize=9); axes[1].grid(alpha=0.3, axis="y")
for ax in axes:
    ax.tick_params(labelsize=8)
fig.tight_layout()
buf = io.StringIO(); fig.savefig(buf, format="svg", metadata={"Date": None}); plt.close(fig)
svg = buf.getvalue()[buf.getvalue().index("<svg"):]
svg = re.sub(r'\s(width|height)="[^"]*pt"', "", svg, count=2).replace("<svg ", '<svg style="width:100%;height:auto" role="img" aria-label="Quote risk and hedge notionals by OIS tenor" ', 1)

def table(headers, rows):
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
calib_rows = [(t, p.last_date.ISO(), f"{tt:.6f}", f"{100 * fwd:.4f}%", f"{q:.3f}", f"{m:.3f}", f"{m - q:.1e}")
              for t, p, tt, fwd, q, m in zip(TENORS, swaps, sofr.state_data[0], sofr.state_data[1], QUOTES_BP, model_bp)]
risk_rows = [(t, f"{q / 1e4:.6f}", f"{r:,.6f}", f"{b:,.6f}", f"{wi * NOTIONAL:,.2f}", f"{res:.1e}")
             for t, q, r, b, wi, res in zip(TENORS, QUOTES_BP, risk, np.diag(B), w, risk + w @ B)]

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Assignment 8: SOFR OIS calibration and hedging risk (Raj Pawar)</title>
<style>
body{{font-family:Georgia,'Times New Roman',serif;max-width:900px;margin:32px auto;padding:0 16px;color:#222;line-height:1.45;background:#fff}}
h1{{font-size:1.6em;margin-bottom:0}} h2{{font-size:1.2em;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:3px}}
.meta{{color:#555;margin-top:4px}} code{{font-family:Menlo,Consolas,monospace;font-size:0.92em;background:#f4f4f4;padding:1px 4px;border-radius:3px}}
table{{border-collapse:collapse;font-size:0.85em;margin:10px 0;width:100%;overflow-x:auto;display:block}} th,td{{border:1px solid #ddd;padding:4px 6px;text-align:right;white-space:nowrap}}
th{{background:#f7f7f7}} td:first-child,th:first-child{{text-align:left}} .formula{{background:#f8f8ff;border-left:3px solid #1f4e79;padding:8px 12px;margin:8px 0;font-size:0.95em}}
.note{{color:#444;font-size:0.95em}} sup,sub{{line-height:0}} figure{{margin:12px 0}} figcaption{{color:#555;font-size:0.9em}}
</style></head><body>
<h1>Assignment 8: SOFR OIS Calibration and Hedging Risk</h1>
<p class="meta">Raj Pawar, FRE-GY 9743 (Spec Topics in Risk Management), NYU Tandon, November 2026.
Branch <a href="https://github.com/pawarraj8888/FRE-GY-9743-Assignments-8/tree/Raj_Pawar_Branch"><code>Raj_Pawar_Branch</code></a>,
write-up <a href="https://github.com/pawarraj8888/FRE-GY-9743-Assignments-8/blob/Raj_Pawar_Branch/writeup/Assignment8_Writeup_RajPawar.pdf">PDF</a>.
The five TODOs are in <code>fixedincomelib/yield_curve/valuation_engine.py</code> (annuity, par rate, par-rate gradient),
<code>fixedincomelib/yield_curve/yield_curve_model.py</code> (Jacobian rows) and <code>fixedincomelib/valuation/utilities.py</code> (quote risk).</p>

<h2>Calibration</h2>
<p class="note">Valuation date 2026-10-09; 17 SOFR OIS par quotes from <code>SOFR OIS.xlsx</code>. The SOFR forward curve is piecewise constant with one node per quote at the
swap's last payment date, bootstrapped one node at a time with Brent's method; the funding spread over SOFR is a single node fixed at zero.</p>
{table(["OIS", "Node (last payment)", "Time (years)", "Node forward", "Market (bp)", "Model (bp)", "Error (bp)"], calib_rows)}

<h2>The pieces</h2>
<div class="formula">annuity per unit notional A = V<sub>fixed</sub> / (N K); &nbsp; par rate R = V<sub>float</sub> / (N A); &nbsp;
&nabla;<sub>X<sup>I</sup></sub>R = &nabla;<sub>X<sup>I</sup></sub>V<sub>float</sub> / (N A) &minus; R &nabla;<sub>X<sup>I</sup></sub>V<sub>fixed</sub> / V<sub>fixed</sub>; &nbsp;
J = &part;X<sup>M</sup>/&part;X<sup>I</sup> (one row per calibration instrument); &nbsp; &nabla;<sub>X<sup>M</sup></sub>V = &nabla;<sub>X<sup>I</sup></sub>V J<sup>&minus;1</sup>; &nbsp;
hedge: B<sup>T</sup> w = &minus;r<sup>T</sup>.</div>
<p>The annuity is the fixed-leg PV per unit coupon and notional; the par rate is the coupon that prices the swap to zero. The par-rate gradient with respect to the internal
state (the node forwards, flattened in model component order) follows from the quotient rule with the coupon held fixed, and each calibration instrument's gradient is one
row of the Jacobian, which is 18 &times; 18 here (17 OIS nodes plus the funding node) and lower triangular on the OIS block because a quote only depends on nodes up to its own
maturity. Quote risk is the chain rule across the change of coordinates from state to quotes, computed by solving J<sup>T</sup> r<sup>T</sup> = (&nabla;<sub>X<sup>I</sup></sub>V)<sup>T</sup>;
the report scales by 10<sup>&minus;4</sup> to USD per basis point.</p>

<h2>Risk of a USD 1,000,000 receive-fixed 9Y6M OIS at 4.5% and its hedge</h2>
<p class="note">PV {pv(model, target()):,.2f} USD (annuity {eng.annuity_:.4f}, par rate {100 * eng.par_rate_or_spread_:.4f}%). Risk is USD per +1 bp of each quote, all other quotes fixed and the curve
recalibrated; the hedge is one receive-fixed par OIS per quote with B<sup>T</sup> w = &minus;r<sup>T</sup> (negative notional = pay fixed).</p>
<figure>{svg}<figcaption>Quote risk of the target (left) and the hedge notionals (right).</figcaption></figure>
{table(["OIS", "Quote (decimal)", "Target risk (USD/bp)", "1m par OIS risk to own quote (USD/bp)", "Hedge notional (USD, receive > 0)", "Residual (USD/bp)"], risk_rows)}
<p>A receive-fixed swap loses when rates rise, so the risk is negative and concentrated in the 9Y and 10Y quotes, whose forward segments carry the swap's last cashflows. Raising an
earlier quote lifts the forward on its own segment and lowers the next one (the following swap must still reprice), and a swap that pays both segments sees the two moves cancel to
first order, which leaves the small positive residuals at 1Y to 8Y; quotes beyond 10Y cannot move the swap at all. The hedge is about one million of pay-fixed split roughly half
and half between 9Y and 10Y, a few thousand of receive-fixed at the shorter tenors, and nothing beyond 10Y. A par OIS has risk only to its own quote, equal to minus its PV01.</p>

<h2>Checks</h2>
<p class="note">{n_tests} tests in <code>tests/test_ois_calibration_and_risk.py</code>: repricing and zero PV of every calibration swap, node placement, the annuity against the cashflow
report, the par rate and its invariances, the reference PV, the par-rate gradient and Jacobian rows against finite differences, the quote risk against the shipped reference, the
explicit inverse and bump-and-recalibrate, signs and locality, a par OIS's own-quote risk as its PV01, the report labels, and the hedge (reference notionals, zero residual, the
9Y/10Y split, a real one basis point move of the hedged book). Source for this page: <code>docs/build_page.py</code>.</p>
</body></html>
"""
(ROOT / "docs" / "index.html").write_text(page)
print("wrote docs/index.html", len(page), "bytes")
