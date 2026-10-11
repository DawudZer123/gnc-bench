"""Automatic campaign report: one HTML file you can open in any browser or send to someone.

Everything is inside the file (figures included), so it can be emailed as is.
"""

import base64
import html
import io
from datetime import datetime

from .campaign import summarize

STYLE = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 980px;
       margin: 32px auto; padding: 0 16px; color: #222; }
h1 { margin-bottom: 4px; } .sub { color: #666; margin-top: 0; }
table { border-collapse: collapse; margin: 12px 0 24px; width: 100%; }
th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { background: #f4f4f4; }
.PASS { color: #1a7f37; font-weight: bold; } .FAIL { color: #c62828; font-weight: bold; }
img { max-width: 100%; }
"""


def _figure_to_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{data}">'


def _table(header, rows):
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _verdict(pass_rate):
    return '<span class="PASS">PASS</span>' if pass_rate == 1.0 else '<span class="FAIL">FAIL</span>'


def write_report(path, results, corridor, figures=(), title="GNC Bench campaign report"):
    """Writes the report.

    results: {version name: campaign rows}, as returned by run_campaign.
    figures: matplotlib figures to embed at the end.
    A version passes the campaign only if EVERY flight stays in the corridor.
    """
    summaries = {name: summarize(rows) for name, rows in results.items()}
    n = next(iter(summaries.values()))["flights"]

    summary_table = _table(
        ["Version", "Campaign", "Pass rate", "Median max error", "95th pct", "Worst flight",
         "Control effort (RMS)", "Flights with saturation"],
        [[html.escape(name), _verdict(s["pass_rate"]), f"{s['pass_rate'] * 100:.0f} %",
          f"{s['median_max_error_m']:.2f} m", f"{s['p95_max_error_m']:.2f} m",
          f"{s['worst_max_error_m']:.2f} m (seed {s['worst_seed']})",
          f"{s['mean_control_effort_N']:.3f} N", s["saturated_flights"]]
         for name, s in summaries.items()],
    )

    worst_sections = []
    for name, rows in results.items():
        worst = sorted(rows, key=lambda r: r["max_error_m"], reverse=True)[:5]
        worst_sections.append(f"<h3>{html.escape(name)}</h3>" + _table(
            ["Seed", "Verdict", "Max error", "Time out", "Wind", "Gusts", "Outage", "Mass error"],
            [[r["seed"], f'<span class="{r["verdict"]}">{r["verdict"]}</span>',
              f"{r['max_error_m']:.2f} m", f"{r['time_out_of_corridor_s']:.1f} s",
              f"{r['wind_speed']:.1f} m/s", f"{r['gust_std']:.1f} m/s",
              f"{r['outage_length']:.1f} s", f"{r['mass_error'] * 100:+.0f} %"]
             for r in worst],
        ))

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{STYLE}</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="sub">Generated {datetime.now():%Y-%m-%d %H:%M} · {n} random flights per version ·
corridor ±{corridor} m during the GPS outage</p>

<h2>Summary</h2>
{summary_table}

<h2>Test conditions</h2>
<p>Each flight follows a 6 m × 3 m figure-8 and draws its own conditions: mean wind 0 to 5 m/s
in any direction, gusts 0.5 to 2 m/s, a GPS outage of 15 to 50 % of the flight, and a mass error
of ±10 % that the controller does not know about. Every version flies exactly the same flights.
Any flight can be replayed with <code>Scenario.random(seed).fly()</code>.</p>

<h2>Worst flights</h2>
{"".join(worst_sections)}

<h2>Figures</h2>
{"".join(_figure_to_html(f) for f in figures)}
</body></html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path
