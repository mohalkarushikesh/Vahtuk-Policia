"""
Traffic Violation Dashboard
Reads the JSON report and renders an HTML summary with charts.
Run after detector.py completes.
"""

import json
import os
import sys
from pathlib import Path


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Traffic Violation Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --accent: #f85149;
    --accent2: #58a6ff;
    --accent3: #3fb950;
    --text: #e6edf3;
    --muted: #8b949e;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: 'Segoe UI', system-ui, sans-serif;
    padding: 2rem;
  }}
  header {{
    display: flex; align-items: center; gap: 1rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.5rem; margin-bottom: 2rem;
  }}
  header h1 {{ font-size: 1.6rem; font-weight: 700; }}
  header p  {{ color: var(--muted); font-size: 0.9rem; margin-top: 0.25rem; }}
  .badge {{
    background: var(--accent); color: #fff;
    padding: 0.25rem 0.75rem; border-radius: 99px;
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.05em;
  }}

  .cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem; margin-bottom: 2rem;
  }}
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 1.25rem;
  }}
  .card .num {{
    font-size: 2.2rem; font-weight: 800; line-height: 1;
  }}
  .card .label {{
    color: var(--muted); font-size: 0.75rem;
    text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.4rem;
  }}
  .red   {{ color: var(--accent); }}
  .blue  {{ color: var(--accent2); }}
  .green {{ color: var(--accent3); }}

  .charts {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 1.5rem; margin-bottom: 2rem;
  }}
  .chart-box {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 1.25rem;
  }}
  .chart-box h3 {{
    font-size: 0.85rem; font-weight: 600; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 1rem;
  }}

  table {{
    width: 100%; border-collapse: collapse;
    background: var(--surface); border-radius: 10px; overflow: hidden;
    border: 1px solid var(--border);
  }}
  th {{
    background: #1c2128; padding: 0.75rem 1rem;
    text-align: left; font-size: 0.8rem;
    color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em;
  }}
  td {{ padding: 0.65rem 1rem; font-size: 0.9rem; border-top: 1px solid var(--border); }}
  tr:hover td {{ background: #1c2128; }}
  .sev-HIGH   {{ color: var(--accent);  font-weight: 700; }}
  .sev-MEDIUM {{ color: #d29922; font-weight: 600; }}
  .sev-LOW    {{ color: var(--accent3); }}
  .section-title {{
    font-size: 1rem; font-weight: 700; margin-bottom: 0.75rem;
    display: flex; align-items: center; gap: 0.5rem;
  }}
</style>
</head>
<body>

<header>
  <div>
    <h1>🚦 Traffic Violation Detection Report</h1>
    <p>Powered by YOLOv8n + ByteTrack</p>
  </div>
  <span class="badge">LIVE ANALYSIS</span>
</header>

<div class="cards">
  <div class="card">
    <div class="num red">{total}</div>
    <div class="label">Total Violations</div>
  </div>
  {stat_cards}
</div>

<div class="charts">
  <div class="chart-box">
    <h3>Violations by Type</h3>
    <canvas id="typeChart"></canvas>
  </div>
  <div class="chart-box">
    <h3>Violations by Vehicle Class</h3>
    <canvas id="vehicleChart"></canvas>
  </div>
</div>

<div class="section-title">📋 Violation Log</div>
<table>
  <thead>
    <tr>
      <th>#</th><th>Frame</th><th>Track ID</th>
      <th>Vehicle</th><th>Violation</th><th>Severity</th><th>Confidence</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>

<script>
const typeData = {type_data};
const vehicleData = {vehicle_data};

const palette = [
  '#f85149','#58a6ff','#3fb950','#d29922',
  '#a371f7','#39d3f0','#ff9500','#ff6b6b'
];

new Chart(document.getElementById('typeChart'), {{
  type: 'doughnut',
  data: {{
    labels: Object.keys(typeData),
    datasets: [{{ data: Object.values(typeData), backgroundColor: palette }}]
  }},
  options: {{
    plugins: {{ legend: {{ labels: {{ color: '#e6edf3', font: {{ size: 12 }} }} }} }},
    responsive: true
  }}
}});

new Chart(document.getElementById('vehicleChart'), {{
  type: 'bar',
  data: {{
    labels: Object.keys(vehicleData),
    datasets: [{{
      label: 'Count', data: Object.values(vehicleData),
      backgroundColor: '#58a6ff', borderRadius: 6
    }}]
  }},
  options: {{
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }},
      y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }}
    }},
    responsive: true
  }}
}});
</script>
</body>
</html>"""


def build_stat_cards(by_type):
    icons = {
        "Red Light Running": ("🔴", "red"),
        "Speeding": ("⚡", "red"),
        "Wrong-Way Driving": ("↩️", "red"),
        "Stopped in Intersection": ("🛑", "blue"),
    }
    cards = []
    for vtype, count in by_type.items():
        icon, color = icons.get(vtype, ("⚠️", "blue"))
        short = vtype.split("(")[0].strip()
        cards.append(
            f'<div class="card"><div class="num {color}">{count}</div>'
            f'<div class="label">{icon} {short}</div></div>'
        )
    return "\n".join(cards)


def build_rows(violations):
    rows = []
    for i, v in enumerate(violations, 1):
        sev_cls = f"sev-{v['severity']}"
        rows.append(
            f"<tr>"
            f"<td>{i}</td>"
            f"<td>{v['frame_no']}</td>"
            f"<td>#{v['track_id']}</td>"
            f"<td>{v['vehicle_class']}</td>"
            f"<td>{v['violation_type']}</td>"
            f"<td class='{sev_cls}'>{v['severity']}</td>"
            f"<td>{v['confidence']:.2f}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def generate(report_path="output/violation_report.json",
             out_path="output/dashboard.html"):
    with open(report_path) as f:
        data = json.load(f)

    html = TEMPLATE.format(
        total=data["total_violations"],
        stat_cards=build_stat_cards(data["by_type"]),
        type_data=json.dumps(data["by_type"]),
        vehicle_data=json.dumps(data["by_vehicle"]),
        rows=build_rows(data["violations"])
    )

    with open(out_path, "w") as f:
        f.write(html)
    print(f"[Dashboard] → {out_path}")


if __name__ == "__main__":
    rp = sys.argv[1] if len(sys.argv) > 1 else "output/violation_report.json"
    generate(rp)