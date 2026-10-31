"""HTML and JSON renderings of a finished check."""

from __future__ import annotations

import html
import json


def report_json(result: dict) -> str:
    return json.dumps(result, indent=2) + "\n"


def render_html(result: dict) -> str:
    esc = html.escape
    balance = result["balance"]
    dupes = result["duplicates"]
    coverage = result["coverage"]
    gate = result["gate"]
    gap_items = "".join(f"<li>{esc(gap['suggestion'])}</li>" for gap in coverage["gaps"])
    if not gap_items:
        gap_items = "<li>No pose or framing gaps at the current target.</li>"
    group_rows = []
    for group in dupes["multi_groups"]:
        names = ", ".join(esc(image["rel"]) + (" (keep)" if image["role"] == "keep" else "") for image in group["images"])
        group_rows.append(
            f"<tr><td>{group['id']}</td><td>{group['size']}</td><td>{esc(group['suggested_keep'])}</td><td>{names}</td></tr>"
        )
    if not group_rows:
        group_rows.append("<tr><td colspan='4'>No near-duplicate groups.</td></tr>")
    bucket_rows = []
    for model, rows in coverage["buckets"].items():
        for row in rows:
            if row["count"]:
                bucket_rows.append(
                    f"<tr><td>{esc(model)}</td><td>{esc(row['name'])}</td><td>{row['count']}</td></tr>"
                )
    gate_line = esc(gate["block_reason"] or "Every crop is approved. Export may write those crops.")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LoraDatasetGate report</title>
<style>
body {{ font-family: "Segoe UI", sans-serif; background: #0c1422; color: #e7f0f8; margin: 0; }}
main {{ max-width: 960px; margin: 0 auto; padding: 32px 20px 64px; }}
h1 {{ font-weight: 650; letter-spacing: -0.03em; }}
table {{ width: 100%; border-collapse: collapse; background: #121b2c; }}
th, td {{ border-bottom: 1px solid #24344a; text-align: left; padding: 8px 10px; vertical-align: top; }}
.note {{ color: #9fb0c4; }}
.callout {{ background: #13281f; border: 1px solid #1f6b4a; padding: 12px 14px; border-radius: 10px; }}
</style>
</head>
<body>
<main>
<h1>LoraDatasetGate report</h1>
<p class="note">{esc(result["generated_at"])} · {esc(result["dataset"])}</p>
<p>Subject images: {result["counts"]["subject"]} · Regularization images: {result["counts"]["reg"]}</p>
<h2>Balance</h2>
<div class="callout">{esc(balance["message"])}</div>
<h2>Near duplicates</h2>
<p>{dupes["multi_group_count"]} group(s) with more than one image. Suggested keep is highest resolution, then sharpness. Nothing was deleted.</p>
<table>
<thead><tr><th>Group</th><th>Size</th><th>Keep</th><th>Files</th></tr></thead>
<tbody>{"".join(group_rows)}</tbody>
</table>
<h2>Coverage gaps</h2>
<ul>{gap_items}</ul>
<h2>Aspect buckets</h2>
<table>
<thead><tr><th>Model</th><th>Bucket</th><th>Count</th></tr></thead>
<tbody>{"".join(bucket_rows)}</tbody>
</table>
<h2>Face gate</h2>
<p>{gate_line}</p>
<p>Approved {gate["approved_count"]} · Rejected {gate["rejected_count"]} · Pending {gate["pending_count"]}</p>
<p class="note">CPU only. Loopback only. No files were deleted. The approval log is append-only.</p>
</main>
</body>
</html>
"""
