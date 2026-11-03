"""Refresh the README results table using only results/metrics.csv."""
import csv
from pathlib import Path

root=Path(__file__).resolve().parents[1]; metrics=root/"results"/"metrics.csv"; readme=root/"README.md"
start="<!-- RESULTS TABLE START -->"; end="<!-- RESULTS TABLE END -->"
if metrics.is_file():
    with metrics.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
else: rows=[]
lines=["| Model | Aspect | F-max | Micro F-max | Mean term AP | Rare-term F-max | Frequent-term F-max |","|---|---|---:|---:|---:|---:|---:|"]
if rows:
    for r in rows:
        lines.append("| {model} | {aspect} | {fmax} | {micro_fmax} | {mean_term_ap} | {rare_fmax} | {frequent_fmax} |".format(**{k:r.get(k,"TBD") for k in ("model","aspect","fmax","micro_fmax","mean_term_ap","rare_fmax","frequent_fmax")}))
else: lines.append("| TBD | TBD | TBD | TBD | TBD | TBD | TBD |")
table="\n".join(lines)
text=readme.read_text(encoding="utf-8") if readme.exists() else "# ProtFunc\n\n"
if start in text and end in text: text=text[:text.index(start)+len(start)]+"\n"+table+"\n"+text[text.index(end):]
else: text += "\n"+start+"\n"+table+"\n"+end+"\n"
readme.write_text(text,encoding="utf-8")
print(f"Updated {readme} from {metrics if metrics.is_file() else 'no metrics.csv (TBD)'}")
