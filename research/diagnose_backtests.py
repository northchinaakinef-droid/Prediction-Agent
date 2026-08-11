"""Read-only diagnostics for existing locked walk-forward records."""
from __future__ import annotations
import argparse, json, math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

EDGE = ((-math.inf,0,"<0%"),(0,.02,"0%-2%"),(.02,.05,"2%-5%"),(.05,.1,"5%-10%"),(.1,math.inf,">=10%"))
PRICE = ((0,.5,"<0.50"),(.5,.6,"0.50-0.60"),(.6,.7,"0.60-0.70"),(.7,.8,"0.70-0.80"),
         (.8,.9,"0.80-0.90"),(.9,.95,"0.90-0.95"),(.95,.97,"0.95-0.97"),(.97,math.inf,"0.97+"))

def logloss(y, p):
    p = min(1-1e-12, max(1e-12, p))
    return -(y*math.log(p)+(1-y)*math.log(1-p))

def dd(pnls, initial=1000):
    bank = peak = initial
    worst = 0.
    for pnl in pnls:
        bank += pnl; peak = max(peak, bank); worst = max(worst, (peak-bank)/peak)
    return worst

def pf(pnls):
    gain, loss = sum(x for x in pnls if x > 0), -sum(x for x in pnls if x < 0)
    return gain/loss if loss else None

def streak(pnls):
    best = current = 0
    for pnl in pnls:
        current = current+1 if pnl < 0 else 0; best = max(best, current)
    return best

def strata(rows, bins, key):
    out = []
    for low, high, label in bins:
        xs = [r for r in rows if low <= float(r[key]) < high]
        pnls, stakes = [float(r["pnl"]) for r in xs], [float(r["stake"]) for r in xs]
        out.append({"range":label,"samples":len(xs),
          "win_rate":mean(float(r["won"]) for r in xs) if xs else None,
          "average_entry_price":mean(float(r["execution_price"]) for r in xs) if xs else None,
          "roi":sum(pnls)/sum(stakes) if stakes and sum(stakes) else None,
          "max_drawdown":dd(pnls),"profit_factor":pf(pnls)})
    return out

def calibration(rows, key):
    out = []
    for i in range(10):
        xs = [r for r in rows if i/10 <= float(r[key]) < (i+1)/10 or i == 9 and float(r[key]) == 1]
        if xs:
            out.append({"range":f"{i/10:.1f}-{(i+1)/10:.1f}","samples":len(xs),
                        "predicted":mean(float(r[key]) for r in xs),
                        "actual":mean(float(r["winner"] == 0) for r in xs)})
    return out

def diagnose(report):
    rows, trades = report["records"], []
    for row in rows:
        if float(row.get("stake") or 0) <= 0: continue
        r, side = dict(row), int(row["side"])
        r["market_selected"] = float(row["market_p_a"]) if side == 0 else 1-float(row["market_p_a"])
        r["model_selected"] = float(row["estimate_p_a"]) if side == 0 else 1-float(row["estimate_p_a"])
        r["edge"] = r["model_selected"]-float(row["execution_price"])
        trades.append(r)
    y = [int(r["winner"] == 0) for r in rows]
    model, market = [float(r["estimate_p_a"]) for r in rows], [float(r["market_p_a"]) for r in rows]
    pnls, stakes = [float(r["pnl"]) for r in trades], [float(r["stake"]) for r in trades]
    raw = [float(r["stake"])/float(r["market_selected"])*float(r["won"])-float(r["stake"]) for r in trades]
    mb, bb = mean((p-a)**2 for p,a in zip(model,y)), mean((p-a)**2 for p,a in zip(market,y))
    ml, bl = mean(logloss(a,p) for p,a in zip(model,y)), mean(logloss(a,p) for p,a in zip(market,y))
    roi, raw_roi = sum(pnls)/sum(stakes), sum(raw)/sum(stakes)
    return {"league":report["league"],"samples":len(rows),"trades":len(trades),
      "wins":sum(bool(r["won"]) for r in trades),"win_rate":mean(float(r["won"]) for r in trades),
      "model_direction_accuracy":mean(float((p>=.5)==bool(a)) for p,a in zip(model,y)),
      "market_direction_accuracy":mean(float((p>=.5)==bool(a)) for p,a in zip(market,y)),
      "average_entry_price":mean(float(r["execution_price"]) for r in trades),
      "average_model_probability":mean(float(r["model_selected"]) for r in trades),
      "average_edge":mean(float(r["edge"]) for r in trades),"turnover":sum(stakes),"profit":sum(pnls),
      "roi":roi,"frictionless_roi":raw_roi,"cost_drag":raw_roi-roi,"max_drawdown":dd(pnls),
      "profit_factor":pf(pnls),"maximum_losing_streak":streak(pnls),
      "model_brier":mb,"market_brier":bb,"brier_skill_vs_market":1-mb/bb,
      "model_log_loss":ml,"market_log_loss":bl,"model_beats_market_probability":mb<bb and ml<bl,
      "calibration":calibration(rows,"estimate_p_a"),"edge_strata":strata(trades,EDGE,"edge"),
      "entry_price_strata":strata(trades,PRICE,"execution_price")}

def fmt(v, pct=False):
    return "-" if v is None else f"{v:.1%}" if pct else f"{v:.3f}"

def render_report(results, smart):
    lines = ["# Negative ROI diagnostic (no tuning)", "",
      f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
      "This report only reads existing locked/walk-forward outputs; it does not refit or select thresholds.", "",
      "|League|Samples|Trades|Win rate|Model/market Brier|Model/market Log Loss|Net/frictionless ROI|Cost drag|Max DD|PF|Max loss streak|",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"|{r['league'].upper()}|{r['samples']}|{r['trades']}|{fmt(r['win_rate'],1)}|{r['model_brier']:.3f}/{r['market_brier']:.3f}|{r['model_log_loss']:.3f}/{r['market_log_loss']:.3f}|{fmt(r['roi'],1)}/{fmt(r['frictionless_roi'],1)}|{fmt(r['cost_drag'],1)}|{fmt(r['max_drawdown'],1)}|{fmt(r['profit_factor'])}|{r['maximum_losing_streak']}|")
    lines += ["|CS2|0|0|-|-|-|-|-|-|-|-|", "",
              "CS2 has no historical records. Keep NO TRADE until an independent lockbox exists.", ""]
    for r in results:
        verdict = "beats" if r["model_beats_market_probability"] else "does not beat"
        lines += [f"## {r['league'].upper()}", "", f"The model {verdict} the market on both Brier and Log Loss.", "",
                  "|Edge|Samples|Win rate|Average entry|ROI|Max DD|PF|",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for b in r["edge_strata"]:
            lines.append(f"|{b['range']}|{b['samples']}|{fmt(b['win_rate'],1)}|{fmt(b['average_entry_price'])}|{fmt(b['roi'],1)}|{fmt(b['max_drawdown'],1)}|{fmt(b['profit_factor'])}|")
        lines += ["", "|Entry price|Samples|Win rate|ROI|Max DD|PF|",
                  "|---|---:|---:|---:|---:|---:|"]
        for b in r["entry_price_strata"]:
            lines.append(f"|{b['range']}|{b['samples']}|{fmt(b['win_rate'],1)}|{fmt(b['roi'],1)}|{fmt(b['max_drawdown'],1)}|{fmt(b['profit_factor'])}|")
        lines.append("")
    if smart:
        lines += ["## Second internal test (smart-money model)", "",
                  "The old lockbox saved summaries and curves, but no per-signal probability/entry ask; retrospective strata would be unreliable.", "",
                  "|League|Samples|Trades|Brier|Log Loss|ROI|Max DD|",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for league,v in smart.get("results",{}).items():
            p,t=v["probability"],v["trading"]
            lines.append(f"|{league.upper()}|{v['samples']}|{t['bets']}|{p['brier']:.3f}|{p['log_loss']:.3f}|{fmt(t['roi'],1)}|{fmt(t['max_drawdown'],1)}|")
    lines += ["", "## Decision", "",
              "No tradable edge has been demonstrated. Keep all markets at NO TRADE. Select models/rules only on a new validation window and run OOS once. Require probability superiority to market, improving edge strata, positive net returns, and consistency across walk-forward windows before paper trading.", ""]
    return "\n".join(lines)


def run():
    p=argparse.ArgumentParser()
    p.add_argument("--source",default="reports/polymarket_walkforward.json")
    p.add_argument("--smart-lock",default="reports/smart_money/FINAL_TEST_LOCK.json")
    p.add_argument("--json",default="reports/diagnostic_backtest.json")
    p.add_argument("--markdown",default="reports/negative_roi_diagnosis.md")
    a=p.parse_args()
    results=[diagnose(r) for r in json.loads(Path(a.source).read_text(encoding="utf-8"))["reports"]]
    smart_path=Path(a.smart_lock)
    smart=json.loads(smart_path.read_text(encoding="utf-8")) if smart_path.exists() else None
    Path(a.json).write_text(json.dumps({"generated_at":datetime.now(timezone.utc).isoformat(),
      "tuning_performed":False,"leagues":results},ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    Path(a.markdown).write_text(render_report(results,smart),encoding="utf-8")

if __name__=="__main__":
    run()
