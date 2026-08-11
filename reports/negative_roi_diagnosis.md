# Negative ROI diagnostic (no tuning)

Generated: 2026-08-11T11:41:44.742253+00:00

This report only reads existing locked/walk-forward outputs; it does not refit or select thresholds.

|League|Samples|Trades|Win rate|Model/market Brier|Model/market Log Loss|Net/frictionless ROI|Cost drag|Max DD|PF|Max loss streak|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|NBA|533|191|25.7%|0.178/0.173|0.535/0.517|-14.5%/-8.3%|6.2%|29.6%|0.810|22|
|CBA|130|95|23.2%|0.175/0.168|0.531/0.502|-25.6%/-20.3%|5.3%|20.2%|0.676|11|
|LOL|791|533|19.3%|0.186/0.173|0.556/0.518|-26.5%/-20.4%|6.2%|64.7%|0.680|19|
|CS2|0|0|-|-|-|-|-|-|-|-|

CS2 has no historical records. Keep NO TRADE until an independent lockbox exists.

## NBA

The model does not beat the market on both Brier and Log Loss.

|Edge|Samples|Win rate|Average entry|ROI|Max DD|PF|
|---|---:|---:|---:|---:|---:|---:|
|<0%|0|-|-|-|0.0%|-|
|0%-2%|0|-|-|-|0.0%|-|
|2%-5%|36|38.9%|0.271|36.0%|5.0%|1.580|
|5%-10%|127|22.0%|0.256|-29.1%|32.5%|0.638|
|>=10%|28|25.0%|0.228|-14.8%|5.3%|0.808|

|Entry price|Samples|Win rate|ROI|Max DD|PF|
|---|---:|---:|---:|---:|---:|
|<0.50|182|24.2%|-15.1%|28.4%|0.806|
|0.50-0.60|7|42.9%|-20.5%|2.1%|0.641|
|0.60-0.70|2|100.0%|62.7%|0.0%|-|
|0.70-0.80|0|-|-|0.0%|-|
|0.80-0.90|0|-|-|0.0%|-|
|0.90-0.95|0|-|-|0.0%|-|
|0.95-0.97|0|-|-|0.0%|-|
|0.97+|0|-|-|0.0%|-|

## CBA

The model does not beat the market on both Brier and Log Loss.

|Edge|Samples|Win rate|Average entry|ROI|Max DD|PF|
|---|---:|---:|---:|---:|---:|---:|
|<0%|0|-|-|-|0.0%|-|
|0%-2%|0|-|-|-|0.0%|-|
|2%-5%|10|50.0%|0.322|22.8%|1.9%|1.443|
|5%-10%|56|21.4%|0.267|-30.5%|14.0%|0.622|
|>=10%|29|17.2%|0.166|-32.6%|9.8%|0.619|

|Entry price|Samples|Win rate|ROI|Max DD|PF|
|---|---:|---:|---:|---:|---:|
|<0.50|93|21.5%|-27.9%|20.8%|0.654|
|0.50-0.60|2|100.0%|83.0%|0.0%|-|
|0.60-0.70|0|-|-|0.0%|-|
|0.70-0.80|0|-|-|0.0%|-|
|0.80-0.90|0|-|-|0.0%|-|
|0.90-0.95|0|-|-|0.0%|-|
|0.95-0.97|0|-|-|0.0%|-|
|0.97+|0|-|-|0.0%|-|

## LOL

The model does not beat the market on both Brier and Log Loss.

|Edge|Samples|Win rate|Average entry|ROI|Max DD|PF|
|---|---:|---:|---:|---:|---:|---:|
|<0%|0|-|-|-|0.0%|-|
|0%-2%|0|-|-|-|0.0%|-|
|2%-5%|72|26.4%|0.331|-15.6%|11.3%|0.783|
|5%-10%|251|23.5%|0.256|-15.8%|23.0%|0.799|
|>=10%|210|11.9%|0.155|-43.4%|41.0%|0.524|

|Entry price|Samples|Win rate|ROI|Max DD|PF|
|---|---:|---:|---:|---:|---:|
|<0.50|530|19.2%|-26.5%|64.8%|0.680|
|0.50-0.60|2|50.0%|7.7%|0.3%|1.172|
|0.60-0.70|1|0.0%|-101.0%|0.3%|0.000|
|0.70-0.80|0|-|-|0.0%|-|
|0.80-0.90|0|-|-|0.0%|-|
|0.90-0.95|0|-|-|0.0%|-|
|0.95-0.97|0|-|-|0.0%|-|
|0.97+|0|-|-|0.0%|-|

## Second internal test (smart-money model)

The old lockbox saved summaries and curves, but no per-signal probability/entry ask; retrospective strata would be unreliable.

|League|Samples|Trades|Brier|Log Loss|ROI|Max DD|
|---|---:|---:|---:|---:|---:|---:|
|NBA|267|116|0.177|0.527|-10.5%|28.0%|
|CBA|66|57|0.233|0.652|-15.4%|11.3%|
|LOL|124|59|0.201|0.590|-8.1%|12.1%|

## Decision

No tradable edge has been demonstrated. Keep all markets at NO TRADE. Select models/rules only on a new validation window and run OOS once. Require probability superiority to market, improving edge strata, positive net returns, and consistency across walk-forward windows before paper trading.
