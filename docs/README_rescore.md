# Re-score por centroide

Fluxo para recalcular rankings penalizando deslocamentos de centroide (off-target).

## Preparacao
- Ambiente ativado (`conda activate projeto-toi`).
- Checkpoint/preds gerados pelo treino (`artifacts/runs/<exp>/preds.csv`).
- Dataset npz acessivel (padrao: `data/datasets/windows`).

## Executar
```bash
python -m projeto_toi.cli.rescore_predictions \
  --preds artifacts/runs/exp_01/preds.csv \
  --data-root data/datasets/windows \
  --tau-list "0.7,1.5,2,3,5" \
  --out artifacts/runs/exp_01/ranking.csv
```
- Se ja tiver um CSV com `shift_px`, passe `--centroid-csv <arquivo>` para reutilizar.
- Se `preds` nao tiver caminho completo, `--data-root` e usado para localizar os .npz.

## Saidas
- `ranking_tau*.csv` com colunas `key`, `y_true`, probabilidade original, `shift_px` e `score_final`.
- `ranking_tau*.metrics.json` com AUC/AP para cada tau.
- Log impresso com o melhor AUC na lista de tau.

## Detalhes do calculo
- `shift_px` = distancia euclidiana entre centroide do frame mediano e o centroide medio antes do evento (mask `mask_before` ou primeiros 20% dos frames).
- `score_final = proba_pos * exp(-tau * shift_px)`.
