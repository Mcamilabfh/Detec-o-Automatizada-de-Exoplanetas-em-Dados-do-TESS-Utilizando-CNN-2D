# Guia de desenvolvimento

## Estrutura atual
- `src/projeto_toi/` – funcoes puras do pipeline (dados, datasets, modelos, treino/pos-treino).
- `scripts/` – wrappers de compatibilidade e utilitarios legados.
- `data/` – `raw/`, `processed/`, `datasets/windows`, `datasets/images`.
- `artifacts/runs/` – checkpoints, preds, rankings, gradcam.
- `tests/` – unitarios e smoke tests em cima do pacote.

## CLIs recomendados
- Download: `python -m projeto_toi.cli.download_cutout --target ...`
- Fold/limpeza: `python -m projeto_toi.cli.preprocess_lightcurve --tic ... --period ...`
- Janelas POS/NEG: `python -m projeto_toi.cli.make_windows --outdir data/raw/<alvo> --tic <id>`
- PNGs: `python -m projeto_toi.cli.create_image_windows --input data/processed`
- Treino: `python -m projeto_toi.cli.train_model --data data/datasets/windows --outdir artifacts/runs/exp_01`
- Re-score: `python -m projeto_toi.cli.rescore_predictions --preds ... --data-root data/datasets/windows`
- Grad-CAM: `python -m projeto_toi.cli.explain_model --data ... --ckpt ... --preds ...`

## Testes
```bash
python -m pip install -r requirements-dev.txt
pytest tests -q
```

## Dicas de contribuicao
- Prefira funcoes puras em `src/projeto_toi` e apenas wrappers finos nos scripts.
- Caminhos padrao vem de `projeto_toi.config` e podem ser sobrescritos por `PROJETO_TOI_*`.
- Mantenda dados fora do controle de versao; use `data/` e `artifacts/` para tudo que for gerado.
