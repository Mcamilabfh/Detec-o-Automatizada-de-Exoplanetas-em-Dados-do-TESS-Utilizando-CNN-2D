# Pipeline TESS/TOI (funcional)

Pipeline funcional para baixar cutouts do TESS, gerar janelas POS/NEG, treinar a CNN 2D e produzir rankings/explicacoes. O codigo foi reorganizado em pacotes (`src/projeto_toi`) e os dados ficam separados por estagio (`data/` e `artifacts/`).

## Estrutura
- `src/projeto_toi/`: modulos funcionais (`data`, `datasets`, `models`, `pipeline`, `cli`).
- `scripts/`: wrappers de compatibilidade e utilitarios legados.
- `data/`: `raw/` (FITS/cubos), `processed/` (curvas dobradas), `datasets/windows` (npz POS/NEG), `datasets/images` (PNGs).
- `artifacts/runs/`: saidas de experimento (checkpoints, preds, rankings, gradcam).
- `docs/`: guias tecnicos; `tests/`: unitarios/smoke.

## Ambiente (Miniconda)
```bash
conda env create -f environment.yml
conda activate projeto-toi
python -m pip install -e .          # instala pacote local (opcional mas recomendado)
python -m pip install -r requirements-dev.txt  # se for rodar testes
```

Variaveis de ambiente opcionais para paths: `PROJETO_TOI_ROOT`, `PROJETO_TOI_DATA`, `PROJETO_TOI_ARTIFACTS`.

## Fluxo ponta a ponta
1) **Baixar cutout + quicklooks**
```bash
python -m projeto_toi.cli.download_cutout --target "TIC 307210830" --sector 1 --cutout 15 \
  --outdir data/raw/tic307210830_s1
# ou via coordenadas: --ra <graus> --dec <graus>
```
Saida: `data/raw/<alvo>/data/tpf.fits`, `cube.npy`, `lightcurve.csv` e PNGs em `figs/`.

2) **Pre-processar curva (dobrar pelo periodo)**
```bash
python -m projeto_toi.cli.preprocess_lightcurve --tic "TIC 307210830" --period 4.65 \
  --search-root data/raw/tic307210830_s1 --output-dir data/processed
```
Gera `<tic>_folded.npz` com `phase`/`flux` prontos para janelas ou PNGs.

3) **Gerar janelas POS/NEG**
```bash
python -m projeto_toi.cli.make_windows --outdir data/raw/tic307210830_s1 --tic 307210830 \
  --pos-margin-durations 1.5 --neg-per-pos 2 --min-gap-factor 2.0
```
Cria `data/windows/*.npz` dentro do alvo (usa parametros do TOI ou fornecidos manualmente).

4) **(Opcional) PNGs para visao**  
`python -m projeto_toi.cli.create_image_windows --input data/processed --output data/datasets/images`

5) **Treino da CNN 2D**
```bash
python -m projeto_toi.cli.train_model --data data/datasets/windows --outdir artifacts/runs/exp_01 \
  --epochs 12 --batch-size 16 --pos-weight 2 --lr 5e-5
```
Salva checkpoints, splits e `train_log.json`.

6) **Re-score por centroide (inferencia/ranking)**
```bash
python -m projeto_toi.cli.rescore_predictions \
  --preds artifacts/runs/exp_01/preds.csv \
  --data-root data/datasets/windows \
  --tau-list "0.7,1.5,2,3" \
  --out artifacts/runs/exp_01/ranking.csv
```
Cria `ranking_tau*.csv` e `.metrics.json` com AUC/AP ajustados pelo deslocamento do centroide.

7) **Grad-CAM para interpretabilidade**
```bash
python -m projeto_toi.cli.explain_model \
  --data data/datasets/windows \
  --ckpt artifacts/runs/exp_01/best.pt \
  --preds artifacts/runs/exp_01/ranking_tau2p0.csv \
  --outdir artifacts/runs/exp_01/gradcam --top-k 12
```

## Componentes funcionais
- `projeto_toi.data.download`: download do TESScut e quicklooks.
- `projeto_toi.data.preprocess`: leitura de TPFs e curva dobrada.
- `projeto_toi.data.windows`: janelas POS/NEG com parametros do TOI ou manuais.
- `projeto_toi.data.images`: conversao de janelas em PNGs.
- `projeto_toi.datasets.npz_dataset`: dataset torch com canais estatisticos/mask.
- `projeto_toi.pipeline.train`: treino/validacao; `rescore`: ranking penalizado; `explain`: Grad-CAM.

## Testes
```bash
pytest tests -q
```

## Compatibilidade
Wrappers antigos continuam em `scripts/` (ex.: `scripts/run_one_target.py`, `scripts/train_cnn2d.py`) e redirecionam para os modulos acima. Prefira usar os CLIs via `python -m projeto_toi.cli.<comando>`.


## Dados Brutos (TPF/FITS e TICs)
Dados Brutos do TESS (TPF/FITS)

Este repositório não inclui os arquivos brutos do TESS devido ao tamanho extremamente elevado dos dados. Os Target Pixel Files (TPF) e arquivos FITS associados podem ultrapassar 300 MB por objeto, excedendo o limite máximo permitido pelo GitHub (100 MB por arquivo).

Para manter o projeto leve, acessível e compatível com versionamento, todos os arquivos FITS e pastas das TICs foram movidos para o Google Drive, onde podem ser baixados separadamente quando necessário.

🔗 Acesso aos dados

Os dados brutos utilizados neste trabalho estão disponíveis no Google Drive:

https://drive.google.com/drive/folders/1PZbD6R6dYtRPoNh5EWZlaPODRxuTA8Qz?usp=sharing


Pastas completas de cada TIC utilizada no estudo
Arquivos FITS (TPF, light curves, cutouts, mastDownload, etc.)
Setores individuais de observação
Versões “fast cadence” quando aplicável

## Estrutura esperada localmente

Ao baixar os dados, coloque-os na seguinte estrutura:

data/
  raw/
    raw_data/
      TIC_XXXXXXXX/
        sector_XX/
          mastDownload/
            ... arquivos .fits


O código do pipeline detecta automaticamente essa estrutura ao executar o pré-processamento.

## Como executar o pipeline com os dados brutos

Baixar o conteúdo do Drive

Colocar dentro de data/raw/raw_data/

Garantir que o .gitignore continue ignorando essa pasta

Executar o pipeline normalmente:

python run_pipeline.py \
    --input data/raw/raw_data \
    --output artifacts/results