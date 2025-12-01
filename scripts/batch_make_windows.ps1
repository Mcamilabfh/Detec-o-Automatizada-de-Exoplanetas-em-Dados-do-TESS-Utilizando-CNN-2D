param(
  [string]$Root = "$PWD",                             # raiz do projeto (onde há várias pastas de TIC)
  [double]$MinPeriod = 0.5,
  [double]$MaxPeriod = 20.0,
  [double]$MinDurHours = 0.5,
  [double]$MaxDurHours = 8.0,
  [int]$NegPerPos = 4,
  [double]$PosMarginDur = 2.0,
  [double]$MinGapFactor = 1.25,
  [int]$SamplesPerPeak = 10,
  [int]$NDurations = 25
)

# 1) Encontrar todos os tpf.fits (esperado em ...\<TIC>\data\tpf.fits)
$tpfFiles = Get-ChildItem -Recurse -Path $Root -Filter "tpf.fits" -ErrorAction SilentlyContinue

if ($tpfFiles.Count -eq 0) {
  Write-Host "Nenhum tpf.fits encontrado em $Root. Estrutura esperada: <TIC>\data\tpf.fits"
  exit 1
}

# 2) Para cada TPF, rodar o gerador usando --outdir como a pasta do TIC (pai do 'data')
foreach ($f in $tpfFiles) {
  $dataDir = Split-Path $f.FullName -Parent        # ...\<TIC>\data
  $ticDir  = Split-Path $dataDir -Parent           # ...\<TIC>
  Write-Host "Processando: $ticDir"

  python scripts\make_windows_auto_bls.py `
    --outdir "$ticDir" `
    --min_period $MinPeriod `
    --max_period $MaxPeriod `
    --min_duration_hours $MinDurHours `
    --max_duration_hours $MaxDurHours `
    --neg_per_pos $NegPerPos `
    --pos_margin_durations $PosMarginDur `
    --min_gap_factor $MinGapFactor `
    --samples_per_peak $SamplesPerPeak `
    --n_durations $NDurations
}

# 3) Gerar lista mestre com TODOS os .npz das janelas (pos/neg) criadas
$npz = Get-ChildItem -Recurse -Path $Root -Include *.npz -ErrorAction SilentlyContinue `
       | Where-Object { $_.FullName -match "\\data\\windows_auto\\" -or $_.FullName -match "/data/windows_auto/" } `
       | ForEach-Object { $_.FullName }

$listsDir = "runs\exp_windows_auto\lists"
New-Item -ItemType Directory -Force -Path $listsDir | Out-Null

$allList = Join-Path $listsDir "all_windows.txt"
$npz | Set-Content $allList -Encoding UTF8

Write-Host "`n[OK] Lista mestre gerada em: $allList"
