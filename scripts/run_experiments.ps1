# Full experiment plan for Windows PowerShell. Resumable: finished runs (best.pt) and finished
# evaluations (.json) are skipped, so you can stop with Ctrl+C and start it again later.
#   .\scripts\run_experiments.ps1                 # everything
#   .\scripts\run_experiments.ps1 -Only isbi      # just one block: isbi | aariz | ablation | eval | tables
#   .\scripts\run_experiments.ps1 -Seeds 0         # one seed only (e.g. to fit a deadline)
param([string]$Only = "all", [int[]]$Seeds = @(0, 1, 2))
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"


function Train($config, $algo, $seed) {
    $out = "runs/${config}_${algo}_s${seed}"
    if (Test-Path "$out/best.pt") { Write-Host "skip $out (done)"; return }
    if (Test-Path $out) { Remove-Item -Recurse -Force $out }   # half-finished run: start it again
    python scripts/train.py --config "configs/$config.yaml" --seed $seed
    if ($LASTEXITCODE -ne 0) { throw "training failed: $config seed $seed" }
}

function Evaluate($run, $split, $dataRoot) {
    $name = Split-Path $dataRoot -Leaf
    if (Test-Path "$run/eval_${name}_$split.json") { return }
    python scripts/evaluate.py --checkpoint "$run/best.pt" --data-root $dataRoot --split $split
    if ($LASTEXITCODE -ne 0) { throw "evaluation failed: $run $split" }
}

# --- training (ISBI ~11 h, Aariz ~15 h, ablations ~5 h on an RTX 4050) -----------------------
$methods = @(@("dqn", "dqn"), @("mappo", "mappo"), @("coma", "coma"), @("coma_ind", "coma"), @("unet", "unet"))
foreach ($ds in "isbi", "aariz") {
    if ($Only -ne "all" -and $Only -ne $ds) { continue }
    foreach ($m in $methods) { foreach ($s in $Seeds) { Train "${ds}_$($m[0])" $m[1] $s } }
}
if ($Only -eq "all" -or $Only -eq "ablation") {
    foreach ($ab in @(@("ablation_noshare", "dqn"), @("ablation_randinit", "dqn"),
                      @("ablation_singlescale", "dqn"), @("ablation_team_reward", "mappo"))) {
        foreach ($s in $Seeds) { Train $ab[0] $ab[1] $s }
    }
}

# --- evaluation and flagging -----------------------------------------------------------------
if ($Only -eq "all" -or $Only -eq "eval") {
    $finished = @(Get-ChildItem runs -Directory | Where-Object { Test-Path "$($_.FullName)/best.pt" })
    foreach ($run in $finished) {
        $r = "runs/$($run.Name)"
        if ($run.Name -like "aariz_*") {
            foreach ($sp in "val", "test") { Evaluate $r $sp "data/aariz" }
            python scripts/flag.py --val "$r/eval_aariz_val.npz" --test "$r/eval_aariz_test.npz" | Out-Null
        } else {
            foreach ($sp in "val", "test1", "test2") { Evaluate $r $sp "data/isbi" }
            if ($run.Name -like "isbi_*") { Evaluate $r "test" "data/aariz" }   # cross-dataset
            foreach ($sp in "test1", "test2") {
                python scripts/flag.py --val "$r/eval_isbi_val.npz" --test "$r/eval_isbi_$sp.npz" | Out-Null
            }
        }
    }
    foreach ($sig in "spread", "conf", "shape") {                            # single-signal ablation
        foreach ($run in @(Get-ChildItem runs -Directory -Filter "isbi_dqn_dqn_s*")) {
            python scripts/flag.py --val "runs/$($run.Name)/eval_isbi_val.npz" --test "runs/$($run.Name)/eval_isbi_test1.npz" --signals $sig | Out-Null
        }
    }
}

# --- tables + significance -------------------------------------------------------------------
if ($Only -eq "all" -or $Only -eq "tables") {
    $main = @("--method", "DQN", "runs/{0}_dqn_dqn_s*", "--method", "MAPPO", "runs/{0}_mappo_mappo_s*",
              "--method", "COMA", "runs/{0}_coma_coma_s*", "--method", "COMA indiv", "runs/{0}_coma_ind_coma_s*",
              "--method", "U-Net", "runs/{0}_unet_unet_s*")
    foreach ($sp in "isbi_test1", "isbi_test2") {
        $m = ($main | ForEach-Object { $_ -f "isbi" }) + @(
            "--method", "DQN no sharing", "runs/ablation_noshare_dqn_s*",
            "--method", "DQN random init", "runs/ablation_randinit_dqn_s*",
            "--method", "DQN single scale", "runs/ablation_singlescale_dqn_s*",
            "--method", "MAPPO team reward", "runs/ablation_team_reward_mappo_s*")
        python scripts/compare.py --split $sp @m
    }
    python scripts/compare.py --split aariz_test @($main | ForEach-Object { $_ -f "aariz" })
    python scripts/compare.py --split aariz_test --out results/cross_dataset @($main | ForEach-Object { $_ -f "isbi" })
    # copy results + every run's json/jsonl (not the large best.pt) into share/, keeping folders
    if (Test-Path share) { Remove-Item -Recurse -Force share }
    Copy-Item -Recurse results share/results
    foreach ($f in @(Get-ChildItem runs -Recurse -File -Include *.json, *.jsonl)) {
        $dest = Join-Path "share/runs" $f.Directory.Name
        New-Item -ItemType Directory -Force $dest | Out-Null
        Copy-Item $f.FullName $dest
    }
    Compress-Archive -Path share\* -DestinationPath final_results.zip -Force
    Write-Host "done: tables in results/, everything to share in final_results.zip"
}
