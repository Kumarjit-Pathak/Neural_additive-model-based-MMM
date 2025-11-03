# Complete NAM Multi-Agent System Runner
# PowerShell script to run the complete pipeline with monitoring

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "NAM Multi-Agent System - Complete Workflow" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check environment
Write-Host "[1/6] Checking environment..." -ForegroundColor Yellow

if (!(Test-Path ".venv_main")) {
    Write-Host "ERROR: .venv_main not found. Please run setup first." -ForegroundColor Red
    exit 1
}

Write-Host "  Environment found" -ForegroundColor Green

# Step 2: Set Keras backend
Write-Host ""
Write-Host "[2/6] Setting Keras backend to JAX..." -ForegroundColor Yellow
$env:KERAS_BACKEND = "jax"
Write-Host "  KERAS_BACKEND = jax" -ForegroundColor Green

# Step 3: Check configuration
Write-Host ""
Write-Host "[3/6] Checking configuration..." -ForegroundColor Yellow

$config = Get-Content "configs\training_config.yaml" | Select-String "max_epochs"
Write-Host "  $config" -ForegroundColor Green

# Step 4: Show agent status before run
Write-Host ""
Write-Host "[4/6] Initial agent status:" -ForegroundColor Yellow
python scripts\check_agent_status.py

# Step 5: Ask user confirmation
Write-Host ""
Write-Host "[5/6] Ready to run complete pipeline" -ForegroundColor Yellow
Write-Host "  This will execute all 6 agents:" -ForegroundColor White
Write-Host "    Agent 1: Data Engineering" -ForegroundColor White
Write-Host "    Agent 2: Model Architecture" -ForegroundColor White
Write-Host "    Agent 3: Training & Validation" -ForegroundColor White
Write-Host "    Agent 4: Evaluation" -ForegroundColor White
Write-Host "    Agent 5: Business Tools" -ForegroundColor White
Write-Host "    Agent 6: Testing" -ForegroundColor White
Write-Host ""

$response = Read-Host "Do you want to start the pipeline? (yes/no)"

if ($response -ne "yes") {
    Write-Host "Pipeline cancelled by user." -ForegroundColor Yellow
    exit 0
}

# Step 6: Run pipeline
Write-Host ""
Write-Host "[6/6] Starting NAM pipeline..." -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Activate environment and run
& .venv_main\Scripts\activate.ps1
python main.py

# Check exit code
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Green
    Write-Host "Pipeline completed successfully!" -ForegroundColor Green
    Write-Host "======================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Review outputs\figures\ for visualizations" -ForegroundColor White
    Write-Host "  2. Check outputs\models\ for trained model" -ForegroundColor White
    Write-Host "  3. View outputs\nam_pipeline.log for complete log" -ForegroundColor White
    Write-Host "  4. Run 'python scripts\check_agent_status.py' to see final status" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Red
    Write-Host "Pipeline failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "======================================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Check outputs\nam_pipeline.log for errors" -ForegroundColor Yellow
}
