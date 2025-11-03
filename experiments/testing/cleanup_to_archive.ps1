# PowerShell script to move unimportant files to archive folder
# This script archives intermediate/old/duplicate documentation and scripts

# Create archive subdirectories
New-Item -ItemType Directory -Force -Path "archive\old_docs"
New-Item -ItemType Directory -Force -Path "archive\phase_scripts"
New-Item -ItemType Directory -Force -Path "archive\old_notebooks"
New-Item -ItemType Directory -Force -Path "archive\intermediate_fixes"
New-Item -ItemType Directory -Force -Path "archive\old_reports"

Write-Host "Moving files to archive..." -ForegroundColor Green

# Move old/intermediate documentation files
$oldDocs = @(
    "AGENT_SYSTEM_ANALYSIS.md",
    "AGENTS_IMPLEMENTATION_COMPLETE.md",
    "COMPREHENSIVE_PROJECT_STATUS.md",
    "COMPREHENSIVE_REMEDIATION_PLAN.md",
    "CRITICAL_MODEL_ARCHITECTURE_FIX.md",
    "DEVIATION_FROM_ORIGINAL_PLAN.md",
    "ERROR_REPORT.md",
    "FINAL_STATUS_AND_HANDOFF.md",
    "FIX_ALL_STREAMLIT_ERRORS.md",
    "HOW_TO_RUN_WITH_MONITORING.md",
    "PHASE_4_COMPLETION_REPORT.md",
    "PHASES_1_5_COMPLETE.md",
    "PHASES_1_7_ACHIEVEMENT_REPORT.md",
    "PROJECT_SETUP_COMPLETE.md",
    "README_FINAL.md",
    "RECOMMENDED_MMM_ARCHITECTURE.md",
    "RUN_ALL_AGENTS.md",
    "SETUP_GUIDE.md",
    "STREAMLIT_FIXED_GUIDE.md",
    "streamliterror.md",
    "SYSTEM_STATUS_AND_NEXT_STEPS.md",
    "WHAT_YOU_HAVE_AND_NEXT_STEPS.md",
    "WFO.md",
    "github.md"
)

foreach ($file in $oldDocs) {
    if (Test-Path $file) {
        Move-Item -Path $file -Destination "archive\old_docs\" -Force
        Write-Host "  Moved: $file" -ForegroundColor Yellow
    }
}

# Move phase-specific scripts (keeping main.py and main_daily.py)
$phaseScripts = @(
    "phase4_quick_train.py",
    "phase4_train_hierarchical_nam.py",
    "phase4_train_tf_nam.py",
    "phase5_fix_walk_forward.py",
    "phase6_adaptive_training.py",
    "phase6_adaptive_training_simple.py",
    "phase7_business_tools.py",
    "phase8_final_validation.py"
)

foreach ($file in $phaseScripts) {
    if (Test-Path $file) {
        Move-Item -Path $file -Destination "archive\phase_scripts\" -Force
        Write-Host "  Moved: $file" -ForegroundColor Yellow
    }
}

# Move intermediate fix scripts
$fixScripts = @(
    "check_model_architecture.py",
    "verify_architecture.py",
    "generate_diagnostics_only.py",
    "show_data_details.py"
)

foreach ($file in $fixScripts) {
    if (Test-Path $file) {
        Move-Item -Path $file -Destination "archive\intermediate_fixes\" -Force
        Write-Host "  Moved: $file" -ForegroundColor Yellow
    }
}

# Move old notebook creation scripts (keeping the final tutorial notebooks)
$notebookScripts = @(
    "create_tutorial_notebook.py",
    "create_clean_notebook.py",
    "create_remaining_notebooks.py"
)

foreach ($file in $notebookScripts) {
    if (Test-Path $file) {
        Move-Item -Path $file -Destination "archive\old_notebooks\" -Force
        Write-Host "  Moved: $file" -ForegroundColor Yellow
    }
}

# Move old visualization scripts
if (Test-Path "generate_interactive_viz.py") {
    Move-Item -Path "generate_interactive_viz.py" -Destination "archive\intermediate_fixes\" -Force
    Write-Host "  Moved: generate_interactive_viz.py" -ForegroundColor Yellow
}

# Move old HTML files if they exist
$htmlFiles = @("viz_impact.html", "viz_country_impact.html")
foreach ($file in $htmlFiles) {
    if (Test-Path $file) {
        Move-Item -Path $file -Destination "archive\old_reports\" -Force
        Write-Host "  Moved: $file" -ForegroundColor Yellow
    }
}

Write-Host "`nCleanup complete!" -ForegroundColor Green
Write-Host "`nRemaining important files:" -ForegroundColor Cyan

# List remaining important files
Write-Host "`nCore Implementation:" -ForegroundColor Magenta
@("main.py", "main_daily.py", "train_and_diagnose_200epochs.py", "streamlit_app.py") | ForEach-Object {
    if (Test-Path $_) { Write-Host "  - $_" }
}

Write-Host "`nData Pipeline:" -ForegroundColor Magenta
@("fix_data_pipeline.py", "create_marketing_features.py", "fix_feature_mapping.py", "implement_hierarchical_nam.py") | ForEach-Object {
    if (Test-Path $_) { Write-Host "  - $_" }
}

Write-Host "`nKey Documentation:" -ForegroundColor Magenta
@("README.md", "NAM_Comprehensive_Implementation_Plan.md", "COMPLETE_DELIVERABLES.md",
  "FINAL_SUMMARY.md", "PHASE_2_ROADMAP.md", "NEXT_STEPS_ROADMAP.md",
  "START_HERE.md", "README_COMPLETE_SYSTEM.md", "data-details.md") | ForEach-Object {
    if (Test-Path $_) { Write-Host "  - $_" }
}

Write-Host "`nTutorial Notebooks:" -ForegroundColor Magenta
@("NAM_Educational_Tutorial.ipynb", "NAM_MMM_Tutorial_Clean.ipynb", "01_Data_Foundation.ipynb",
  "02_Feature_Engineering.ipynb", "03_Model_Architecture.ipynb", "04_Training_Validation.ipynb",
  "05_Diagnostics_Visualization.ipynb", "06_Business_Applications.ipynb") | ForEach-Object {
    if (Test-Path $_) { Write-Host "  - $_" }
}

Write-Host "`nGuides:" -ForegroundColor Magenta
@("HOW_TO_RUN_VISUALIZATIONS.md", "VISUALIZATION_TOOLS_GUIDE.md",
  "DIAGNOSTIC_PLOTS_GUIDE.md", "MULTI_AGENT_RUN_GUIDE.md",
  "DAILY_DATA_MIGRATION_PLAN.md", "FINAL_PROJECT_SUMMARY.md",
  "INTERACTIVE_VISUALIZATION_GUIDE.md") | ForEach-Object {
    if (Test-Path $_) { Write-Host "  - $_" }
}

Write-Host "`nNotebook Creation Script:" -ForegroundColor Magenta
if (Test-Path "create_notebook_series.py") { Write-Host "  - create_notebook_series.py" }