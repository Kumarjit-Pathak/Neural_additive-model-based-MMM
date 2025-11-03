# PowerShell script to organize all files into proper folder structure
# This creates a clean, professional project structure

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "ORGANIZING PROJECT INTO PROPER FOLDER STRUCTURE" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

# Create the new folder structure
Write-Host "`nCreating folder structure..." -ForegroundColor Green

$folders = @(
    "notebooks",
    "notebooks\tutorials",
    "notebooks\examples",
    "docs",
    "docs\guides",
    "docs\plans",
    "docs\reports",
    "pipeline",
    "pipeline\data_processing",
    "pipeline\feature_engineering",
    "pipeline\model_implementation",
    "pipeline\training",
    "pipeline\diagnostics",
    "experiments",
    "experiments\main_experiments",
    "experiments\testing"
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
    Write-Host "  Created: $folder" -ForegroundColor Yellow
}

Write-Host "`n" + "=" * 80 -ForegroundColor Cyan
Write-Host "MOVING FILES TO ORGANIZED STRUCTURE" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

# 1. Move notebooks to notebooks folder
Write-Host "`n[1] Organizing Notebooks..." -ForegroundColor Magenta

# Tutorial notebooks
$tutorialNotebooks = @(
    "01_Data_Foundation.ipynb",
    "02_Feature_Engineering.ipynb",
    "03_Model_Architecture.ipynb",
    "04_Training_Validation.ipynb",
    "05_Diagnostics_Visualization.ipynb",
    "06_Business_Applications.ipynb"
)

foreach ($nb in $tutorialNotebooks) {
    if (Test-Path $nb) {
        Move-Item -Path $nb -Destination "notebooks\tutorials\" -Force
        Write-Host "  Moved to tutorials: $nb" -ForegroundColor Green
    }
}

# Main example notebooks
$exampleNotebooks = @(
    "NAM_Educational_Tutorial.ipynb",
    "NAM_MMM_Tutorial_Clean.ipynb",
    "test_minimal.ipynb"
)

foreach ($nb in $exampleNotebooks) {
    if (Test-Path $nb) {
        Move-Item -Path $nb -Destination "notebooks\examples\" -Force
        Write-Host "  Moved to examples: $nb" -ForegroundColor Green
    }
}

# 2. Move documentation files
Write-Host "`n[2] Organizing Documentation..." -ForegroundColor Magenta

# Main docs
$mainDocs = @(
    "README.md",
    "START_HERE.md",
    "README_COMPLETE_SYSTEM.md"
)

foreach ($doc in $mainDocs) {
    if (Test-Path $doc) {
        Move-Item -Path $doc -Destination "docs\" -Force
        Write-Host "  Moved to docs: $doc" -ForegroundColor Green
    }
}

# Planning documents
$planDocs = @(
    "NAM_Comprehensive_Implementation_Plan.md",
    "Agent_Based_Development_Proposal.md",
    "PHASE_2_ROADMAP.md",
    "NEXT_STEPS_ROADMAP.md",
    "data-details.md"
)

foreach ($doc in $planDocs) {
    if (Test-Path $doc) {
        Move-Item -Path $doc -Destination "docs\plans\" -Force
        Write-Host "  Moved to plans: $doc" -ForegroundColor Green
    }
}

# Reports and summaries
$reportDocs = @(
    "COMPLETE_DELIVERABLES.md",
    "FINAL_SUMMARY.md",
    "FINAL_PROJECT_SUMMARY.md",
    "CLEANUP_COMPLETE.md"
)

foreach ($doc in $reportDocs) {
    if (Test-Path $doc) {
        Move-Item -Path $doc -Destination "docs\reports\" -Force
        Write-Host "  Moved to reports: $doc" -ForegroundColor Green
    }
}

# Guides
$guideDocs = @(
    "HOW_TO_RUN_VISUALIZATIONS.md",
    "VISUALIZATION_TOOLS_GUIDE.md",
    "DIAGNOSTIC_PLOTS_GUIDE.md",
    "MULTI_AGENT_RUN_GUIDE.md",
    "DAILY_DATA_MIGRATION_PLAN.md",
    "INTERACTIVE_VISUALIZATION_GUIDE.md"
)

foreach ($doc in $guideDocs) {
    if (Test-Path $doc) {
        Move-Item -Path $doc -Destination "docs\guides\" -Force
        Write-Host "  Moved to guides: $doc" -ForegroundColor Green
    }
}

# 3. Move pipeline scripts
Write-Host "`n[3] Organizing Pipeline Scripts..." -ForegroundColor Magenta

# Data processing
if (Test-Path "fix_data_pipeline.py") {
    Move-Item -Path "fix_data_pipeline.py" -Destination "pipeline\data_processing\" -Force
    Write-Host "  Moved: fix_data_pipeline.py" -ForegroundColor Green
}

# Feature engineering
$featureScripts = @(
    "create_marketing_features.py",
    "fix_feature_mapping.py"
)

foreach ($script in $featureScripts) {
    if (Test-Path $script) {
        Move-Item -Path $script -Destination "pipeline\feature_engineering\" -Force
        Write-Host "  Moved: $script" -ForegroundColor Green
    }
}

# Model implementation
if (Test-Path "implement_hierarchical_nam.py") {
    Move-Item -Path "implement_hierarchical_nam.py" -Destination "pipeline\model_implementation\" -Force
    Write-Host "  Moved: implement_hierarchical_nam.py" -ForegroundColor Green
}

# Training scripts
if (Test-Path "train_and_diagnose_200epochs.py") {
    Move-Item -Path "train_and_diagnose_200epochs.py" -Destination "pipeline\training\" -Force
    Write-Host "  Moved: train_and_diagnose_200epochs.py" -ForegroundColor Green
}

# 4. Move main experiments
Write-Host "`n[4] Organizing Main Experiments..." -ForegroundColor Magenta

$mainScripts = @(
    "main.py",
    "main_daily.py"
)

foreach ($script in $mainScripts) {
    if (Test-Path $script) {
        Move-Item -Path $script -Destination "experiments\main_experiments\" -Force
        Write-Host "  Moved: $script" -ForegroundColor Green
    }
}

# 5. Move utility/test scripts
Write-Host "`n[5] Organizing Utility Scripts..." -ForegroundColor Magenta

$utilityScripts = @(
    "test_system_simple.py",
    "create_notebook_series.py"
)

foreach ($script in $utilityScripts) {
    if (Test-Path $script) {
        Move-Item -Path $script -Destination "experiments\testing\" -Force
        Write-Host "  Moved: $script" -ForegroundColor Green
    }
}

# 6. Move Streamlit app
Write-Host "`n[6] Moving Streamlit App..." -ForegroundColor Magenta

if (Test-Path "streamlit_app.py") {
    Move-Item -Path "streamlit_app.py" -Destination "pipeline\diagnostics\" -Force
    Write-Host "  Moved: streamlit_app.py to diagnostics" -ForegroundColor Green
}

# 7. Clean up PowerShell scripts - move to experiments/testing
Write-Host "`n[7] Cleaning Up Scripts..." -ForegroundColor Magenta

$psScripts = @(
    "run_complete_system.ps1",
    "cleanup_to_archive.ps1"
)

foreach ($script in $psScripts) {
    if (Test-Path $script) {
        Move-Item -Path $script -Destination "experiments\testing\" -Force
        Write-Host "  Moved: $script" -ForegroundColor Green
    }
}

# Move this organize script too
if (Test-Path "organize_structure.ps1") {
    Copy-Item -Path "organize_structure.ps1" -Destination "experiments\testing\" -Force
    Write-Host "  Copied: organize_structure.ps1" -ForegroundColor Green
}

# 8. Create main README.md at root
Write-Host "`n[8] Creating Root README..." -ForegroundColor Magenta

$readmeContent = @"
# Neural Additive Model for Marketing Mix Modeling

## Project Structure

\`\`\`
Neural-Additive_Model/
├── src/                        # Core source code
│   ├── models/                 # Model implementations
│   ├── data/                   # Data loading and processing
│   ├── training/               # Training utilities
│   ├── evaluation/             # Evaluation metrics
│   ├── optimization/           # Business optimization tools
│   └── visualization/          # Visualization utilities
│
├── pipeline/                   # Main pipeline components
│   ├── data_processing/        # Data pipeline scripts
│   ├── feature_engineering/    # Feature creation scripts
│   ├── model_implementation/   # Model building scripts
│   ├── training/              # Training scripts
│   └── diagnostics/           # Streamlit and diagnostics
│
├── notebooks/                  # Jupyter notebooks
│   ├── tutorials/             # 6-part tutorial series
│   └── examples/              # Example implementations
│
├── experiments/               # Experimental scripts
│   ├── main_experiments/      # Main training scripts
│   └── testing/              # Test and utility scripts
│
├── docs/                      # Documentation
│   ├── guides/               # How-to guides
│   ├── plans/                # Project plans
│   └── reports/              # Status reports
│
├── configs/                   # Configuration files
├── scripts/                   # Utility scripts
├── models/                    # Saved models
├── plots/                     # Generated plots
├── outputs/                   # Output files
└── archive/                   # Archived old files
\`\`\`

## Quick Start

1. **Start Here**: See \`docs/START_HERE.md\`
2. **Run Training**: \`python pipeline/training/train_and_diagnose_200epochs.py\`
3. **View Dashboard**: \`streamlit run pipeline/diagnostics/streamlit_app.py\`
4. **Learn NAM-MMM**: Open notebooks in \`notebooks/tutorials/\`

## Key Features

- Neural Additive Model with Marketing Mix Modeling
- 28+ Beta-Gamma features for marketing saturation
- Hierarchical structure with category/subcategory pooling
- Walk-forward validation
- Comprehensive diagnostic plots
- Interactive Streamlit dashboard

## Current Performance

- **Data points**: 4,381 (250 daily aggregated)
- **Features**: 99 (28 Beta-Gamma)
- **Best validation loss**: 0.0032
- **Training**: 200 epochs completed

## Documentation

- **Planning**: \`docs/plans/\`
- **Guides**: \`docs/guides/\`
- **Reports**: \`docs/reports/\`
"@

$readmeContent | Out-File -FilePath "README.md" -Encoding UTF8
Write-Host "  Created: README.md at root" -ForegroundColor Green

Write-Host "`n" + "=" * 80 -ForegroundColor Cyan
Write-Host "FOLDER ORGANIZATION COMPLETE!" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan

Write-Host "`nNew Structure Summary:" -ForegroundColor Yellow
Write-Host "  /src              - Source code (unchanged)" -ForegroundColor White
Write-Host "  /pipeline         - Main pipeline components" -ForegroundColor White
Write-Host "  /notebooks        - All Jupyter notebooks" -ForegroundColor White
Write-Host "  /experiments      - Experimental scripts" -ForegroundColor White
Write-Host "  /docs             - All documentation" -ForegroundColor White
Write-Host "  /configs          - Configuration files" -ForegroundColor White
Write-Host "  /scripts          - Utility scripts" -ForegroundColor White
Write-Host "  /models           - Saved models" -ForegroundColor White
Write-Host "  /plots            - Generated plots" -ForegroundColor White
Write-Host "  /outputs          - Output files" -ForegroundColor White
Write-Host "  /archive          - Archived files" -ForegroundColor White

Write-Host "`nNext: Run test_organized_system.py to verify everything works!" -ForegroundColor Cyan