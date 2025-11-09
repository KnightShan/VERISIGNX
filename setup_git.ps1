# PowerShell script to set up Git and upload VERISIGNX to GitHub
# Run this script in PowerShell: .\setup_git.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "VERISIGNX GitHub Upload Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Git is installed
Write-Host "Checking Git installation..." -ForegroundColor Yellow
try {
    $gitVersion = git --version
    Write-Host "✓ Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Git is not installed!" -ForegroundColor Red
    Write-Host "Please install Git from: https://git-scm.com/downloads" -ForegroundColor Yellow
    exit 1
}

# Check if Git LFS is installed
Write-Host "Checking Git LFS installation..." -ForegroundColor Yellow
try {
    $lfsVersion = git lfs version
    Write-Host "✓ Git LFS found: $lfsVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Git LFS is not installed!" -ForegroundColor Red
    Write-Host "Please install Git LFS from: https://git-lfs.github.com/" -ForegroundColor Yellow
    Write-Host "Or run: choco install git-lfs" -ForegroundColor Yellow
    exit 1
}

# Initialize Git LFS
Write-Host ""
Write-Host "Initializing Git LFS..." -ForegroundColor Yellow
git lfs install
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Git LFS initialized" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to initialize Git LFS" -ForegroundColor Red
    exit 1
}

# Check if .git exists
Write-Host ""
Write-Host "Checking Git repository..." -ForegroundColor Yellow
if (Test-Path .git) {
    Write-Host "✓ Git repository already initialized" -ForegroundColor Green
} else {
    Write-Host "Initializing Git repository..." -ForegroundColor Yellow
    git init
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Git repository initialized" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to initialize Git repository" -ForegroundColor Red
        exit 1
    }
}

# Check remote
Write-Host ""
Write-Host "Checking remote repository..." -ForegroundColor Yellow
$remote = git remote get-url origin 2>$null
if ($remote) {
    Write-Host "✓ Remote found: $remote" -ForegroundColor Green
    $changeRemote = Read-Host "Do you want to change the remote URL? (y/n)"
    if ($changeRemote -eq "y" -or $changeRemote -eq "Y") {
        git remote set-url origin https://github.com/KnightShan/VERISIGNX.git
        Write-Host "✓ Remote updated" -ForegroundColor Green
    }
} else {
    Write-Host "Adding remote repository..." -ForegroundColor Yellow
    git remote add origin https://github.com/KnightShan/VERISIGNX.git
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Remote added" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to add remote" -ForegroundColor Red
        exit 1
    }
}

# Check .gitattributes
Write-Host ""
Write-Host "Checking .gitattributes..." -ForegroundColor Yellow
if (Test-Path .gitattributes) {
    Write-Host "✓ .gitattributes found" -ForegroundColor Green
} else {
    Write-Host "✗ .gitattributes not found! This is required for Git LFS." -ForegroundColor Red
    Write-Host "Please ensure .gitattributes file exists." -ForegroundColor Yellow
    exit 1
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Review files to be committed: git status" -ForegroundColor White
Write-Host "2. Add all files: git add ." -ForegroundColor White
Write-Host "3. Commit: git commit -m 'Initial commit'" -ForegroundColor White
Write-Host "4. Push to GitHub: git push -u origin main" -ForegroundColor White
Write-Host ""
Write-Host "Note: First push may take a long time due to large files." -ForegroundColor Yellow
Write-Host ""
