#!/bin/bash
# Bash script to set up Git and upload VERISIGNX to GitHub
# Run this script: bash setup_git.sh

echo "========================================"
echo "VERISIGNX GitHub Upload Setup"
echo "========================================"
echo ""

# Check if Git is installed
echo "Checking Git installation..."
if command -v git &> /dev/null; then
    GIT_VERSION=$(git --version)
    echo "✓ Git found: $GIT_VERSION"
else
    echo "✗ Git is not installed!"
    echo "Please install Git from: https://git-scm.com/downloads"
    exit 1
fi

# Check if Git LFS is installed
echo "Checking Git LFS installation..."
if command -v git-lfs &> /dev/null; then
    LFS_VERSION=$(git lfs version)
    echo "✓ Git LFS found: $LFS_VERSION"
else
    echo "✗ Git LFS is not installed!"
    echo "Please install Git LFS:"
    echo "  Ubuntu/Debian: sudo apt-get install git-lfs"
    echo "  macOS: brew install git-lfs"
    echo "  Or visit: https://git-lfs.github.com/"
    exit 1
fi

# Initialize Git LFS
echo ""
echo "Initializing Git LFS..."
git lfs install
if [ $? -eq 0 ]; then
    echo "✓ Git LFS initialized"
else
    echo "✗ Failed to initialize Git LFS"
    exit 1
fi

# Check if .git exists
echo ""
echo "Checking Git repository..."
if [ -d .git ]; then
    echo "✓ Git repository already initialized"
else
    echo "Initializing Git repository..."
    git init
    if [ $? -eq 0 ]; then
        echo "✓ Git repository initialized"
    else
        echo "✗ Failed to initialize Git repository"
        exit 1
    fi
fi

# Check remote
echo ""
echo "Checking remote repository..."
REMOTE=$(git remote get-url origin 2>/dev/null)
if [ $? -eq 0 ] && [ -n "$REMOTE" ]; then
    echo "✓ Remote found: $REMOTE"
    read -p "Do you want to change the remote URL? (y/n): " change_remote
    if [ "$change_remote" = "y" ] || [ "$change_remote" = "Y" ]; then
        git remote set-url origin https://github.com/KnightShan/VERISIGNX.git
        echo "✓ Remote updated"
    fi
else
    echo "Adding remote repository..."
    git remote add origin https://github.com/KnightShan/VERISIGNX.git
    if [ $? -eq 0 ]; then
        echo "✓ Remote added"
    else
        echo "✗ Failed to add remote"
        exit 1
    fi
fi

# Check .gitattributes
echo ""
echo "Checking .gitattributes..."
if [ -f .gitattributes ]; then
    echo "✓ .gitattributes found"
else
    echo "✗ .gitattributes not found! This is required for Git LFS."
    echo "Please ensure .gitattributes file exists."
    exit 1
fi

# Summary
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Review files to be committed: git status"
echo "2. Add all files: git add ."
echo "3. Commit: git commit -m 'Initial commit'"
echo "4. Push to GitHub: git push -u origin main"
echo ""
echo "Note: First push may take a long time due to large files."
echo ""
