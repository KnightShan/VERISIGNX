# 📤 Upload Instructions for VERISIGNX

This guide will help you upload your entire VERISIGNX repository to GitHub, including all large files (datasets, images, models) using Git LFS.

## Prerequisites

1. **Git** installed on your system
   - Download from: https://git-scm.com/downloads
   - Verify: `git --version`

2. **Git LFS** (Large File Storage) installed
   - Download from: https://git-lfs.github.com/
   - Verify: `git lfs version`

3. **GitHub Account** with access to https://github.com/KnightShan/VERISIGNX

## Step-by-Step Upload Process

### Step 1: Install Git LFS

**Windows:**
```powershell
# Download and install from https://git-lfs.github.com/
# Or use chocolatey:
choco install git-lfs

# Verify installation
git lfs version
```

**Linux:**
```bash
# Ubuntu/Debian
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# Verify installation
git lfs version
```

**macOS:**
```bash
# Using Homebrew
brew install git-lfs

# Verify installation
git lfs version
```

### Step 2: Initialize Git Repository

Open PowerShell (Windows) or Terminal (Linux/Mac) in your project directory:

```bash
cd "C:\Users\Shantanu\OneDrive\Desktop\Fake Sign Detection"

# Initialize Git repository (if not already initialized)
git init

# Initialize Git LFS
git lfs install
```

### Step 3: Configure Git LFS for Large Files

The `.gitattributes` file is already configured to track:
- Image files (`.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, `.bmp`)
- Model files (`.pkl`)
- All files in Dataset and Results folders

Verify the `.gitattributes` file exists and contains the tracking rules.

### Step 4: Add Remote Repository

```bash
# Add GitHub repository as remote
git remote add origin https://github.com/KnightShan/VERISIGNX.git

# Verify remote
git remote -v
```

### Step 5: Stage All Files

```bash
# Add all files to staging
git add .

# Check what will be committed
git status
```

**Note:** Git LFS will automatically handle large files. You should see messages like "Tracking *.tif with Git LFS" during the add process.

### Step 6: Commit Changes

```bash
# Create initial commit
git commit -m "Initial commit: VERISIGNX - AI Signature Verification System

- Complete detection pipeline (OCR, Line Sweep, Connected Components)
- SVM-based verification system
- Flask web application
- IDRBT Cheque Image Dataset
- Trained model and all dependencies"
```

### Step 7: Push to GitHub

**Important:** For the first push with large files, this may take time depending on your internet speed.

```bash
# Push to GitHub (use main branch)
git branch -M main
git push -u origin main
```

**If you encounter authentication issues:**

1. **Use Personal Access Token:**
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Generate a new token with `repo` permissions
   - Use token as password when prompted

2. **Or use SSH:**
   ```bash
   git remote set-url origin git@github.com:KnightShan/VERISIGNX.git
   git push -u origin main
   ```

### Step 8: Verify Upload

1. Visit https://github.com/KnightShan/VERISIGNX
2. Check that all files are present
3. Verify large files show "Stored with Git LFS" badge
4. Check repository size (should reflect all files)

## Troubleshooting

### Issue: "Git LFS not installed"
**Solution:** Install Git LFS and run `git lfs install` again.

### Issue: "File too large" error
**Solution:** Ensure `.gitattributes` is properly configured and Git LFS is tracking the file types.

### Issue: Push fails due to large files
**Solution:** 
- Check your GitHub account's Git LFS quota (1 GB free)
- For larger datasets, consider using GitHub Releases or external storage
- Or upgrade to GitHub Pro for more LFS storage

### Issue: Slow upload speed
**Solution:**
- Large files take time to upload
- Consider uploading in batches
- Use a stable internet connection

### Issue: Authentication failed
**Solution:**
- Use Personal Access Token instead of password
- Or set up SSH keys for GitHub

## Alternative: Upload Large Files Separately

If you have very large files (>100MB each), consider:

1. **Using GitHub Releases:**
   - Create a release and attach large files as assets
   - Update README with download links

2. **Using External Storage:**
   - Upload dataset to Google Drive, Dropbox, or OneDrive
   - Share download links in README
   - Keep code repository lightweight

3. **Using Git LFS with External Hosting:**
   - Use services like GitLab, Bitbucket, or self-hosted Git LFS

## File Size Estimates

Approximate sizes to expect:
- Dataset folder: ~500MB - 2GB (depending on image count)
- Results folders: ~200MB - 1GB
- Model files: ~10MB - 50MB
- Total repository: ~1GB - 3GB

## Post-Upload Checklist

- [ ] All files visible on GitHub
- [ ] Large files show "Stored with Git LFS" badge
- [ ] README.md displays correctly
- [ ] All code files are present
- [ ] Dataset folder is accessible
- [ ] Model files are tracked
- [ ] Repository is public/private as intended

## Next Steps

After successful upload:

1. Add repository description on GitHub
2. Add topics/tags for discoverability
3. Create a GitHub Release
4. Add license file (MIT License)
5. Enable GitHub Pages if needed
6. Set up GitHub Actions for CI/CD (optional)

## Support

If you encounter issues:
- Check Git LFS documentation: https://git-lfs.github.com/
- GitHub LFS documentation: https://docs.github.com/en/repositories/working-with-files/managing-large-files
- Open an issue on the repository

---

**Note:** The first push may take 30 minutes to several hours depending on:
- Total file size
- Internet connection speed
- Number of files
- GitHub server load

Be patient and ensure your connection is stable!
