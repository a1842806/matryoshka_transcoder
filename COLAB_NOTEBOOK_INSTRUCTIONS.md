# 📓 Google Colab Notebook Instructions

## 🚀 How to Use the Notebook

### Step 1: Download the Notebook
1. Go to your GitHub repository: `https://github.com/a1842806/matryoshka_transcoder`
2. Navigate to the `interpretability-evaluation` branch
3. Download `Matryoshka_Transcoder_Training.ipynb`

### Step 2: Upload to Google Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **"Upload"** (📁 icon in the top left)
3. Select the downloaded `Matryoshka_Transcoder_Training.ipynb` file
4. Click **"Open"**

### Step 3: Enable GPU
1. Go to **Runtime** → **Change runtime type**
2. Select **T4 GPU** (free) or **A100** (paid)
3. Click **"Save"**

### Step 4: Run the Notebook
1. **Run All Cells**: Runtime → Run all
2. **Or Run Step by Step**: Click each cell and press Shift+Enter

## 📋 What the Notebook Does

### 🔧 Setup (Cells 1-3)
- ✅ Checks GPU availability
- ✅ Installs all dependencies
- ✅ Clones your repository with nested groups
- ✅ Switches to the correct branch

### 🔐 W&B Login (Cell 4)
- ✅ Optional Weights & Biases setup
- ✅ You can skip this if you don't want online tracking

### 🎯 Training Options (Cells 5-7)
Choose **ONE** of these:

#### Option 1: Layer 17 (Recommended)
- ⏱️ **Time**: ~6 hours on T4 GPU
- 🎯 **Best for**: Research quality results
- 📊 **Steps**: 15,000
- ✨ **Features**: Activation samples, optimized hyperparameters

#### Option 2: Layer 8 (Quick Test)
- ⏱️ **Time**: ~30 minutes on T4 GPU
- 🎯 **Best for**: Testing setup
- 📊 **Steps**: 1,000
- ⚡ **Fast**: Good for quick experiments

#### Option 3: Layer 12 (Alternative)
- ⏱️ **Time**: ~8 hours on T4 GPU
- 🎯 **Best for**: Mid-layer analysis
- 📊 **Steps**: 20,000
- 🔬 **Comprehensive**: Full training run

### 💾 Save Results (Cell 8)
- ✅ Automatically saves to Google Drive
- ✅ Saves checkpoints and activation samples
- ✅ Lists all saved files

## 🎯 Recommended Workflow

### For First Time Users:
1. **Start with Layer 8** (30 min) to test your setup
2. **Then run Layer 17** (6 hours) for final results
3. **Save to Google Drive** when complete

### For Research:
1. **Go straight to Layer 17** (6 hours)
2. **Monitor on W&B dashboard**
3. **Save results** and analyze

## 📊 Monitoring Training

### During Training:
- Watch the progress bar in the cell output
- Check W&B dashboard (link appears in output)
- Look for these metrics:
  - ✅ Loss decreasing
  - ✅ FVU < 0.1 (good reconstruction)
  - ✅ Dead features < 10%

### W&B Dashboard:
Your training automatically logs to Weights & Biases. The dashboard URL will appear in the training output like:
```
W&B dashboard: https://wandb.ai/YOUR_USERNAME/PROJECT_NAME
```

## 💾 Your Results

After training, your files will be saved to:
- **Google Drive**: `/content/drive/MyDrive/matryoshka_checkpoints/`
- **Activation Samples**: `/content/drive/MyDrive/matryoshka_samples/`

## 🆘 Troubleshooting

### "No GPU detected"
1. Go to **Runtime** → **Change runtime type**
2. Select **T4 GPU** or **A100**
3. Click **"Save"** and reconnect

### "Out of memory"
1. Use **Layer 8** instead of Layer 17
2. Or restart runtime and try again

### "Import errors"
1. Re-run the setup cells (cells 1-3)
2. Make sure you're on the `interpretability-evaluation` branch

### "Training too slow"
1. Use **Layer 8** for quick testing (30 min)
2. Make sure GPU is enabled

## 📚 What's Different: Nested Groups

Your transcoder now uses **nested groups** (matching the original Matryoshka paper):

### Before (Your Old Code):
```
Group 0: features [0-1151]       (separate)
Group 1: features [1152-3455]    (separate)
Group 2: features [3456-8063]    (separate)
Group 3: features [8064-18431]   (separate)
```

### After (New Code - Nested):
```
Group 0: features [0-1151]       (base)
Group 1: features [0-3455]       (includes Group 0)
Group 2: features [0-8063]       (includes Groups 0+1)
Group 3: features [0-18431]      (all features)
```

**Benefits:**
- 🎯 Hierarchical learning (coarse → fine)
- 🔄 Each group is a complete model
- 📚 Matches original paper design
- 🧠 Better interpretability

## 🎉 You're Ready!

1. ✅ Download the notebook
2. ✅ Upload to Google Colab
3. ✅ Enable GPU
4. ✅ Run the cells
5. ✅ Choose your training option
6. ✅ Wait for completion
7. ✅ Save to Google Drive

**Happy training! 🚀**

---

## 📞 Support

- **Notebook Issues**: Check the cell outputs for error messages
- **Training Problems**: Try Layer 8 first (30 min test)
- **GitHub Issues**: Open an issue on the repository
- **Documentation**: See other `.md` files in the repository
