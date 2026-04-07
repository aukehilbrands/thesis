## Daily Workflow

After finishing a meaningful research step, run the following commands in order:

```bash
git status
git add .
git commit -m "..."
git push
```

Example: 
```bash
git commit -m "EDA: target skewness and missingness analysis"
```

## When to Commit:
Commit when you
- Finish an EDA step
- Implement preprocessing logic
- Add feature engineering
- Train or evaluate a model
- Generate figures for the thesis

Do **NOT** commit every small edit or minor change. Commits should reflect logical research steps.

## Good Commit Message Examples:
- EDA: missing value inspection  
- Preprocessing: drop 100% missing columns  
- Feature engineering: population-normalized features  
- Model: baseline elastic net with cross-validation  
- Evaluation: RMSE comparison  

Commit messages should clearly describe the research step so the Git history becomes a structured research log.

## Experiment Branches (Optional but Recommended for Risky Changes):
To test something experimental:
```bash
git checkout -b experiment-name  
```
If it works:
```bash
git checkout main  
git merge experiment-name
```

If it fails:
```bash
git checkout main  
git branch -D experiment-name  
```

## Undo and Inspection Commands:
Discard unstaged changes:
```bash
git checkout .  
```

View commit history:
```bash
git log --oneline
```

## Important Rules:
- Never commit large raw datasets.  
- Never commit secrets such as .env files or API keys.  
- Push changes at least once per working day.  
- Keep commit messages meaningful and research-focused.  

## Recommended Repository Structure:

Thesis/
  datasets/
    archive/   (ignored)
  notebooks/
  src/
  figures/
  README.md
  .gitignore
  GIT_WORKFLOW.md