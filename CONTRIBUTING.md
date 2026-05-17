# Contributing to IndiVoice-DeepASR

First of all, thank you for taking the time to contribute to IndiVoice-DeepASR! Contributions from the open-source community are key to improving speech recognition accuracy for regional Indian accents.

Please review the following guidelines before you get started.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please report any violations or unacceptable behavior to purvanshjoshi17@gmail.com.

## How to Contribute

### 1. Reporting Bugs

Before creating a new bug report, please check the existing Issues to see if the bug has already been reported. If you find a new bug, please open an Issue using the **Bug Report** template and include:
* A clear and concise description of the issue.
* Steps to reproduce the behavior.
* Expected and actual behavior.
* Detailed system logs, traceback, or hardware environment details (e.g., CUDA version, GPU type, and installed library versions).

### 2. Suggesting Enhancements

We welcome ideas for improving the training pipeline, expanding support for more languages/accents, or optimizing inference/system performance. To suggest an enhancement:
* Open an Issue using the **Feature Request** template.
* Explain the utility of the feature and how it can be implemented.
* Provide mockups, references, or code snippets if applicable.

### 3. Submitting Code Changes

If you plan to contribute code (fixing bugs or implementing features):

#### Step 1: Fork and Clone
Fork the repository to your own GitHub account and clone it locally:
```bash
git clone https://github.com/YOUR_USERNAME/IndiVoice-DeepASR.git
cd IndiVoice-DeepASR
```

#### Step 2: Set Up the Development Environment
We recommend creating a virtual environment (conda or venv):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Step 3: Create a Branch
Always create a descriptive branch for your changes:
```bash
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b bugfix/issue-number-description
```

#### Step 4: Write Code and Test
* Keep your code clean, modular, and well-documented.
* Ensure your changes are compatible with distributed multi-GPU environments (Accelerate/DDP) if you are modifying the training pipeline in `src/train.py`.
* Test your changes locally or on a secure cloud notebook.

#### Step 5: Commit and Push
Commit your changes using clear and concise commit messages:
```bash
git add .
git commit -m "feat: add support for Punjabi accent normalization"
git push origin feature/your-feature-name
```

#### Step 6: Create a Pull Request (PR)
* Submit a PR to the `master` branch of the main repository.
* Fill out the Pull Request template completely.
* Link the PR to any related Issues (e.g., `Closes #12`).

---

## Coding Style Guidelines

* **Python Standard**: Follow PEP 8 guidelines for Python code style.
* **Typing**: Use type hints where possible to ensure stability.
* **Docstrings**: Document functions and classes clearly using Google Style docstrings.
* **No Emojis**: Keep commit messages, code comments, and markdown documentation strictly professional and free of emojis.
