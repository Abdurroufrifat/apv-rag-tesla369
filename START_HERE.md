# Start Here

Complete only the steps in this file first. They establish a reproducible local environment and verify that the starter kit works.

## A. Local setup in VS Code (Windows 11)

1. Extract the project ZIP to a short path, for example:

   `C:\research\apv-rag-tesla369`

2. Open that folder in VS Code: **File → Open Folder**.

3. Open a PowerShell terminal in VS Code: **Terminal → New Terminal**.

   If you prefer the guided automatic setup, run the command below and then skip to Step 8:

   ```powershell
   .\scripts\setup_windows.cmd
   ```

4. Create and activate an isolated Python environment:

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

5. Upgrade pip and install only the lightweight local dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements-local.txt
   ```

6. Run the hardware/environment report:

   ```powershell
   python scripts/check_environment.py
   ```

7. Run the tests:

   ```powershell
   pytest -q
   ```

8. Validate the included claim template (the automatic setup already does this):

   ```powershell
   python scripts/validate_claims.py data/templates/claims_template.csv
   ```

Expected result: the tests pass and the validator reports two valid starter records.

If PowerShell blocks activation, run this command once in the same terminal and repeat Step 4:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

This changes policy only for the current terminal session.

## B. Colab check (after local tests pass)

1. Go to [Google Colab](https://colab.research.google.com/).
2. Upload `notebooks/00_colab_setup.ipynb`.
3. Select **Runtime → Change runtime type → T4 GPU** when available.
4. Run all cells.
5. Save the notebook output to Google Drive or download the executed notebook.

The notebook only checks the cloud environment and installs Phase-1 packages. It does not train a model or incur deliberate paid-cloud usage.

## C. What to send back

Copy or screenshot these three items:

1. the output of `python scripts/check_environment.py`;
2. the output of `pytest -q`; and
3. the Colab GPU line, if you ran the notebook.

Once these pass, the next project step is **Phase 1A: build the search protocol and collect the first 20 Tesla attribution records**.
