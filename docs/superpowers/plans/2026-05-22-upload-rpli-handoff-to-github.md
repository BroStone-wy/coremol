# Upload RPLI Handoff To GitHub Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upload the RPLI handoff materials to `BroStone-wy/coremol` in one dedicated repository folder without committing unrelated experiment files.

**Architecture:** Create a new top-level folder `RPLI-backbone-handoff/` containing the existing `RPLI-backbone/` package and the main adaptation document. Commit and push only this folder plus this execution plan on the current feature branch.

**Tech Stack:** Git, Markdown, existing RPLI code snapshots.

---

### Task 1: Stage Handoff Folder

**Files:**
- Create: `RPLI-backbone-handoff/RPLI-backbone/`
- Create: `RPLI-backbone-handoff/RPLI_CoReMol_Backbone_and_SingleMolecule_Adaptation.md`

- [x] **Step 1: Create the upload folder**

Copy the existing local `RPLI-backbone/` folder and the main handoff Markdown into `RPLI-backbone-handoff/`.

- [x] **Step 2: Verify folder contents**

Run:

```bash
find RPLI-backbone-handoff -type f | sort
```

Expected: includes the main Markdown doc, README, code snapshots, and final config.

Result: verified all expected files under `RPLI-backbone-handoff/`.

### Task 2: Commit And Push

**Files:**
- Commit only:
  - `RPLI-backbone-handoff/`
  - `docs/superpowers/plans/2026-05-22-upload-rpli-handoff-to-github.md`

- [x] **Step 1: Stage only intended files**

Run:

```bash
git add RPLI-backbone-handoff docs/superpowers/plans/2026-05-22-upload-rpli-handoff-to-github.md
```

- [x] **Step 2: Verify staged diff**

Run:

```bash
git diff --cached --stat
```

Expected: only the upload folder and this plan are staged.

Result: staged diff contains only `RPLI-backbone-handoff/` and this upload plan.

- [ ] **Step 3: Commit**

Run:

```bash
git commit -m "docs: add RPLI backbone handoff package"
```

- [ ] **Step 4: Push current branch**

Run:

```bash
git push -u origin coremol-net-affinity-gems
```

Expected: push succeeds and files are visible on the remote branch.

### Task 3: Verify Remote Upload

- [ ] **Step 1: Verify branch contains files**

Run:

```bash
git ls-remote --heads origin coremol-net-affinity-gems
```

Expected: remote branch exists.

- [ ] **Step 2: Report GitHub paths**

Report the GitHub branch URL and the uploaded folder path.
