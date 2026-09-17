# Publishing this prepared repository

## Repository versus release attachment

Commit the project with Git so that `.gitignore` is honored. Code, processed tables, archived probabilities, documentation and checksums belong in the repository. `data/raw/`, `checkpoints/`, `release_assets/` and generated `outputs/` remain local and are excluded from ordinary commits. The raw inputs and checkpoints are included in the checksummed release attachment; ignoring them in Git does not discard them.

GitHub blocks ordinary Git files larger than 100 MiB. Each release attachment must be smaller than 2 GiB. This project creates independent downloadable parts of one ZIP64 stream, each at most 1,900 MiB, and supplies a reader that restores them without requiring a second concatenated archive on disk. See [GitHub file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) and [release limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

## Local validation

```bash
python scripts/verify_repository.py
python scripts/validate.py --checkpoints
python scripts/data_archive.py verify --directory release_assets
python -m unittest discover -s tests
```

The uploadable repository ZIP is prepared outside the project directory. It contains only repository files; the raw-data attachment is supplied separately. Do not upload the entire local working directory through browser drag-and-drop because that bypasses `.gitignore`.

## Publication sequence

1. Keep `LICENSE` and `docs/THIRD_PARTY_NOTICES.md` with the source distribution (GPL-3.0-only). Specify data/weights/results reuse terms separately.
2. Create the actual GitHub repository and upload the prepared Git contents or the contents of the repository-only ZIP. Do not upload caches, generated output, raw rasters or checkpoints as ordinary Git files.
3. Create a versioned release, for example `v1.0.0`, and attach every `crop-data.zip.NNN` file plus `data_release.json`. All parts are required.
4. Set each part's real download URL in `manifests/data_release.json` and set `published` to true only after upload and download checks succeed. Rebuild the repository manifest after changing tracked files: `python scripts/verify_repository.py --write`.
5. Verify a fresh clone and freshly downloaded release parts, then insert the real repository/release URL or DOI in the manuscript availability statement.

For local restoration:

```bash
python scripts/data_archive.py restore --directory release_assets
```

After real URLs are configured, automatic retrieval is available:

```bash
python scripts/data_archive.py restore --directory release_assets --download
```

The restore command verifies both part hashes and every extracted file; it refuses to overwrite an existing file with different content. It restores the paths already used by `configs/paper.json`.

No remote repository, tag, release or DOI is created by these preparation scripts.
