# KiCad Plugin and Content Manager Packaging

OrthoRoute is a native Python IPC plugin. KiCad 10 supports distributing IPC
plugins through the Plugin and Content Manager (PCM) with metadata schema v2
and a version-level `"runtime": "ipc"` declaration.

## Distribution artifacts

Run:

```powershell
python build.py
```

The build creates two ZIP archives:

- `build/OrthoRoute-<version>-KiCad-PCM.zip` is the end-user package for
  KiCad 10's **Install from File** command.
- `build/OrthoRoute-<version>-KiCad-IPC.zip` is a manual-install archive for
  development and troubleshooting.

The generated files under `build/` are disposable artifacts and are ignored by
Git.

## Install the PCM ZIP

1. Open KiCad 10.
2. Open **Plugin and Content Manager**.
3. Choose **Install from File**.
4. Select `OrthoRoute-<version>-KiCad-PCM.zip`.
5. Enable the API server under **Preferences > Plugins** if it is disabled.
6. Restart PCB Editor.
7. Allow the first-run Python environment installation to finish. The GUI and
   optional acceleration dependencies can take several minutes to install.

OrthoRoute then appears under **Tools > External Plugins** in PCB Editor.

## PCM archive layout

The archive has the layout KiCad expects:

```text
metadata.json
resources/
  icon.png
plugins/
  plugin.json
  kicad_plugin.py
  main.py
  requirements.txt
  orthoroute/
  ...
```

The contents of `plugins/` are the plugin's files directly; there is no extra
identifier directory between `plugins/` and `plugin.json`.

The embedded `metadata.json`:

- uses `https://go.kicad.org/pcm/schemas/v2`;
- contains exactly one version;
- declares a minimum KiCad version of 10.0;
- declares `"runtime": "ipc"`; and
- omits repository-only download URL, checksum, and size fields.

`plugin.json` remains the native IPC descriptor. KiCad uses it to create an
isolated Python environment, install `requirements.txt`, and register the PCB
Editor action.

## Direct development install

For a source checkout:

```powershell
python build.py --deploy --kicad-version 10.0
```

This copies the native plugin to:

```text
Documents/KiCad/10.0/plugins/com.github.bbenchoff.orthoroute/
```

It also installs the compatibility toolbar bridge from PR #17 under KiCad's
`3rdparty/plugins` directory. This deployment mode is for local development;
the PCM ZIP is the distributable package.

## Publishing in KiCad's official repository

An installable ZIP and a searchable PCM catalog entry are separate things.
After making a release:

1. Attach the generated PCM ZIP to a GitHub release.
2. Prepare repository metadata with the archive URL, SHA-256 digest, download
   size, and install size.
3. Submit that metadata to KiCad's official package repository.

The metadata embedded inside the ZIP must continue to omit those repository
download fields.

## Troubleshooting

- Confirm KiCad is version 10 or later.
- Confirm the API server is enabled under **Preferences > Plugins**.
- Restart PCB Editor after installing or updating the package.
- Wait for KiCad's Python environment setup to complete on first launch.
- If the package is rejected immediately, inspect the ZIP root: it must contain
  `metadata.json`, `resources/`, and `plugins/`.
- If the package installs but the action is absent, inspect KiCad's plugin logs
  and the environment created for `com.github.bbenchoff.orthoroute`.

## References

- [KiCad add-on and PCM documentation](https://dev-docs.kicad.org/en/addons/)
- [KiCad IPC API guide for add-on developers](https://dev-docs.kicad.org/en/apis-and-binding/ipc-api/for-addon-developers/)
- [KiCad PCM metadata v2 schema](https://go.kicad.org/pcm/schemas/v2)
- [KiCad IPC plugin manifest v1 schema](https://go.kicad.org/api/schemas/v1)
