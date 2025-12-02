# Plugin Manager Integration

This document covers the challenges and solutions for packaging OrthoRoute as a KiCad plugin installable via the Plugin and Content Manager (PCM).

## Overview

OrthoRoute is designed as an IPC API plugin for KiCad 9.0+, which allows it to run in its own Python environment with full control over dependencies (NumPy, PyQt6, CuPy, etc.). However, there are significant challenges getting IPC plugins to work with PCM on Windows.

## Plugin Types in KiCad 9.0

### SWIG Plugins (Legacy)
- Run in KiCad's embedded Python interpreter
- Use `pcbnew.ActionPlugin` base class
- Limited access to dependencies (only what KiCad bundles)
- Install via PCM to `3rdparty/plugins/`
- **Works reliably with PCM** ✅

### IPC Plugins (New in 9.0)
- Run in their own Python virtual environment
- Use `plugin.json` for registration
- Full control over dependencies via `requirements.txt`
- Should install to `plugins/` directory
- **Broken with PCM on Windows** ❌

## The PCM + IPC Bug on Windows

### The Problem

When attempting to install an IPC plugin via PCM with `"runtime": "ipc"` in `metadata.json`:

```json
{
  "versions": [{
    "version": "1.0.0",
    "status": "stable",
    "kicad_version": "9.0",
    "runtime": "ipc"  // This causes a crash!
  }]
}
```

**Result:** KiCad crashes immediately during installation on Windows (tested on KiCad 9.0.1, 9.0.2, 9.0.6).

### What We Tried

1. **Pure IPC Plugin via PCM**
   - Created `plugin.json` with proper schema
   - Set `"runtime": "ipc"` in metadata.json
   - Result: ❌ KiCad crashes during installation

2. **Manual IPC Installation** (bypassing PCM)
   - Extract plugin folder to `C:\Users\<user>\Documents\KiCad\9.0\plugins\`
   - KiCad scans and loads plugin.json
   - Result: ✅ Works! But requires manual installation

3. **SWIG Plugin Running Code Directly**
   - Package as SWIG plugin (no runtime field in metadata)
   - Run OrthoRoute code in KiCad's embedded Python
   - Result: ❌ Missing dependencies (numpy, PyQt6, cupy, psutil, etc.)

4. **SWIG Wrapper + Subprocess Launch**
   - SWIG plugin installs via PCM
   - Launches `python main.py` as subprocess with system Python
   - Result: ❌ Subprocess doesn't get IPC environment variables (KICAD_API_SOCKET, KICAD_API_TOKEN)

## Technical Details

### Why SWIG + Subprocess Doesn't Work

When a SWIG plugin runs, it executes in KiCad's embedded Python environment. This environment does NOT have the IPC API connection credentials:

```
KICAD_API_SOCKET: NOT SET
KICAD_API_TOKEN: NOT SET
```

These environment variables are only set by KiCad when launching true IPC plugins. Since SWIG plugins don't get them, any subprocess launched from a SWIG plugin also won't have them, making IPC connection impossible.

### Plugin Discovery Locations

KiCad scans multiple locations for plugins:

1. **User Plugins (IPC):** `C:\Users\<user>\Documents\KiCad\9.0\plugins\`
   - Manual IPC plugin installations
   - Each plugin in its own subdirectory with `plugin.json`

2. **3rd Party (PCM):** `C:\Users\<user>\Documents\KiCad\9.0\3rdparty\plugins\`
   - PCM-installed plugins
   - Directory name based on identifier (dots → underscores)
   - Example: `com.github.bbenchoff.orthoroute` → `com_github_bbenchoff_orthoroute/`

3. **Scripting (Legacy):** `C:\Users\<user>\Documents\KiCad\9.0\scripting\plugins\`
   - Old SWIG plugin location (still scanned for compatibility)

### Correct File Structures

#### IPC Plugin Structure (Manual Install)
```
com.github.bbenchoff.orthoroute/
├── plugin.json          # IPC plugin registration
├── main.py              # Entry point
├── orthoroute/          # Package code
├── requirements.txt     # Dependencies
└── resources/
    └── icon-24.png      # Toolbar icon (24x24)
    └── icon-48.png      # Toolbar icon (48x48, optional)
```

**plugin.json (Modern schema - v1):**
```json
{
  "$schema": "https://go.kicad.org/api/schemas/v1",
  "identifier": "com.github.bbenchoff.orthoroute",
  "name": "OrthoRoute",
  "description": "GPU-accelerated PCB autorouter",
  "runtime": {
    "type": "python",
    "min_version": "3.10"
  },
  "actions": [{
    "identifier": "orthoroute.run",
    "name": "OrthoRoute",
    "scopes": ["pcb"],
    "entrypoint": "main.py",
    "show-button": true,
    "icons-light": ["icon-24.png", "icon-48.png"]
  }]
}
```

Note: This is the modern schema format based on the official v1 schema and confirmed working in production plugins.

#### SWIG Plugin Structure (PCM Install)
```
<zip root>/
├── metadata.json
├── plugins/
│   ├── __init__.py      # ActionPlugin registration
│   ├── icon.png         # 24x24 toolbar icon
│   └── orthoroute/      # Package code
└── resources/
    └── icon.png         # 64x64 catalog icon
```

**metadata.json (SWIG):**
```json
{
  "$schema": "https://go.kicad.org/pcm/schemas/v1",
  "name": "OrthoRoute",
  "description": "GPU-accelerated PCB autorouter",
  "identifier": "com.github.bbenchoff.orthoroute",
  "type": "plugin",
  "versions": [{
    "version": "1.0.0",
    "status": "stable",
    "kicad_version": "9.0"
    // NO "runtime" field - defaults to "swig"
  }]
}
```

**plugins/__init__.py:**
```python
import pcbnew

class OrthoRoutePlugin(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "OrthoRoute"
        self.category = "Routing"
        self.description = "GPU-accelerated PCB autorouter"
        self.show_toolbar_button = True
        self.icon_file_name = os.path.join(os.path.dirname(__file__), "icon.png")

    def Run(self):
        # Your code here
        pass

OrthoRoutePlugin().register()
```

## Current Solution

Since IPC plugins via PCM don't work on Windows, and SWIG plugins can't access external dependencies, **there is currently no perfect solution for PCM distribution**.

### Recommended Solution: Manual IPC Installation

**This is the confirmed working approach used by production plugins like layout_stamp.**

1. **Manual IPC Installation**
   - Distribute as a ZIP with instructions to extract to `plugins/` folder
   - KiCad manages venv and dependencies automatically via `requirements.txt`
   - Best user experience once installed
   - **Critical:** Users must manually enable IPC API in KiCad settings (Preferences → Plugins)
   - Downside: No "one-click" PCM install

### Alternative Workarounds (Not Recommended)

2. **SWIG + External Script** (Historical - doesn't work)
   - SWIG plugin installs via PCM
   - Downloads/launches separate Python script
   - Requires documenting external Python setup
   - Complex for users

3. **Wait for KiCad Fix**
   - The `"runtime": "ipc"` crash appears to be a KiCad bug
   - May be fixed in future releases
   - Monitor KiCad GitLab issues

## Working Examples

### layout_stamp by Heath Raftery

**Repository:** https://github.com/hraftery/layout_stamp

This is a confirmed working IPC plugin (as of October 2025) that demonstrates proper plugin structure and modern schema usage.

**Key Features:**
- Uses modern `$schema` v1 format
- Multiple actions (copy/paste) in single plugin
- Proper icon integration (24px and 48px)
- Clean requirements.txt with dependencies
- Manual installation only (not via PCM)

**Installation Method:**
Users manually download and extract to:
- Windows: `C:\Users\<username>\Documents\KiCad\9.0\plugins\layout_stamp\`
- macOS: `/Users/<username>/Documents/KiCad/9.0/plugins/layout_stamp/`
- Linux: `~/.local/share/KiCad/9.0/plugins/layout_stamp/`

**Requirements:**
- KiCad 9.0.4+
- Python 3.10+
- IPC API must be manually enabled in KiCad settings (Preferences → Plugins)

This plugin serves as an excellent reference for structuring OrthoRoute's manual distribution.

## Known Issues

### Issue #19465: IPC Python Plugin Loading Broken in Windows
- GitLab: https://gitlab.com/kicad/code/kicad/-/issues/19465
- PCM cannot install IPC plugins on Windows without crashing
- Affects KiCad 9.0.0 through at least 9.0.6
- **Update (January 2025):** Python auto-detection on Windows has been fixed in nightly builds after Jan 1, 2025
- **Status:** Plugin execution and PCM installation still broken as of October 2025
- Workaround: Manual installation to `plugins/` directory

### ActionPlugin Registration in Subprocess
When importing orthoroute code in a subprocess (non-KiCad Python), the `OrthoRoutePlugin().register()` call at the bottom of `kicad_plugin.py` will crash because `pcbnew.ActionPlugin` expects to run in KiCad's context.

**Solution:** Conditional registration:
```python
try:
    import wx
    if wx.GetApp() and wx.GetApp().IsMainLoopRunning():
        OrthoRoutePlugin().register()
except:
    pass  # Don't register if not in KiCad context
```

### Type Hints with CuPy
Type annotations using `cp.ndarray` will fail at import time if CuPy isn't installed or `cp` is None.

**Solution:** Use string annotations:
```python
def my_function() -> 'cp.ndarray':  # Quote the type hint
    ...
```

## Testing Checklist

When testing plugin packaging:

- [ ] Archive extracts with correct structure (plugin.json at root of plugin folder)
- [ ] Icons are 24x24 and optionally 48x48 (PNG format)
- [ ] `identifier` matches between metadata.json and plugin.json (if using PCM)
- [ ] Plugin folder placed in correct location: `Documents\KiCad\9.0\plugins\<plugin-name>\`
- [ ] **IPC API enabled in KiCad settings** (Preferences → Plugins → Enable Python API)
- [ ] KiCad restarted after plugin installation
- [ ] Button appears in PCB Editor toolbar
- [ ] Button appears in Tools → External Plugins menu
- [ ] Clicking button launches the application
- [ ] Application can connect to KiCad IPC API (check console for connection messages)
- [ ] Dependencies are automatically installed to venv by KiCad
- [ ] Check venv at: `C:\Users\<user>\AppData\Local\KiCad\9.0\python-environments\<identifier>\`

## Debugging Tips

### Enable KiCad API Logging

1. Set environment variables:
```
KICAD_ALLOC_CONSOLE=1
KICAD_ENABLE_WXTRACE=1
WXTRACE=KICAD_API
```

2. Add to `kicad_advanced` config:
```
EnableAPILogging=1
```

3. Logs will appear at: `C:\Users\<user>\Documents\KiCad\9.0\logs\api.log`

### Check Plugin Installation

1. **Open Package Directory:** Plugin Manager → "Open Package Directory"
2. **Check structure:** Verify files are in the right locations
3. **Check logs:** Look for Python errors in plugin directory
4. **Check venv:** `C:\Users\<user>\AppData\Local\KiCad\9.0\python-environments\<identifier>\`

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Archive does not contain valid metadata.json" | metadata.json not at ZIP root | Ensure flat ZIP structure |
| KiCad crashes on install | `"runtime": "ipc"` in metadata | Remove runtime field or use manual install |
| Button doesn't appear | Wrong plugin.json schema or location | Check schema and directory structure |
| Button doesn't appear | IPC API not enabled | Enable in Preferences → Plugins → Enable Python API |
| "No module named X" | Missing dependencies | Check requirements.txt; KiCad installs to venv automatically |
| Can't connect to KiCad | Missing IPC env vars | Must use true IPC plugin, not subprocess |
| Plugin not detected | Wrong directory | Must be in `plugins/` not `3rdparty/plugins/` |

## Future Work

- Monitor KiCad releases for PCM+IPC bug fixes
- Consider contributing a patch to KiCad
- Investigate alternate installation methods
- Document manual IPC installation process for users

## References

### Official Documentation
- [KiCad IPC API Documentation](https://dev-docs.kicad.org/en/apis-and-binding/ipc-api/)
- [KiCad IPC API for Add-on Developers](https://dev-docs.kicad.org/en/apis-and-binding/ipc-api/for-addon-developers/)
- [KiCad Addons Documentation](https://dev-docs.kicad.org/en/addons/)
- [KiCad PCM Schema](https://go.kicad.org/pcm/schemas/v1)
- [KiCad IPC Plugin Schema](https://go.kicad.org/api/schemas/v1)
- [kicad-python Library](https://docs.kicad.org/kicad-python-main/)

### Working Examples
- [layout_stamp Plugin](https://github.com/hraftery/layout_stamp) - Confirmed working IPC plugin with modern schema

### Community Resources
- [KiCad 9.0 Python API (IPC API) Forum Thread](https://forum.kicad.info/t/kicad-9-0-python-api-ipc-api/57236)
- [GitLab Issue #19465: IPC Python Plugin Loading Broken in Windows](https://gitlab.com/kicad/code/kicad/-/issues/19465)

---

**Last Updated:** 2025-12-02
**Status:** Manual installation confirmed working. PCM installation remains broken on Windows. Use layout_stamp as reference example.
