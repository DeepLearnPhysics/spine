# Console Script Naming Strategy - Final Decision

## ✅ Final Configuration

### GitHub Repo: `spine` (Keep as-is)
- ✅ **Established identity**: Project known as "SPINE"
- ✅ **Stable URLs**: No breaking changes to links, documentation
- ✅ **Clean branding**: Represents the SPINE project as a whole
- ✅ **Common pattern**: Many projects have different repo vs PyPI names

### PyPI Packages: `spine-ml` & `spine-ml-core` 
- ✅ **Avoids conflicts**: No collision with existing `spine` PyPI package (from 2012)
- ✅ **Descriptive**: Clearly ML-focused packages
- ✅ **Multiple variants**: Clean separation between full and core versions

### Console Command: `spine`
- ✅ **User-friendly**: Short, easy to type
- ✅ **Natural**: Matches project name and user expectations  
- ✅ **No conflicts**: PyPI package names ≠ console command names
- ✅ **Professional**: Follows industry patterns

## 🎯 Final User Experience

```bash
# Install package (PyPI name)
pip install spine-ml[full,viz]

# Use command (simple name)
spine --config config/train_uresnet.cfg --source data.h5
```

## 📋 Similar Patterns in Popular Tools

This is a very common and accepted pattern:

| GitHub Repo | PyPI Package | Console Command |
|-------------|--------------|-----------------|
| `pytorch/pytorch` | `torch` | `python -m torch` |
| `scikit-learn/scikit-learn` | `scikit-learn` | (import `sklearn`) |
| `psf/requests` | `requests` | (library only) |
| `pallets/flask` | `Flask` | `flask` |

## 🚀 Benefits of This Approach

1. **Best of all worlds**:
   - Stable repo identity (`spine`)
   - Clear PyPI namespacing (`spine-ml`)  
   - Simple user commands (`spine`)

2. **Future-proof**:
   - No conflicts with existing packages
   - Room for expansion (spine-viz, spine-analysis, etc.)
   - Professional naming strategy

3. **User-centric**:
   - Easy installation: `pip install spine-ml`
   - Easy usage: `spine --help`
   - Clear documentation and examples

Perfect! This gives users the cleanest possible experience. 🎯