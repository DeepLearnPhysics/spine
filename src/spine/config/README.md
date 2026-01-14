# SPINE Configuration System

The SPINE configuration system provides advanced YAML configuration management with file composition, parameter overrides, and metadata-based version control.

## Table of Contents

- [Quick Start](#quick-start)
- [Core Features](#core-features)
- [File Composition](#file-composition)
- [Parameter Overrides](#parameter-overrides)
- [Removing Keys](#removing-keys)
- [Command Line](#command-line)
- [Metadata System](#metadata-system)
- [Configuration Types](#configuration-types)
- [Version Compatibility](#version-compatibility)
- [Modifier System](#modifier-system)
- [Complete Examples](#complete-examples)
- [API Reference](#api-reference)

## Quick Start

```python
from spine.config import load_config

# Load a simple config
config = load_config('config.yaml')

# Load with command-line overrides
config = load_config('config.yaml', overrides=['io.batch_size=16', 'model.debug=true'])
```

```yaml
# config.yaml
__meta__:
  version: "260107"
  description: "My training configuration"

include: base_config.yaml

override:
  io.loader.batch_size: 16
  model.learning_rate: 0.001

model:
  name: full_chain
  num_layers: 5
```

## Core Features

### File Composition with `include:`

Compose configurations from multiple files:

```yaml
# Single include
include: base_config.yaml

# Multiple includes (processed in order)
include:
  - base/base.yaml
  - io/io.yaml
  - model/model.yaml
  - post/post.yaml
```

**Inline includes** with `!include` tag:

```yaml
model:
  backbone: !include model/backbone.yaml
  head: !include model/head.yaml
```

#### Include Path Resolution

SPINE searches for included files in this order:

1. **Absolute paths**: Used as-is if they exist
2. **Relative paths**: Resolved relative to the including config file
3. **SPINE_CONFIG_PATH**: Searches through directories specified in the `SPINE_CONFIG_PATH` environment variable

**Extension handling**: If a file isn't found, SPINE automatically tries adding `.yaml` or `.yml` extensions.

**Example with SPINE_CONFIG_PATH**:

```bash
# Set shared config directories
export SPINE_CONFIG_PATH="/usr/local/spine/configs:/home/user/shared_configs"

# Now configs can include files from these directories without absolute paths
```

```yaml
# config.yaml - can include files from SPINE_CONFIG_PATH
include:
  - base/default.yaml      # Found in /usr/local/spine/configs/base/default.yaml
  - detectors/icarus       # Auto-adds .yaml, finds in /home/user/shared_configs/detectors/icarus.yaml
  
io:
  reader:
    batch_size: 32
```

**Multiple search paths**: Separate paths with `:` (like `PATH` or `PYTHONPATH`):

```bash
export SPINE_CONFIG_PATH="$HOME/.spine/configs:/opt/spine/configs:/shared/configs"
```

**Path precedence**: Relative paths (from the config's directory) always take precedence over `SPINE_CONFIG_PATH` entries. This allows local overrides of shared configs.

### Parameter Overrides with `override:`

Override specific parameters using dot notation:

```yaml
include: base_config.yaml

override:
  io.loader.batch_size: 16
  io.loader.num_workers: 8
  model.learning_rate: 0.001
  model.modules.uresnet.depth: 5
```

**List operations**:

```yaml
override:
  # Append to list
  parsers+: [meta, run_info]
  
  # Replace entire list
  parsers: [sparse3d, cluster3d]
```

### Removing Keys

Three ways to remove unwanted keys from included configurations:

#### Method 1: `null` in `override:`

```yaml
include: base_config.yaml

override:
  io.loader.shuffle: null        # Remove entirely
  base.debug_mode: null          # Remove entirely
  io.loader.batch_size: 16       # Override normally
```

#### Method 2: `remove:` directive

```yaml
include: base_config.yaml

# Remove single key
remove: io.loader.shuffle

# Remove multiple keys
remove:
  - io.loader.shuffle
  - base.debug_mode
  - io.loader.num_workers
```

#### Method 3: Combine both

```yaml
include: base_config.yaml

remove:
  - io.loader.shuffle
  - base.debug_mode

override:
  io.loader.num_workers: null    # Delete
  io.loader.batch_size: 16       # Override
```

**Order of operations:**
1. Load included files (with recursive includes)
2. Merge main config
3. Apply `remove:` deletions
4. Apply `override:` (including null deletions and regular overrides)

### Command Line

Override any parameter from command line:

```bash
# Single override
spine --config config.yaml --set io.batch_size=16

# Multiple overrides
spine --config config.yaml \
  --set io.batch_size=16 \
  --set model.learning_rate=0.001 \
  --set model.debug=true
```

## Metadata System

Add version control and compatibility checking via `__meta__` blocks:

```yaml
__meta__:
  # Identity
  version: "260107"              # YYMMDD format
  date: "2026-01-07"
  name: "my_config"
  description: "Clear description"
  tags: [production, latest]
  
  # Type and behavior
  kind: "bundle"                 # "bundle" or "mod"
  strict: "error"                # "error" or "warn"
  list_append: "unique"          # "unique" or "append"
  
  # Compatibility
  compatible_with:
    base: "==260107"
    io: ">=260101"
  
  # Modifier fields
  priority: 10                   # Lower = applied first
  applies_to: [mc, data]
  requires: []
  conflicts_with: []
  
  # Model weights
  weights:
    path: /path/to/model.ckpt

# Actual configuration
base:
  world_size: 1
```

### Version Format

Versions must be **exactly 6 digits** in `YYMMDD` format:

- ✓ Valid: `"260107"`, `"240719"`
- ✗ Invalid: `"1.0.0"`, `"26.01.07"`, `"2026-01-07"`

### Metadata Fields

| Field | Type | Purpose |
|-------|------|---------|
| `version` | string | Config version (YYMMDD) |
| `date` | string | Creation date (ISO format) |
| `name` | string | Config/modifier name |
| `description` | string | Human-readable description |
| `tags` | list[str] | Categorization tags |
| `kind` | string | `"bundle"` or `"mod"` |
| `strict` | string | `"error"` or `"warn"` |
| `list_append` | string | `"append"` or `"unique"` |
| `compatible_with` | dict | Version requirements |
| `extends` | string | Modifier category |
| `priority` | int | Modifier application order |
| `applies_to` | list[str] | Data types (`mc`, `data`) |
| `requires` | list[str] | Required modifiers |
| `conflicts_with` | list[str] | Conflicting modifiers |
| `weights` | dict | Model weight paths |
| `components` | dict | Component versions |

## Configuration Types

### Bundles (`kind: "bundle"`)

Complete, standalone configurations that can be run directly:

```yaml
# icarus_production.yaml
__meta__:
  kind: "bundle"
  version: "260107"
  strict: "error"        # Fail on missing paths
  description: "Complete ICARUS production config"

include:
  - base/base.yaml
  - io/io.yaml
  - model/model.yaml
  - post/post.yaml

base:
  world_size: 1
  iterations: 1000
```

**Characteristics:**
- Self-contained and complete
- Run directly: `spine --config my_bundle.yaml`
- Default `strict: "error"` (fail on invalid overrides)
- Typically include full base configuration

### Modifiers (`kind: "mod"`)

Partial configurations designed to be included into bundles:

```yaml
# gpu_optimizations.yaml
__meta__:
  kind: "mod"
  version: "260107"
  strict: "warn"         # Warn on missing paths
  description: "GPU memory optimizations"
  priority: 20
  applies_to: [mc, data]

override:
  io.loader.batch_size: 64
  io.loader.num_workers: 16
  model.use_mixed_precision: true
```

**Characteristics:**
- Incomplete by itself
- Meant to be included: `include: my_mod.yaml`
- Default `strict: "warn"` (allow optional modifications)
- Contains overrides or specialized settings

### Strict Mode

Controls behavior when overriding non-existent paths:

- **`strict: "error"`** (bundles): Raise exception on missing path
- **`strict: "warn"`** (modifiers): Issue warning, continue processing

```yaml
__meta__:
  kind: "bundle"
  strict: "error"    # Catch configuration errors early

# vs.

__meta__:
  kind: "mod"
  strict: "warn"     # Allow optional modifications
```

### List Append Mode

Controls duplicate behavior when appending to lists:

- **`list_append: "append"`**: Allow duplicates (faster)
- **`list_append: "unique"`**: Prevent duplicates (safer)

```yaml
__meta__:
  list_append: "unique"

override:
  parsers+: [meta, run_info]  # Won't duplicate if already present
```

## Version Compatibility

### Compatibility Constraints

Declare version requirements to ensure configs work together:

```yaml
# model/model_260107.yaml
__meta__:
  version: "260107"
  compatible_with:
    base: "==260107"      # Exact version
    io: ">=260101"        # Minimum version
    post: ">=260101"      # Minimum version
```

### Supported Operators

- `==` - Exact match
- `>=` - Minimum version
- `<=` - Maximum version
- `>` - Greater than
- `<` - Less than
- `!=` - Not equal

### Deferred Validation

The loader performs **deferred compatibility checking** to support forward references:

```yaml
# top_level.yaml
include:
  - base/base_260107.yaml
  - io/io_260107.yaml
  - model/model_260107.yaml    # Requires "post >= 260107"
  - post/post_260107.yaml       # Loaded AFTER model
```

**Process:**
1. Load all includes recursively
2. Accumulate component versions automatically
3. Validate all compatibility constraints together

This allows components to reference each other regardless of include order.

### Component Inference

The loader automatically infers component names from directory structure:

```yaml
# base/base_260107.yaml
__meta__:
  version: "260107"
# → Automatically registers: components: {base: "260107"}

# io/io_260107.yaml  
__meta__:
  version: "260107"
# → Automatically registers: components: {io: "260107"}
```

## Modifier System

### Priority-Based Application

Modifiers are applied in order of priority (lower = first):

```yaml
# mod_data.yaml
__meta__:
  name: "data"
  priority: 10        # Applied first

# mod_calibration.yaml
__meta__:
  name: "4ms"
  priority: 20        # Applied second

# mod_lite.yaml
__meta__:
  name: "lite"
  priority: 50        # Applied last
```

### Data Type Targeting

Specify which data types a modifier supports:

```yaml
__meta__:
  name: "numi"
  applies_to: [mc, data]    # Both Monte Carlo and real data

# vs.

__meta__:
  name: "data"
  applies_to: [data]        # Real data only
```

### Dependencies and Conflicts

Declare modifier relationships:

```yaml
__meta__:
  name: "unblind"
  requires: [data]           # Must apply 'data' first
  conflicts_with: []

# vs.

__meta__:
  name: "4ms"
  requires: []
  conflicts_with: [8ms]      # Mutually exclusive
```

## Complete Examples

### Basic Training Config

```yaml
# train.yaml
__meta__:
  version: "260107"
  date: "2026-01-07"
  description: "Basic training configuration"
  kind: "bundle"

include: base_config.yaml

override:
  io.loader.batch_size: 16
  model.learning_rate: 0.001

model:
  name: uresnet
  depth: 5
  filters: 16
```

### Production Config with Modifiers

```yaml
# icarus_production.yaml
__meta__:
  name: "icarus_full_chain_260107"
  version: "260107"
  date: "2026-01-07"
  description: "ICARUS production with data modifiers"
  kind: "bundle"
  components:
    base: "260107"
    io: "260107"
    model: "260107"
    post: "260107"

include:
  - base/base_260107.yaml
  - io/io_260107.yaml
  - model/model_260107.yaml
  - post/post_260107.yaml
  - modifier/mod_data_260107.yaml
  - modifier/mod_4ms_260107.yaml
```

### Data Modifier

```yaml
# mod_data_260107.yaml
__meta__:
  name: "data"
  version: "260107"
  extends: "data"
  date: "2026-01-07"
  priority: 10
  applies_to: [data]

override:
  post.apply_calibrations:
    gain:
      gain: [76.02, 75.29, 77.36, 77.1]
    recombination:
      model: mbox
      efield: 0.4938
    lifetime:
      lifetime_db: /path/to/lifetime.db
```

### Model Config with Weights

```yaml
# model_260107.yaml
__meta__:
  version: "260107"
  date: "2026-01-07"
  description: "Full chain model with trained weights"
  compatible_with:
    base: "==260107"
    io: ">=260107"
    post: ">=260107"
  weights:
    path: /path/to/weights/snapshot-7999.ckpt

include: model_common.yaml

override:
  model.weight_path: /path/to/weights/snapshot-7999.ckpt
  model.modules.chain.use_checkpointing: true
```

### Removing Unwanted Features

```yaml
# production.yaml
include: development_config.yaml

# Remove development settings
remove:
  - base.debug_mode
  - io.loader.shuffle
  - model.detect_anomaly

override:
  # Set production values
  io.loader.num_workers: 16
  base.log_level: warning
  
  # Remove optional features
  model.profiler: null
  model.visualizer: null
```

## API Reference

### Functions

#### `load_config(path, overrides=None, strict=None)`

Load and process a YAML configuration file.

**Parameters:**
- `path` (str): Path to YAML config file
- `overrides` (list[str], optional): Command-line style overrides (`["key.path=value"]`)
- `strict` (str, optional): Override strict mode (`"error"` or `"warn"`)

**Returns:**
- `dict`: Processed configuration

**Raises:**
- `ConfigError`: Base exception for all config errors
- `ConfigIncludeError`: Include file not found or invalid
- `ConfigCycleError`: Circular include detected
- `ConfigPathError`: Invalid path in override/remove
- `ConfigTypeError`: Type mismatch in operation
- `ConfigOperationError`: Invalid operation (e.g., append to non-list)
- `ConfigValidationError`: Compatibility validation failed

**Example:**

```python
from spine.config import load_config

# Basic usage
config = load_config('config.yaml')

# With overrides
config = load_config('config.yaml', overrides=[
    'io.batch_size=16',
    'model.debug=true'
])

# With strict mode
config = load_config('config.yaml', strict='warn')
```

#### `extract_metadata(config, warn_missing=True)`

Extract metadata from configuration.

**Parameters:**
- `config` (dict): Configuration dictionary
- `warn_missing` (bool): Warn if `__meta__` block missing

**Returns:**
- `dict`: Metadata dictionary (empty if no metadata)

#### `get_nested_value(config, key_path)`

Get value at nested key path.

**Parameters:**
- `config` (dict): Configuration dictionary
- `key_path` (str): Dot-notation path (`"io.loader.batch_size"`)

**Returns:**
- Value at path

**Raises:**
- `KeyError`: Path not found

#### `set_nested_value(config, key_path, value)`

Set value at nested key path.

**Parameters:**
- `config` (dict): Configuration dictionary
- `key_path` (str): Dot-notation path
- `value`: Value to set (or `None` to delete)

**Returns:**
- `(dict, bool)`: Updated config and success flag

### Constants

```python
from spine.config import API_VERSION, META_KEY

API_VERSION  # Current config API version
META_KEY     # Metadata key name ("__meta__")
```

### Exceptions

```python
from spine.config import (
    ConfigError,              # Base exception
    ConfigIncludeError,       # Include errors
    ConfigCycleError,         # Circular includes
    ConfigPathError,          # Invalid paths
    ConfigTypeError,          # Type errors
    ConfigOperationError,     # Invalid operations
    ConfigValidationError,    # Compatibility errors
)
```

## Best Practices

1. **Always add metadata to production configs:**
   ```yaml
   __meta__:
     version: "YYMMDD"
     date: "YYYY-MM-DD"
     description: "Clear description"
   ```

2. **Use appropriate `kind`:**
   - `"bundle"` for complete configs
   - `"mod"` for modifiers

3. **Set `strict` based on `kind`:**
   - `"error"` for bundles (catch problems early)
   - `"warn"` for mods (allow flexibility)

4. **Use compatibility constraints:**
   ```yaml
   __meta__:
     compatible_with:
       base: ">=260107"
       io: ">=260107"
   ```

5. **Document with tags:**
   ```yaml
   __meta__:
     tags: [production, detector-name, purpose]
   ```

6. **Use `list_append: "unique"` with multiple modifiers:**
   ```yaml
   __meta__:
     list_append: "unique"
   ```

7. **Organize by component:**
   ```
   config/
     base/
       base_260107.yaml
     io/
       io_260107.yaml
     model/
       model_260107.yaml
     post/
       post_260107.yaml
     modifier/
       mod_data_260107.yaml
   ```

## Troubleshooting

### Missing Metadata Warning

```
UserWarning: Included file 'base.yaml' has no __meta__ block.
```

**Fix:** Add metadata to the file:

```yaml
__meta__:
  version: "260107"
  description: "What this config does"
```

### Compatibility Error

```
ConfigValidationError: io: parent version 240719 does not satisfy >=260107
```

**Fix:** Update parent component or adjust compatibility constraint

### Strict Mode Error

```
ConfigPathError: Cannot override 'model.new_feature': path does not exist (strict=error)
```

**Fix:** Either:
- Add the key to base config, or
- Set `strict: "warn"` in metadata, or
- Use `--set` with lenient mode

### Circular Include

```
ConfigCycleError: Circular include detected: a.yaml -> b.yaml -> a.yaml
```

**Fix:** Reorganize includes to break the cycle

## Migration Guide

### From `spine.utils.config` (v0.8.1)

The configuration system has been moved and enhanced:

**Old:**
```python
from spine.utils.config import load_config
```

**New:**
```python
from spine.config import load_config
```

**What's new in v0.9.0:**
- File composition with `include:`
- Parameter overrides with `override:`
- Key removal with `remove:` and `null`
- Metadata system with `__meta__`
- Version compatibility checking
- Typed exception hierarchy
- Command-line `--set` support

Existing configs without `include:`/`override:` blocks continue to work unchanged.
