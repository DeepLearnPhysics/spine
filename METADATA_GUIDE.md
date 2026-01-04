# Configuration Metadata Guide

## Overview

The SPINE configuration system supports a `__meta__` block that provides metadata about configuration files, enabling versioning, compatibility checking, and behavioral control. This guide explains all available metadata fields.

## Basic Structure

```yaml
__meta__:
  # Identity and versioning
  version: "240719"
  date: "2024-07-19"
  description: "Base configuration for ICARUS detector"
  tags: [icarus, production, v4]
  
  # Configuration type and behavior (system fields)
  kind: "bundle"              # or "mod"
  strict: "error"             # or "warn"
  list_append: "append"       # or "unique"
  
  # Compatibility declarations
  compatible_with:
    base: "==240719"
    io: ">=240812"
  extends: "data"             # For modifiers
  
  # Modifier-specific fields
  name: "modifier_name"
  priority: 10                # Lower = applied first
  applies_to: [mc, data]
  requires: []
  conflicts_with: []
  
  # Model-specific fields
  weights:
    path: /path/to/model.ckpt
  
  # Top-level config fields
  components:
    base: "240719"
    io: "240812"

# Your actual configuration
base:
  world_size: 1
  iterations: 1000
```

## Metadata Fields

### Core Identity Fields

#### `version`
- **Type**: string or number
- **Purpose**: Identifies the config version
- **Format**: Typically `"YYMMDD"` (e.g., `"240719"`) or semantic versioning (e.g., `"1.0.0"`)
- **Used by**: Component configs (base, io, model, post) and modifiers
- **Example**: 
  ```yaml
  __meta__:
    version: "240719"
  ```

#### `date`
- **Type**: string
- **Purpose**: Documents when the config was created/updated
- **Format**: ISO date format recommended (e.g., `"2024-07-19"`)
- **Example**:
  ```yaml
  __meta__:
    date: "2024-07-19"
  ```

#### `name`
- **Type**: string
- **Purpose**: Identifies the config or modifier by name
- **Used by**: Modifiers and top-level configs
- **Example**:
  ```yaml
  __meta__:
    name: "numi"  # For modifiers
    # or
    name: "icarus_full_chain_240719"  # For top-level configs
  ```

#### `description`
- **Type**: string
- **Purpose**: Human-readable description of the config's purpose
- **Example**:
  ```yaml
  __meta__:
    description: "Production configuration for ICARUS Run 3"
  ```

#### `tags`
- **Type**: list of strings
- **Purpose**: Categorization and searching
- **Common tags**: `production`, `latest`, `stable`, `average-charge`, `collection-only`, `calibrated`
- **Example**:
  ```yaml
  __meta__:
    tags: [icarus, production, run3, cosmic]
  ```

### Model-Specific Fields

#### `weights`
- **Type**: dict
- **Purpose**: Documents paths to trained model weights
- **Used by**: Model configuration files
- **Example**:
  ```yaml
  __meta__:
    version: "240719"
    weights:
      path: /sdf/data/neutrino/icarus/spine/train/mpvmpr_v02/weights/full_chain/default/snapshot-7999.ckpt
  ```

### Top-Level Config Fields

#### `components`
- **Type**: dict (component_name: version)
- **Purpose**: Documents which component versions are used in this complete configuration
- **Used by**: Top-level configuration bundles (optional)
- **Note**: The loader automatically accumulates component versions from included files, so declaring components upfront is optional. However, explicitly listing components provides clear documentation of the configuration's composition.
- **Example**:
  ```yaml
  __meta__:
    name: "icarus_full_chain_250115"
    version: "250115"
    components:
      base: "240719"
      io: "240812"
      model: "250115"
      post: "250115"
  ```
- **Auto-accumulation**: When a config includes `base/base_240719.yaml` with `version: "240719"`, the loader automatically infers `components: {base: "240719"}` without needing explicit declaration.

### System Behavior Control

These fields control internal loader behavior. Most configs don't need to set these explicitly as they have sensible defaults based on `kind`.

#### `kind`
The `kind` field determines the **fundamental nature** of the configuration file.

- **Type**: string
- **Valid values**: `"bundle"` or `"mod"`
- **Default**: `"bundle"`
- **Note**: Usually inferred from file location/purpose; rarely set explicitly in production

#### `kind: "bundle"` (Base Configuration)
A **bundle** is a **complete, standalone configuration** that can be used independently. It represents a full configuration that doesn't rely on being included into something else.

**Characteristics**:
- Self-contained and complete
- Can be run directly: `spine --config my_bundle.yaml`
- Default strict mode: `"error"` (fails on missing paths)
- Typically includes full base configuration

**Use cases**:
- Main experiment configurations (e.g., `icarus_production.yaml`)
- Complete training configurations
- Detector run configurations

**Example**:
```yaml
# icarus_production.yaml
__meta__:
  kind: "bundle"
  version: "240719"
  description: "Complete ICARUS production configuration"
  strict: "error"

base:
  world_size: 1
  iterations: 1000
  
geo:
  detector: icarus
  tag: icarus_v4

io:
  loader:
    batch_size: 8

model:
  name: full_chain
```

#### `kind: "mod"` (Modifier Configuration)
A **mod** (modifier) is a **partial configuration** designed to be included into a bundle. It modifies or extends an existing configuration.

**Characteristics**:
- Incomplete by itself
- Meant to be included: `include: my_mod.yaml`
- Default strict mode: `"warn"` (warns on missing paths, doesn't fail)
- Contains overrides, extensions, or specialized settings

**Use cases**:
- Detector-specific modifications
- Runtime environment adjustments  
- Feature toggles or experimental settings
- Performance tuning overlays

**Example**:
```yaml
# gpu_optimizations.yaml
__meta__:
  kind: "mod"
  extends: performance
  description: "GPU memory optimizations"
  strict: "warn"

override:
  io.loader.batch_size: 64
  io.loader.num_workers: 16
  model.use_mixed_precision: true
```

### Modifier-Specific Fields

These fields are used in modifier configurations (`modifier/`) to control how modifiers are applied and composed.

#### `priority`
- **Type**: integer
- **Purpose**: Controls the order in which modifiers are applied
- **Behavior**: Lower numbers = applied first
- **Range**: Typically 10-60
- **Example**:
  ```yaml
  __meta__:
    name: "data"
    priority: 10  # Applied first
  ```

**Standard priority levels** (ICARUS convention):
- **10**: `data` - Transform to data-only mode
- **20**: `4ms`, `8ms` - Lifetime calibration (mutually exclusive)
- **25**: `transp` - Transparency correction
- **30**: `single` - Single-cryostat processing
- **40**: `numi` - NuMI-specific settings
- **50**: `lite` - Lite output mode
- **60**: `unblind` - Unblinded data only

#### `applies_to`
- **Type**: list of strings
- **Purpose**: Declares what data types this modifier works with
- **Valid values**: `"mc"` (Monte Carlo), `"data"` (real detector data)
- **Example**:
  ```yaml
  __meta__:
    name: "numi"
    applies_to: [mc, data]  # Works with both
  ```
  ```yaml
  __meta__:
    name: "data"
    applies_to: [data]  # Data only
  ```

#### `requires`
- **Type**: list of strings
- **Purpose**: Declares other modifiers that must be applied first
- **Behavior**: System should validate dependencies (implementation-dependent)
- **Example**:
  ```yaml
  __meta__:
    name: "unblind"
    requires: [data]  # Must apply 'data' modifier first
  ```

#### `conflicts_with`
- **Type**: list of strings
- **Purpose**: Declares modifiers that cannot be used together
- **Behavior**: System should validate conflicts (implementation-dependent)
- **Example**:
  ```yaml
  __meta__:
    name: "4ms"
    conflicts_with: [8ms]  # Mutually exclusive calibrations
  ```

#### `strict`
Controls how the loader handles operations on non-existent paths (e.g., override attempts on missing keys).

- **Type**: string
- **Valid values**: `"warn"` or `"error"`
- **Default**: 
  - `"error"` for `kind: "bundle"` (fail fast on missing paths)
  - `"warn"` for `kind: "mod"` (allow optional modifications)

**`strict: "error"`** (Strict mode):
- Raises exception if override path doesn't exist
- Best for bundles where all paths should exist
- Catches typos and configuration errors early

**`strict: "warn"`** (Lenient mode):
- Issues warning if override path doesn't exist
- Best for mods that may apply to multiple base configs
- Allows optional modifications that may not apply everywhere

**Example**:
```yaml
# production_bundle.yaml
__meta__:
  kind: "bundle"
  strict: "error"  # Fail on any missing paths

# optional_features.yaml
__meta__:
  kind: "mod"
  strict: "warn"   # Warn but don't fail if features don't exist
```

#### `list_append`
Controls behavior when appending to lists using the `+` operator.

- **Type**: string
- **Valid values**: `"append"` or `"unique"`
- **Default**: `"append"`

**`list_append: "append"`** (Allow duplicates):
- Appends items even if they already exist in the list
- Preserves insertion order and duplicates
- Faster (no duplicate checking)

**`list_append: "unique"`** (No duplicates):
- Only appends items that don't already exist
- Useful when multiple mods might add the same items
- Prevents accidental duplication

**Example**:
```yaml
__meta__:
  list_append: "unique"  # Prevent duplicate parsers

override:
  parsers+: [meta, run_info]  # Won't add duplicates if already present
```

### Compatibility System

#### `compatible_with`
Declares version requirements for component dependencies. Used to ensure configs work together.

- **Type**: dict (component: version_constraint)
- **Purpose**: Prevents incompatible config combinations
- **Behavior**: Raises `ConfigValidationError` if parent doesn't match requirements
- **Version operators**: `==` (exact), `>=` (minimum), `<=` (maximum)

**Dict format with version constraints** (PRODUCTION SCHEMA):
```yaml
__meta__:
  compatible_with:
    base: "==240719"      # Exact version required
    io: ">=240812"        # Minimum version
    post: ">=250625"      # Minimum version
```

**String format** (simple compatibility):
```yaml
__meta__:
  compatible_with: "icarus_base"
```

**List format** (multiple compatible parents):
```yaml
__meta__:
  compatible_with:
    - icarus_base
    - icarus_base_v3
    - icarus_production
```

**How it works**:
When file A includes file B:
1. System checks if B has `compatible_with` field
2. If dict format: validates version requirements for each component
3. If string/list format: compares B's requirements against A's `kind` and `extends` fields
4. Raises `ConfigValidationError` if requirements not met

#### `extends`
Declares what category this modifier extends (for modifier configs).

- **Type**: string
- **Purpose**: Documents modification category, used in compatibility checking
- **Used by**: Modifier configs
- **Common values**: `"data"`, `"unblind"`, `"numi"`, `"single"`, etc.

**Example**:
```yaml
# mod_data_240719.yaml
__meta__:
  version: "240719"
  extends: "data"
  date: "2024-07-19"
```

## Complete Examples

### Example 1: Model Configuration (Component)
```yaml
# model_240719.yaml
__meta__:
  version: "240719"
  date: "2024-07-19"
  description: "ICARUS model with average charge rescaling"
  compatible_with:
    base: "==240719"
    io: ">=240719"
    post: ">=240719"
  tags: [average-charge]
  weights:
    path: /sdf/data/neutrino/icarus/spine/train/mpvmpr_v02/weights/full_chain/default/snapshot-7999.ckpt

include: model_common.yaml

override:
  model.weight_path: /sdf/data/neutrino/icarus/spine/train/mpvmpr_v02/weights/full_chain/default/snapshot-7999.ckpt
  model.modules.chain.charge_rescaling: average
```

### Example 2: Simple Modifier
```yaml
# mod_unblind_240719.yaml
__meta__:
  version: "240719"
  extends: "unblind"
  date: "2024-07-19"

include: mod_unblind_common.yaml
```

### Example 3: Modifier with Metadata
```yaml
# mod_numi_240719.yaml
__meta__:
  name: "numi"
  version: "240719"
  description: "NuMI-specific settings (wider flash matching window)"
  priority: 40
  applies_to: [mc, data]
  requires: []
  conflicts_with: []
  date: "2024-07-19"

override:
  post.flash_match.cfg: flashmatch/flashmatch_numi_230930.cfg
```

### Example 4: Modifier with Dependencies
```yaml
# mod_4ms_240719.yaml
__meta__:
  name: "4ms"
  version: "240719"
  description: "4ms lifetime calibration for data"
  priority: 20
  applies_to: [data]
  requires: []
  conflicts_with: [8ms]  # Mutually exclusive with 8ms calibration
  date: "2024-07-19"

override:
  post.apply_calibrations.lifetime.lifetime: 4.0  # ms
```

### Example 5: Data Modifier
```yaml
# mod_data_240719.yaml  
__meta__:
  version: "240719"
  extends: "data"
  date: "2024-07-19"

include: mod_data_common.yaml

override:
  # Set calibration parameters for data (post-processor)
  post.apply_calibrations:
    gain:
      gain: [76.02, 75.29, 77.36, 77.1]  # e-/ADC for collection planes
    recombination:
      model: mbox
      efield: 0.4938  # kV/cm
    lifetime:
      lifetime_db: /sdf/data/neutrino/icarus/db/v09_84_01/tpc_elifetime_data.db
    transparency:
      transparency_db: /sdf/data/neutrino/icarus/db/v09_84_01/tpc_yz_correction_data.db
  
  post.flash_match.scaling: 1/1400
```

### Example 6: Top-Level Bundle
```yaml
# icarus_full_chain_250115.yaml
__meta__:
  name: "icarus_full_chain_250115"
  version: "250115"
  date: "2025-01-15"
  description: "ICARUS full chain with v3 dataset"
  components:
    base: "240719"
    io: "240812"
    model: "250115"
    post: "250115"
  tags: [production, average-charge, stable]

include:
  - base/base_240719.yaml
  - io/io_240812.yaml
  - model/model_250115.yaml
  - post/post_250115.yaml
```

## Compatibility Checking Flow

The loader performs **deferred compatibility checking** to support forward references between components.

### Loading Process

1. **Recursive loading**: The loader processes includes recursively, accumulating all components.

2. **Component accumulation**: As each file is included, its component version is registered:
   ```yaml
   # base/base_240719.yaml
   __meta__:
     version: "240719"
   # → Registers component: base = "240719"
   ```

3. **Deferred validation**: All compatibility requirements are collected but not checked until all includes are loaded.

4. **Final validation**: After all files are included and all components accumulated, the loader validates all compatibility constraints at once.

### Why Deferred Checking?

This allows configurations with forward references:

```yaml
# icarus_full_chain.yaml
include:
  - base/base_240719.yaml
  - io/io_240812.yaml
  - model/model_250625.yaml    # requires "post >= 250625"
  - post/post_250625.yaml       # loaded AFTER model

# Without deferred checking:
# ✗ model validation would fail (post not yet loaded)

# With deferred checking:
# ✓ All components accumulated first, then validated together
```

### Validation Rules

When checking `compatible_with: {component: ">=version"}`:

1. **Version format**: Must be exactly 6 digits (YYMMDD)
   - Valid: `"240719"`, `"250625"`
   - Invalid: `"1.0.0"`, `"240719.1"`, `"24.07.19"`

2. **Operators**: `==`, `>=`, `<=`, `>`, `<`, `!=`
   - Example: `{base: ">=240719", io: "==240812"}`

3. **Missing components**: If required component not found, validation fails
   - Error: `"base: parent has no version (required >=240719)"`

4. **Version comparison**: Integer comparison of 6-digit dates
   - `240719 >= 240719` → True
   - `250625 >= 240719` → True  
   - `240101 >= 240719` → False

## Best Practices

1. **Always add metadata to production configs**:
   ```yaml
   __meta__:
     version: "YYMMDD"
     date: "YYYY-MM-DD"
     description: "Clear description"
   ```

2. **Use `kind` appropriately**:
   - `"bundle"` for complete, runnable configs
   - `"mod"` for partial configs meant to be included

3. **Set `strict` based on `kind`**:
   - `"error"` for bundles (catch problems early)
   - `"warn"` for mods (allow flexibility)

4. **Use `compatible_with` for safety**:
   ```yaml
   __meta__:
     kind: "mod"
     compatible_with: [my_base_config]
   ```

5. **Document with tags**:
   ```yaml
   __meta__:
     tags: [detector-name, run-period, purpose]
   ```

6. **Use `list_append: "unique"` when combining multiple mods**:
   ```yaml
   __meta__:
     list_append: "unique"  # Prevent duplicate entries
   ```

## Warnings

If you see warnings like:
```
UserWarning: Included file 'base.yaml' has no __meta__ block. 
Consider adding metadata for better configuration management.
```

This means an included file is missing metadata. While not required, adding metadata enables:
- Version tracking
- Compatibility checking
- Better error messages
- Documentation

To fix, add a `__meta__` block to the file:
```yaml
__meta__:
  version: "240719"
  description: "What this config does"
  kind: "bundle"  # or "mod"
```
