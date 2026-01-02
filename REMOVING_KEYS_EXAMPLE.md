# Examples: Removing Keys from Imported Config Files

## Challenge Solved! ✓

Yes, there are now **two ways** to remove a key entirely from an imported file in the custom config.py parsing scheme:

## Method 1: Using `null` in `override:`

```yaml
# base_config.yaml
base:
  iterations: 1000
  seed: 42
  debug_mode: true

io:
  loader:
    batch_size: 4
    shuffle: true
    num_workers: 8
```

```yaml
# my_config.yaml
include: base_config.yaml

override:
  io.loader.shuffle: null        # Remove shuffle key entirely
  base.debug_mode: null          # Remove debug_mode key entirely
  io.loader.batch_size: 16       # Still can override other keys normally
```

Result: The final config will **not** contain `io.loader.shuffle` or `base.debug_mode` at all.

## Method 2: Using the `remove:` directive

```yaml
# my_config.yaml
include: base_config.yaml

# Remove specific keys using dot notation
remove: io.loader.shuffle

# Or remove multiple keys
remove:
  - io.loader.shuffle
  - base.debug_mode
  - io.loader.num_workers
```

## Method 3: Combine both!

```yaml
include: base_config.yaml

# Use remove for explicit deletions
remove:
  - io.loader.shuffle
  - base.debug_mode

# Use override for both deletions and modifications
override:
  io.loader.num_workers: null    # Delete this one
  io.loader.batch_size: 16       # Override this one
```

## Key Points

1. **Both methods delete the key completely** - it won't exist in the final config dict
2. **Deletions happen after includes but before regular config merge**
3. **Order of operations**:
   - Load included files (with their recursive includes)
   - Merge main config
   - Apply `remove:` directive deletions
   - Apply `override:` (including null deletions and regular override values)
4. **Safe for non-existent keys** - no error if you try to delete a key that doesn't exist
5. **Works with nested paths** - use dot notation like `model.modules.uresnet.dropout`

## Use Cases

- Remove unwanted defaults from base configs
- Clean up development settings for production
- Disable optional features
- Simplify configs by removing unused parameters
