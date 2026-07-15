# `log_writer.h` — Logging System

**Path:** `include/oil/log_writer.h`

Logging and metrics tracking system for training and inference.

## LogWriter Class

```cpp
class LogWriter {
    LogWriter(const std::string& log_dir, const std::string& experiment_name);
    
    void log_scalar(const std::string& name, float value, int step);
    void log_dict(const std::unordered_map<std::string, float>& dict, int step);
    void log_text(const std::string& tag, const std::string& text);
    void save_config(const std::string& json_config);
    void flush();
};
```

### Features

- Training loss/metrics logging
- JSON-formatted log files
- TensorBoard-compatible scalars
- Experiment organization
- Auto-flush at intervals

### Log File Structure

```
logs/
├── experiment_name/
│   ├── config.json
│   ├── train.log
│   ├── events.jsonl
│   └── checkpoints/
│       ├── step_1000.oil
│       ├── step_2000.oil
│       └── best.oil
```
