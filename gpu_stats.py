import json
import time
from jtop import jtop

OUTPUT_PATH = "gpu_stats.json"
INTERVAL = 1  # sample every 500ms

gpu_usages = []
gpu_temps = []
timestamps = []
gpu_mem_used = []
gpu_mem_total = []

print("Starting GPU monitor... Press Ctrl+C to stop and save.")

with jtop() as jetson:
    print("jtop connected.")
    try:
        while True:
            if jetson.ok():
                stats = jetson.stats
                gpu_usages.append(stats.get("GPU", None))
                gpu_temps.append(stats.get("Temp gpu", None))
                
                # Memory info
                mem = stats.get("RAM", None)
                if isinstance(mem, dict):
                    gpu_mem_used.append(mem.get("used", None))
                    gpu_mem_total.append(mem.get("total", None))

                elif isinstance(mem, (int, float)):
                    # This is percentage usage
                    gpu_mem_used.append(mem)


                timestamps.append(time.time())
            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        pass

valid_mem = [m for m in gpu_mem_used if m is not None]
results = {
    "gpu_usage_percent": sum(g for g in gpu_usages if g) / max(1, len([g for g in gpu_usages if g])),
    "gpu_temperature_c": sum(t for t in gpu_temps if t) / max(1, len([t for t in gpu_temps if t])),
    "max_gpu_temperature_c": max((t for t in gpu_temps if t), default=None),
    "system_memory_usage_percent": sum(valid_mem) / max(1, len(valid_mem)),
    "max_system_memory_usage_percent": max(valid_mem, default=None),
    "samples": len(timestamps)
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=4)

print(f"Saved GPU stats to {OUTPUT_PATH}")
print(results)