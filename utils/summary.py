def init_summary():
    return {
        "total": 0,
        "pedestrians": 0,
        "vehicles": 0
    }

def update_summary(summary, key):
    summary[key] += 1
