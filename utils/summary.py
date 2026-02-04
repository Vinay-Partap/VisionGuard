# utils/summary.py

def init_summary():
    """
    Initialize detection summary
    """
    return {
        "total": 0,
        "pedestrians": 0,
        "vehicles": 0
    }


def reset_summary(summary):
    """
    Reset summary counters
    """
    summary["total"] = 0
    summary["pedestrians"] = 0
    summary["vehicles"] = 0
    return summary
