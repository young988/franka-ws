"""Grasp outcome classification and auto-grasp gating utilities."""


def classify_grasp_outcome(action_succeeded, final_width, min_grasp_width):
    """Classify a grasp action outcome.

    Returns:
        'failure': action was not successful
        'empty':   action succeeded but gripped nothing (width below threshold)
        'success': action succeeded and contacted an object
    """
    if not action_succeeded:
        return 'failure'
    if float(final_width) <= float(min_grasp_width):
        return 'empty'
    return 'success'


def clamp_grasp_width(width):
    """Ensure the commanded grasp width is non-negative."""
    return max(float(width), 0)


def should_start_auto_grasp(execution_error_code, enable_auto_grasp):
    """Return True if the motion plan execution succeeded and auto-grasp is enabled."""
    return execution_error_code == 0 and bool(enable_auto_grasp)
