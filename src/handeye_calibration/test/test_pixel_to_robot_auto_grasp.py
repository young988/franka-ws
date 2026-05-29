import sys
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = TEST_ROOT / "handeye_calibration"
sys.path.insert(0, str(PACKAGE_ROOT))

from grasp_logic import (
    classify_grasp_outcome,
    clamp_grasp_width,
    should_start_auto_grasp,
)


def test_classify_grasp_outcome_returns_success_when_width_stays_above_threshold():
    outcome = classify_grasp_outcome(
        action_succeeded=True,
        final_width=0.018,
        min_grasp_width=0.005,
    )
    assert outcome == "success"


def test_classify_grasp_outcome_returns_empty_when_width_closes_to_threshold():
    outcome = classify_grasp_outcome(
        action_succeeded=True,
        final_width=0.004,
        min_grasp_width=0.005,
    )
    assert outcome == "empty"


def test_classify_grasp_outcome_returns_failure_when_action_fails():
    outcome = classify_grasp_outcome(
        action_succeeded=False,
        final_width=0.020,
        min_grasp_width=0.005,
    )
    assert outcome == "failure"


def test_clamp_grasp_width_limits_requested_opening_to_nonnegative_value():
    assert clamp_grasp_width(-0.01) == 0.0
    assert clamp_grasp_width(0.04) == 0.04


def test_should_start_auto_grasp_requires_successful_execution_and_feature_enabled():
    assert should_start_auto_grasp(execution_error_code=0, enable_auto_grasp=True) is True
    assert should_start_auto_grasp(execution_error_code=1, enable_auto_grasp=True) is False
    assert should_start_auto_grasp(execution_error_code=0, enable_auto_grasp=False) is False


def test_grasp_logic_exports_expected_helper_names():
    assert callable(classify_grasp_outcome)
    assert callable(clamp_grasp_width)
    assert callable(should_start_auto_grasp)
