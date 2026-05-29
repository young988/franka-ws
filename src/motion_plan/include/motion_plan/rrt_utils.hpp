#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

namespace motion_plan
{
namespace detail
{

inline double adaptiveStepSizeFromClearance(double clearance, double min_step_size, double max_step_size,
                                            double near_clearance = 0.05, double far_clearance = 0.20)
{
  if (max_step_size < min_step_size)
  {
    std::swap(max_step_size, min_step_size);
  }

  if (!std::isfinite(clearance))
  {
    return max_step_size;
  }

  if (clearance <= near_clearance)
  {
    return min_step_size;
  }

  if (clearance >= far_clearance)
  {
    return max_step_size;
  }

  const double ratio = (clearance - near_clearance) / (far_clearance - near_clearance);
  return min_step_size + ratio * (max_step_size - min_step_size);
}

template <typename URNG>
inline std::pair<std::size_t, std::size_t> sampleShortcutPair(std::size_t path_size, URNG& rng)
{
  if (path_size < 3)
  {
    return { 0U, 0U };
  }

  std::uniform_int_distribution<std::size_t> first_distribution(0U, path_size - 3U);
  const std::size_t first = first_distribution(rng);

  std::uniform_int_distribution<std::size_t> second_distribution(first + 2U, path_size - 1U);
  const std::size_t second = second_distribution(rng);
  return { first, second };
}

template <typename Waypoint, typename SegmentValidator, typename PairSelector>
std::vector<Waypoint> shortcutPath(const std::vector<Waypoint>& path, SegmentValidator&& segment_valid,
                                   int smoothing_iterations, PairSelector&& select_pair)
{
  if (path.size() < 3U || smoothing_iterations <= 0)
  {
    return path;
  }

  std::vector<Waypoint> smoothed(path.begin(), path.end());

  for (int iteration = 0; iteration < smoothing_iterations && smoothed.size() >= 3U; ++iteration)
  {
    const std::pair<std::size_t, std::size_t> indices = select_pair(smoothed.size());
    const std::size_t first = indices.first;
    const std::size_t second = indices.second;

    if (first >= second || second >= smoothed.size() || second < first + 2U)
    {
      continue;
    }

    if (!segment_valid(smoothed[first], smoothed[second]))
    {
      continue;
    }

    std::vector<Waypoint> shortened;
    shortened.reserve(first + 1U + (smoothed.size() - second));
    shortened.insert(shortened.end(), smoothed.begin(), smoothed.begin() + static_cast<std::ptrdiff_t>(first) + 1);
    shortened.insert(shortened.end(), smoothed.begin() + static_cast<std::ptrdiff_t>(second), smoothed.end());
    smoothed.swap(shortened);
  }

  return smoothed;
}

}  // namespace detail
}  // namespace motion_plan
