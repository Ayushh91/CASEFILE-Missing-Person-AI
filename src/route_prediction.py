from collections import defaultdict, Counter
import pandas as pd


def build_transition_matrix(data: pd.DataFrame) -> dict:
    """Build area-to-area transition probabilities."""

    transitions = defaultdict(Counter)

    for _, group in (
        data[data.area_id >= 0]
        .sort_values("timestamp")
        .groupby(["person_id", "trajectory_id"])
    ):
        states = group.area_id.astype(str).tolist()

        for a, b in zip(states, states[1:]):
            transitions[a][b] += 1

    matrix = {}

    for state, counts in transitions.items():
        total = sum(counts.values())

        matrix[state] = {
            target: count / total
            for target, count in counts.items()
        }

    return matrix


def predict_route(
    matrix: dict,
    start_area: int | str,
    length: int = 4,
) -> list[dict]:
    """
    Predict a probable route.

    Prefers a transition to another area when available.
    Falls back to the most probable transition when no
    alternative area exists.
    """

    current = str(start_area)
    route = []
    visited = {current}

    for _ in range(length):

        options = matrix.get(current, {})

        if not options:
            break

        # Prefer unvisited/different areas
        alternatives = {
            area: probability
            for area, probability in options.items()
            if area not in visited
        }

        if alternatives:
            next_area, probability = max(
                alternatives.items(),
                key=lambda pair: pair[1]
            )
        else:
            # If there are no new areas, use the most probable transition
            next_area, probability = max(
                options.items(),
                key=lambda pair: pair[1]
            )

        route.append(
            {
                "from_area": current,
                "to_area": next_area,
                "probability": probability,
            }
        )

        current = next_area
        visited.add(current)

    return route
