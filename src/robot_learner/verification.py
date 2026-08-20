"""Simple independent verifiers suitable for initial integration tests."""

from robot_learner.models import Checkpoint, ExecutionTrace, Outcome, VerificationResult


class ContextPredicateVerifier:
    """Checks a named boolean in the most recent observation context."""

    def evaluate(self, checkpoint: Checkpoint, trace: ExecutionTrace) -> VerificationResult:
        observation = trace.observations[-1] if trace.observations else trace.start_observation
        value = observation.context.get(checkpoint.success_predicate.name)
        if value is True:
            return VerificationResult(Outcome.SUCCESS, "predicate observed", 1.0)
        if value is False:
            return VerificationResult(Outcome.FAILURE, "predicate not satisfied", 1.0)
        return VerificationResult(Outcome.UNCERTAIN, "predicate was not observable", 0.0)

