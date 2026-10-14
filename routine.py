# routine.py
# The deterministic routine state machine.
# Holds the queue of poses and applies a FIXED sentiment -> action table.
# Every decision is a direct lookup of a pointer stored in the current pose's JSON profile
# (easier_variant / harder_variant / rest_pose) - no searching, no re-planning: O(1).
#
#   pain       -> swap the next pose for the rest pose
#   negative   -> swap the next pose for the easier variant (rest pose if there is none)
#   neutral    -> carry on with the plan
#   no_answer  -> carry on with the plan
#   positive   -> swap the next pose for the harder variant; if there is none, extend the hold


class Routine:
    def __init__(self, pose_names, profiles):
        self.queue = list(pose_names)       # a copy, so config.ROUTINE itself never changes
        self.index = 0
        self.profiles = profiles

    def current_pose(self):
        if self.index < len(self.queue):
            return self.queue[self.index]
        return None

    def next_pose(self):
        if self.index + 1 < len(self.queue):
            return self.queue[self.index + 1]
        return None

    def is_finished(self):
        return self.index >= len(self.queue)

    def advance(self):
        self.index = self.index + 1

    def _swap_next(self, new_pose):
        if new_pose is None:
            return
        next_index = self.index + 1
        if next_index < len(self.queue):
            self.queue[next_index] = new_pose
        else:
            self.queue.append(new_pose)     # we were on the last pose -> add it at the end

    def apply_sentiment(self, label):
        """Runs the lookup table. Returns the action taken: REST / EASIER / HARDER / EXTEND / CONTINUE."""
        profile = self.profiles[self.current_pose()]

        if label == "pain":
            self._swap_next(profile["rest_pose"])
            return "REST"

        if label == "negative":
            if profile["easier_variant"] is not None:
                self._swap_next(profile["easier_variant"])
                return "EASIER"
            self._swap_next(profile["rest_pose"])
            return "REST"

        if label == "positive":
            if profile["harder_variant"] is not None:
                self._swap_next(profile["harder_variant"])
                return "HARDER"
            return "EXTEND"

        return "CONTINUE"