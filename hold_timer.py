# hold_timer.py
# The temporal state machine (the "bridge" between the visual loop and the audio loop).
#
#   WAITING  -> user is not in the pose yet
#   HOLDING  -> user is in the pose, stopwatch running
#   DONE     -> held for the full time without a single break (fires exactly once)
#
# Falling out of the pose at ANY point sends it back to WAITING with the stopwatch at zero.
# No partial credit.


class HoldTimer:
    def __init__(self, required_seconds):
        self.required_seconds = required_seconds
        self.state = "WAITING"
        self.start_time = None
        self.checked_in = False      # has this pose's one check-in already happened?

    def reset(self, required_seconds):
        """Start fresh for a new pose (or for an extended hold)."""
        self.required_seconds = required_seconds
        self.state = "WAITING"
        self.start_time = None

    def update(self, in_pose, now):
        """Call once per frame. Returns True ONLY on the single frame the hold completes."""
        if self.state == "DONE":
            return False

        if not in_pose:
            self.state = "WAITING"          # broke the pose -> back to zero
            self.start_time = None
            return False

        if self.state == "WAITING":
            self.state = "HOLDING"          # just got into the pose -> start the stopwatch
            self.start_time = now

        if now - self.start_time >= self.required_seconds:
            self.state = "DONE"
            return True

        return False

    def seconds_held(self, now):
        if self.state == "HOLDING":
            return now - self.start_time
        if self.state == "DONE":
            return self.required_seconds
        return 0