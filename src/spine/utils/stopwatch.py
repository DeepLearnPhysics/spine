import time
from dataclasses import dataclass


@dataclass
class Time:
    """Simple dataclass to hold time information.

    Attributes
    ----------
    wall : float, optional
         Wall time
    cpu : float, optional
         CPU time
    """

    wall: float = None
    cpu: float = None

    def __add__(self, time):
        """Overload the addition operator.

        Parameters
        ----------
        time : Time
            Other Time object

        Returns
        -------
        Time
           Summed times
        """
        return Time(wall=self.wall + time.wall, cpu=self.cpu + time.cpu)

    def __sub__(self, time):
        """Overload the substraction operator.

        Parameters
        ----------
        time : Time
            Other Time object

        Returns
        -------
        Time
           Substracted times
        """
        return Time(wall=self.wall - time.wall, cpu=self.cpu - time.cpu)

    def __eq__(self, time):
        """Overload the equality operator.

        Parameters
        ----------
        time : Time
            Other Time object

        Returns
        -------
        bool
            True if both times are identical
        """
        if isinstance(time, type(self)):
            return self.wall == time.wall and self.cpu == time.cpu
        else:
            return self.wall == time and self.cpu == time

    def copy(self):
        """Returns an independant copy of the object.

        Returns
        -------
        Time
            Copy of the object
        """
        return Time(wall=self.wall, cpu=self.cpu)

    @classmethod
    def current(cls):
        """Simple function which returns the current time (wall and cpu).

        Returns
        -------
        Time
           Current time
        """
        return cls(time.time(), time.process_time())


class Stopwatch:
    """Simple class to hold timing information for a specific process."""

    def __init__(self):
        """Give default values to the underlying class attributes."""
        self._start = Time()
        self._stop = Time()
        self._pause = Time()
        self._time = Time(0.0, 0.0)
        self._total = Time(0.0, 0.0)

    @property
    def running(self):
        """Whether the stopwatch is currently running."""
        return self._start != Time() and self._stop == Time()

    @property
    def paused(self):
        """Whether the stopwatch is currently paused."""
        return self._pause != Time() and self._stop == Time()

    @property
    def start(self):
        """Time when the stopwatch was last started."""
        return self._start

    @start.setter
    def start(self, start):
        # Check that the watch was not already started
        if self.running:
            raise ValueError("Cannot restart a watch that has not been stopped.")

        # Start watch, reinitialize stop
        self._start = start
        self._stop = Time()
        if self._pause == Time():
            self._time = Time(0.0, 0.0)

    @property
    def stop(self):
        """Time when the stopwatch was last stopped."""
        return self._stop

    @stop.setter
    def stop(self, stop):
        # Check that the watch was started
        if self._start == Time():
            raise ValueError("Cannot stop a watch that has not been started.")

        # Check that the watch was not already stopped
        if self._stop != Time():
            raise ValueError("Cannot stop a watch more than once.")

        # Stop the watch, record the relevant quantities
        self._stop = stop
        self._pause = Time()
        self._time += self.stop - self.start
        self._total += self.time

    @property
    def pause(self):
        """Time when the stopwatch was last paused."""
        return self._pause

    @pause.setter
    def pause(self, pause):
        # Check that the watch was started
        if self._start == Time():
            raise ValueError("Cannot pause a watch that has not been started.")

        # Check that the watch was not already stopped
        if self._stop != Time():
            raise ValueError("Cannot pause a watch that has been stopped.")

        # Increment the time, reset the start
        self._pause = pause
        self._time += self.pause - self.start
        self._start = Time()

    @property
    def time(self):
        """Time between the last start and the last stop."""
        # Check that the watch was stopped
        if self._stop == Time():
            raise ValueError("Cannot get time of watch that has not been stopped.")

        return self._time

    @property
    def time_sum(self):
        """Sum of times between all watch starts en stops."""
        # Check that the watch was stopped
        if self._stop == Time():
            raise ValueError("Cannot get time of watch that has not been stopped.")

        return self._total


class StopwatchManager:
    """Simple class to organize various time measurements."""

    def __init__(self):
        """Initalize the basic private stopwatch attributes."""
        self._watch = {}

    def keys(self):
        """Get the list of all initialized stopwatch tags.

        Returns
        -------
        List[str]
            List of stopwatch names
        """
        return self._watch.keys()

    def values(self):
        """Get the list of all initialized stopwatches.

        Returns
        -------
        List[Stopwatch]
            List of stopwatch objects
        """
        return self._watch.values()

    def items(self):
        """Get the list of all initialized stopwatch tags and the
        corresponding Stopwatch object for each of them.

        Returns
        -------
        List[Tuple[str, Stopwatch]]
            List of (key, stopwatch) pairs
        """
        return self._watch.items()

    def initialize(self, key):
        """Initialize one stopwatch. If it's already been initialized,
        reset the global counters to 0.

        Parameters
        ----------
        key : Union[str, List[str]]
            Key or list of keys to initialize a `Stopwatch` for
        """
        # Loop over keys
        keys = [key] if isinstance(key, str) else key
        for k in keys:
            # Initialize stopwatch
            self._watch[k] = Stopwatch()

    def reset(self, key=None):
        """Reset a stopwatch to its initial state.

        Parameters
        ----------
        key : Union[str, List[str]], optional
            Key or list of keys to reset a `Stopwatch` for. If None, reset all stopwatches.
        """
        # Get the list of keys to reset. If not specified, reset all
        key = self.keys() if key is None else key

        # Loop over keys
        keys = [key] if isinstance(key, str) else key
        for k in keys:
            # Check that a stopwatch exists
            if not k in self._watch:
                raise KeyError(f"No stopwatch initialized under the name: {k}")

            # Reset stopwatch
            self._watch[k] = Stopwatch()

    def start(self, key):
        """Starts a stopwatch for a unique key.

        Parameters
        ----------
        key : Union[str, List[str]]
            Key or list of keys for which to start the clock
        """
        # Loop over keys
        start_time = Time.current()
        keys = [key] if isinstance(key, str) else key
        for k in keys:
            # If this is the first time, initialize a new Stopwatch
            if not k in self._watch:
                raise KeyError(f"No stopwatch initialized under the name: {k}")

            # Reinitialize the watch
            self._watch[k].start = start_time.copy()

    def stop(self, key):
        """Stops a stopwatch for a unique key.

        Parameters
        ----------
        key : str
            Key for which to stop the clock
        """
        # Loop over keys
        stop_time = Time.current()
        keys = [key] if isinstance(key, str) else key
        for k in keys:
            # Check that a stopwatch exists
            if not k in self._watch:
                raise KeyError(f"No stopwatch started under the name: {k}")

            # Stop
            self._watch[k].stop = stop_time.copy()

    def pause(self, key):
        """Temporarily pause a watch for a unique key.

        Parameters
        ----------
        key : str
            Key for which to pause the clock
        """
        # Loop over keys
        pause_time = Time.current()
        keys = [key] if isinstance(key, str) else key
        for k in keys:
            # Check that a stopwatch exists
            if not k in self._watch:
                raise KeyError(f"No stopwatch started under the name: {k}")

            # Stop
            self._watch[k].pause = pause_time.copy()

    def time(self, key):
        """Returns the time recorded since the last start.

        Parameters
        ----------
        key : str
            Key for which to return the time

        Returns
        -------
        Time
            Execution time of one iteration of a process
        """
        # Check that a stopwatch exists
        if not key in self._watch:
            raise KeyError(f"No stopwatch started under the name: {key}")

        # Return the time since the start
        return self._watch[key].time

    def time_sum(self, key):
        """Returns the sum of times recorded between each start/stop pairs.

        Parameters
        ----------
        key : str
            Key for which to return the time

        Returns
        -------
        Time
            Execution time of all iterations of a process so far
        """
        # Check that a stopwatch exists
        if not key in self._watch:
            raise KeyError(f"No stopwatch started under the name: {key}")

        # Return the time since the start
        return self._watch[key].time_sum

    def times(self):
        """Returns the times for each of the stopwatches as a dictionary.

        Returns
        -------
        Dict[str, Time]
            Execution time of one iteration of each process
        """
        return {key: value.time for key, value in self.items()}

    def times_sum(self):
        """Returns the times for each of the stopwatches as a dictionary.

        Returns
        -------
        Dict[str, Time]
            Execution time of all iterations of each process so far
        """
        return {key: value.time_sum for key, value in self.items()}

    def update(self, other, prefix=None):
        """Updates this manager with values from another stopwatch manager.

        Parameters
        ----------
        other : StopwatchManager
             Dictionary of execution times from another process
        prefix : str, optional
             String to prefix the timer key with

        Returns
        -------
        Dict[str, Time]
            Combined execution time of all iterations of each process so far
        """
        for key, value in other.items():
            if prefix is None:
                self._watch[key] = value
            else:
                self._watch[f"{prefix}_{key}"] = value
