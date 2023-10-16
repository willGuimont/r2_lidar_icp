from abc import abstractmethod, ABC


class TransformationChecker(ABC):
    def __init__(self):
        """
        Base class for transformation checkers.
        Transformation checkers are used to check if the ICP algorithm should stop.
        """
        self.error = None

    def begin(self):
        """
        Called before the ICP algorithm starts.
        Should be used to initialize the checker.
        :return: None
        """

    def is_finished(self, error: float) -> bool:
        """
        Called after each iteration of the ICP algorithm.
        Should be used to update the checker.
        :param: error: Error between the point cloud and the reference
        :return: True if the ICP algorithm should stop, False otherwise
        """
        self.error = error
        return self._is_finished_check(error)

    @abstractmethod
    def _is_finished_check(self, error: float) -> bool:
        """
        Called after each iteration of the ICP algorithm.
        Should return True if the ICP algorithm should stop.
        :param: error: Error between the point cloud and the reference
        :return: True if the ICP algorithm should stop, False otherwise
        """
