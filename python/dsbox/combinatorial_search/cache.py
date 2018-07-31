from math import sqrt, log
import traceback
import importlib
import typing
from multiprocessing import Pool, current_process, Manager
from dsbox.template.configuration_space import ConfigurationPoint

T = typing.TypeVar("T")


class CacheManager:
    """
    CacheManager encapsulates the ecosystem of caching in our search system. It bundles two set
    of caches: primitive and candidate. These caches will be used by our system to store the
    intermediate computations in two levels (primitive-level and candidate-level), which can be
    used in later iterations of search to
    """

    def __init__(self):
        """
        Initializes the manager, and the two cache objects - candidates and primitives - that
        will be used to prevent recalculation and increase performance
        Args:

        """
        self.manager = Manager()

        self.candidate_cache = CandidateCache(self.manager)

        self.primitive_cache = PrimitivesCache(self.manager)

    def cleanup(self):
        """
        Cleans the manager and its associated distributed cache modules


        """
        self.manager.shutdown()


class CandidateCache:
    def __init__(self, manager: Manager):
        self.storage = manager.dict()

    def lookup(self, candidate: ConfigurationPoint[T]) -> \
            typing.Tuple[ConfigurationPoint[T], float]:

        key = hash(str(candidate))
        if key in self.storage:
            line = self.storage[key]
            print("[INFO] hit@Candidate: ({},{})".format(key, line["id"]))
            # print("[INFO] candidate cache Hit@{}:{}".format(key, line['candidate']))
            return line['candidate'], line['value']
        else:
            return (None, None)

    def push(self, result: typing.Dict, candidate: ConfigurationPoint[T]) -> None:
        key = hash(str(candidate))
        cand_id = result['fitted_pipeline'].id if result else None
        value = result['test_metrics'][0]['value'] if result else None
        # add value to candidate cache

        if self._is_candidate_hit(candidate, self.storage):
            assert value == self.storage[key]['value'], \
                "New value for candidate:" + str(candidate)
            return

        print("[INFO] push@Candidate: ({},{})".format(key, cand_id))
        self.storage[key] = {
            "candidate": candidate,
            "id": cand_id,
            "value": value,
        }

    def is_hit(self, candidate: ConfigurationPoint[T]) -> bool:
        return hash(str(candidate)) in self.storage


class PrimitivesCache:
    def __init__(self, manager: Manager):
        self.storage = manager.dict()

    def push(self):
        pass

    def lookup(self):
        pass

    def is_hit(self):
        pass
