from math import sqrt, log
import traceback
import importlib
import typing
from multiprocessing import Pool, current_process, Manager, Lock
from dsbox.template.configuration_space import ConfigurationPoint
from d3m.metadata.pipeline import PrimitiveStep
from d3m.container.dataset import Dataset
from d3m.primitive_interfaces.base import PrimitiveBase
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
        self.manager = [Manager()]*2

        self.candidate_cache = CandidateCache(self.manager[0])

        self.primitive_cache = PrimitivesCache(self.manager[1])

    def cleanup(self):
        """
        Cleans the manager and its associated distributed cache modules


        """
        print("[INFO] cleanup Cache Manager. candidate_cache:", len(self.candidate_cache.storage))
        for m in self.manager:
            m.shutdown()


class CandidateCache:
    def __init__(self, manager: Manager=None):
        if manager is not None:
            self.storage = manager.dict()
        else:
            self.storage = {}

    def lookup(self, candidate: ConfigurationPoint[T]) -> typing.Dict:

        key = CandidateCache._get_hash(candidate)
        if key in self.storage:
            print("[INFO] hit@Candidate: ({})".format(key))
            return self.storage[key]
        else:
            return (None, None)

    def push_None(self, candidate: ConfigurationPoint[T]) -> None:
        result = {
            "configuration": candidate,
        }
        self.push(result=result)

    def push(self, result: typing.Dict) -> None:
        assert result is not None and 'configuration' in result, \
            'invalid push in candidate_cache: {}'.format(result)

        candidate = result['configuration']
        key = CandidateCache._get_hash(candidate)
        cand_id = result['fitted_pipeline'].id if 'fitted_pipeline' in result else None
        value = result['test_metrics'][0]['value'] if 'test_metrics' in result else None

        # check the candidate in cache. If duplicate is found the metric values must match
        if self.is_hit(candidate):
            assert value == self.storage[key]['value'] or self.storage[key]['value'] is None, \
                "New value for candidate:" + str(candidate)
            return

        # push the candidate and its value into the cache
        print("[INFO] push@Candidate: ({},{})".format(key, cand_id))
        self.storage[key] = {
            "candidate": candidate,
            "id": cand_id,
            "value": value,
        }

    def is_hit(self, candidate: ConfigurationPoint[T]) -> bool:
        return CandidateCache._get_hash(candidate) in self.storage

    @staticmethod
    def _get_hash(candidate: ConfigurationPoint[T]) -> int:
        return hash(str(candidate))

class PrimitivesCache:
    def __init__(self, manager: Manager=None):
        if manager is not None:
            self.storage = manager.dict()
            self.write_lock = manager.Lock()
        else:
            self.storage = {}
            self.write_lock = Lock()

    def push(self, hash_prefix: int, pipe_step: PrimitiveStep, primitive_arguments: typing.Dict,
             primitives_output: Dataset, model: PrimitiveBase) -> int:
        prim_name, prim_hash = PrimitivesCache._get_hash(
            hash_prefix=hash_prefix, pipe_step=pipe_step, primitive_arguments=primitive_arguments)

        return self.push_key(prim_hash=prim_hash, prim_name=prim_name,
                             model=model, primitives_output=primitives_output)

    def push_key(self, prim_hash: int, prim_name: int, model: PrimitiveBase,
                 primitives_output: typing.Dict) -> int:
        self.write_lock.acquire(blocking=True)
        try:
            if not self.is_hit_key(prim_name=prim_name, prim_hash=prim_hash):
                self.storage[(prim_name, prim_hash)] = (primitives_output, model)
                print("[INFO] Push@cache:", (prim_name, prim_hash))
                return 0
            else:
                print("[WARN] Double-push in Primitives Cache")
                return 1
        finally:
            self.write_lock.release()

    def lookup(self, hash_prefix: int, pipe_step: PrimitiveStep,
               primitive_arguments: typing.Dict) -> typing.Tuple[Dataset, PrimitiveBase]:

        prim_name, prim_hash = self._get_hash(hash_prefix=hash_prefix, pipe_step=pipe_step,
                                              primitive_arguments=primitive_arguments)

        return self.lookup_key(prim_name=prim_name, prim_hash=prim_hash)

    def lookup_key(self, prim_hash: int, prim_name: int) -> typing.Tuple[Dataset, PrimitiveBase]:
        if self.is_hit_key(prim_name=prim_name, prim_hash=prim_hash):
            print("[INFO] Hit@cache:", (prim_name, prim_hash))
            return self.storage[(prim_name, prim_hash)]
        else:
            return (None, None)

    def is_hit(self, hash_prefix: int, pipe_step: PrimitiveStep,
               primitive_arguments: typing.Dict) -> bool:
        return (
                (PrimitivesCache._get_hash(
                    hash_prefix=hash_prefix, pipe_step=pipe_step,
                    primitive_arguments=primitive_arguments)
                ) in self.storage)

    def is_hit_key(self, prim_hash: int, prim_name: int) -> bool:
        return (prim_name, prim_hash) in self.storage

    @staticmethod
    def _get_hash(hash_prefix: int, pipe_step: PrimitiveStep,
                  primitive_arguments: typing.Dict) -> typing.Tuple[int,int]:
        hyperparam_hash = hash(str(pipe_step.hyperparams.items()))
        dataset_id = ""
        dataset_digest = ""
        try:
            dataset_id = str(primitive_arguments['inputs'].metadata.query(())['id'])
            dataset_digest = str(primitive_arguments['inputs'].metadata.query(())['digest'])
        except:
            pass
        dataset_hash = hash(str(primitive_arguments) + dataset_id + dataset_digest)
        prim_name = str(pipe_step.primitive)
        prim_hash = hash(str([hyperparam_hash, dataset_hash, hash_prefix]))
        return prim_name, prim_hash
