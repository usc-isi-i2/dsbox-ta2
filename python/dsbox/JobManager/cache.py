import traceback
import typing
import copy
import logging
from multiprocessing import Manager
from dsbox.template.configuration_space import ConfigurationPoint
from d3m.metadata.pipeline import PrimitiveStep
from d3m.container.dataset import Dataset
from d3m.container.pandas import DataFrame
from d3m.primitive_interfaces.base import PrimitiveBase
from dsbox.combinatorial_search.search_utils import comparison_metrics

T = typing.TypeVar("T")
_logger = logging.getLogger(__name__)


class DummyLock:
    def acquire(self, blocking=True, timeout=-1):
        pass

    def release(self):
        pass


class DummyManager:
    def dict(self):
        return dict()

    def Lock(self):
        return DummyLock()

    def shutdown(self):
        pass


class CacheManager:
    """
    CacheManager encapsulates the ecosystem of caching in our search system. It bundles two set
    of caches: primitive and candidate. These caches will be used by our system to store the
    intermediate computations in two levels (primitive-level and candidate-level), which can be
    used in later iterations of search to
    """

    def __init__(self, is_multiprocessing=False):
        """
        Initializes the manager, and the two cache objects - candidates and primitives - that
        will be used to prevent recalculation and increase performance
        Args:

        """
        if is_multiprocessing:
            self.manager = [Manager()]*2
        else:
            self.manager = [DummyManager()]*2

        self.candidate_cache = CandidateCache(self.manager[0])

        self.primitive_cache = PrimitivesCache(self.manager[1])

    def cleanup(self):
        """
        Cleans the manager and its associated distributed cache modules


        """
        _logger.info("Cleanup Cache Manager. candidate_cache:{} primitive_cache:{}".format(
            len(self.candidate_cache.storage), len(self.primitive_cache.storage)))
        self.candidate_cache.storage.clear()
        self.primitive_cache.storage.clear()

    def shutdown(self):
        '''
        Shutdown cache manager
        '''
        _logger.info("Shutdown Cache Manager. candidate_cache:{} primitive_cache:{}".format(
            len(self.candidate_cache.storage), len(self.primitive_cache.storage)))
        for m in self.manager:
            m.shutdown()


class CandidateCache:
    comparison_metrics = ['cross_validation_metrics', 'test_metrics', 'training_metrics']
    S_INVALID = "DUMMY"
    S_VALID = "FULL"

    def __init__(self, manager):
        self.storage = manager.dict()

    def lookup(self, candidate: ConfigurationPoint[T]) -> typing.Dict:

        key = CandidateCache._get_hash(candidate)
        if key in self.storage:
            _logger.info("[INFO] hit@Candidate: ({})".format(key))
            return self.storage[key]
        else:
            return None

    def push_None(self, candidate: ConfigurationPoint[T]) -> None:
        result = {
            "configuration": candidate,
            "status": CandidateCache.S_INVALID,
        }
        self.push(result=result)

    def push(self, result: typing.Dict) -> None:
        assert (result is not None and
                'configuration' in result), 'invalid push in candidate_cache: {}'.format(result)

        candidate = result['configuration']
        key = CandidateCache._get_hash(candidate)

        update = {}
        for k in comparison_metrics + ['configuration', 'status']:
            update[k] = copy.deepcopy(result[k]) if k in result else None
        update['id'] = result['fitted_pipeline'].id if 'fitted_pipeline' in result else None

        # check the candidate in cache. If duplicate is found the metric values must match
        self._check_update_format(candidate, key, update)

        # push the candidate and its value into the cache
        _logger.info(f"push@Candidate_{update['status']}: ({key})")
        _logger.debug(f"push@Candidate_{update['status']}: ({key}, {update})")
        assert 'status' in update
        self.storage[key] = update

    def _check_update_format(self, candidate, key, update):
        """
        checks the format of the update dict. If the candidate is already pushed into the cache
        we need to check the consistency of cache in two cases: 1) if the version that is in
        cache is the dummy push (push_none) then we need to make sure the format is correct and
        the status is set to valid. 2) if the candidate is in cache in valid format the results
        need to be equal (this case should not happen based on current implementation of the
        search methods as they do not submit pipelines to multiple workers but may happen in
        future)
        Args:
            candidate:
            key:
            update:

        Returns:

        """
        if self.is_hit(candidate):
            match = self.storage[key]
            assert match is not None
            assert match['status'] is not None
            assert match['configuration'] is not None
            for k in comparison_metrics:
                # _logger.debug('_check_update_format: metric={k}')
                if match['status'] == CandidateCache.S_INVALID:
                    update['status'] = CandidateCache.S_VALID
                assert k in match
                assert k in update

                # _logger.debug('_check_update_format: update[k]={update[k]}')
                assert update[k] is None or isinstance(update[k], list)
                assert match[k] is None or isinstance(match[k], list)

                assert match[k] is None or all('value' in m for m in match[k]), f"{match}"
                assert update[k] is None or all('value' in u for u in update[k]), f"{update}"

                assert (match[k] is None or
                        all(
                            m['value'] is None or
                            u['value'] == m['value'] for (m, u) in zip(match[k], update[k])
                            )
                        ), "New value for candidate:" + str(candidate)
        else:
            # FIXME I am not sure whether to have this part or not. For now I just throw
            # exception to be able to catch this situation.
            if 'status' not in update:
                assert False
                update['status'] = CandidateCache.S_VALID

    def is_hit(self, candidate: ConfigurationPoint[T]) -> bool:
        return CandidateCache._get_hash(candidate) in self.storage

    @staticmethod
    def _get_hash(candidate: ConfigurationPoint[T]) -> int:
        return hash(str(candidate))


class PrimitivesCache:
    """
        based on my profiling the dataset can be hashed by rate of 500MBpS (on dsbox server)
        which is more than enough for our applications:
            > A = pd.DataFrame(np.random.rand(1000*1000, 1000))
            > A.shape
             (10000000, 1000)
            > %timeit hash(A.values.tobytes())
             7.66 s ± 4.87 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    todo:
        1. only add the primitive to the cache if it is reasonable to do so. (if the time to
        recompute it is significantly more than reading it) (500 MBps)


    """
    def __init__(self, manager: Manager = DummyManager()):
        self.storage = manager.dict()
        self.write_lock = manager.Lock()

    def push(self, hash_prefix: int, pipe_step: PrimitiveStep, primitive_arguments: typing.Dict,
             fitting_time: int, model: PrimitiveBase) -> int:
        prim_name, prim_hash = PrimitivesCache._get_hash(
            hash_prefix=hash_prefix, pipe_step=pipe_step, primitive_arguments=primitive_arguments)

        return self.push_key(prim_hash=prim_hash, prim_name=prim_name, model=model,
                             fitting_time=fitting_time)

    def push_key(self, prim_hash: int, prim_name: int, model: PrimitiveBase,
                 fitting_time: int) -> int:

        self.write_lock.acquire(blocking=True)

        try:
            if not self.is_hit_key(prim_name=prim_name, prim_hash=prim_hash):
                self.storage[(prim_name, prim_hash)] = (fitting_time, model)
                _logger.debug(f"[INFO] Push@cache:{prim_name},{prim_hash}")
                # print(f"[INFO] Push@cache:{prim_name},{prim_hash}")
                return 0
            else:
                # print("[WARN] Double-push in Primitives Cache")
                return 1
        except:
            _logger.warning('Caching model failed. Most likely the primitive does not pickle properly.')
            traceback.print_exc()
        finally:
            # print("[INFO] released")
            self.write_lock.release()

    def lookup(self, hash_prefix: int, pipe_step: PrimitiveStep,
               primitive_arguments: typing.Dict) -> typing.Tuple[Dataset, PrimitiveBase]:

        prim_name, prim_hash = self._get_hash(hash_prefix=hash_prefix, pipe_step=pipe_step,
                                              primitive_arguments=primitive_arguments)

        return self.lookup_key(prim_name=prim_name, prim_hash=prim_hash)

    def lookup_key(self, prim_hash: int, prim_name: int) -> typing.Tuple[Dataset, PrimitiveBase]:
        if self.is_hit_key(prim_name=prim_name, prim_hash=prim_hash):
            _logger.debug("[INFO] Hit@cache: {},{}".format(prim_name, prim_hash))
            # print("[INFO] Hit@cache: {},{}".format(prim_name, prim_hash))
            return self.storage[(prim_name, prim_hash)]
        else:
            return (None, None)

    def is_hit(self, hash_prefix: int, pipe_step: PrimitiveStep,
               primitive_arguments: typing.Dict) -> bool:
        return (
                (
                    PrimitivesCache._get_hash(
                        hash_prefix=hash_prefix, pipe_step=pipe_step,
                        primitive_arguments=primitive_arguments)
                ) in self.storage)

    def is_hit_key(self, prim_hash: int, prim_name: int) -> bool:
        return (prim_name, prim_hash) in self.storage

    @staticmethod
    def _get_hash(pipe_step: PrimitiveStep, primitive_arguments: typing.Dict,
                  hash_prefix: int=None) -> typing.Tuple[int, int]:
        prim_name = str(pipe_step.primitive)
        hyperparam_hash = hash(str(pipe_step.hyperparams.items()))
        dataset_id = ""
        dataset_digest = ""
        try:
            dataset_id = str(primitive_arguments['inputs'].metadata.query(())['id'])
            dataset_digest = str(primitive_arguments['inputs'].metadata.query(())['digest'])
        except:
            pass

        # print(primitive_arguments['inputs'])
        # TODO the list part is related to timeseries datasets. chcek this with team
        assert (isinstance(primitive_arguments['inputs'], Dataset) or
                isinstance(primitive_arguments['inputs'], DataFrame) or
                isinstance(primitive_arguments['inputs'], typing.List)), \
               f"inputs type not valid {type(primitive_arguments['inputs'])}"

        if hash_prefix is None:
            _logger.debug("Primtive cache, hash computed in prefix mode")
            dataset_value_hash = hash(str(primitive_arguments['inputs']))
        else:
            dataset_value_hash = hash(primitive_arguments['inputs'].values.tobytes())

        dataset_hash = hash(str(dataset_value_hash) + dataset_id + dataset_digest)
        prim_hash = hash(str([hyperparam_hash, dataset_hash, hash_prefix]))
        _logger.debug("[INFO] hash: {}, {}".format(prim_name, prim_hash))
        return prim_name, prim_hash
