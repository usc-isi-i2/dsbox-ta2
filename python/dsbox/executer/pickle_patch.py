import sys
import copyreg
import types
import functools

def reduce_method(method):
    '''Reducer for methods.'''
    return (
        getattr,
        (

            method.__self__ or method.__self__.__class__,
            # `im_self` for bound methods, `im_class` for unbound methods.

            method.__func__.__name__

        )
    )

def reduce_module(module):
    '''Reducer for modules.'''
    return (_normal_import, (module.__name__,))


def _normal_import(module_name):
    if '.' in module_name:
        package_name, submodule_name = module_name.rsplit('.', 1)
        package = __import__(module_name)
        return functools.reduce(getattr,
                                [package] + module_name.split('.')[1:])
    else:
        return __import__(module_name)


def _get_ellipsis():
    '''Get the `Ellipsis`.'''
    return Ellipsis

def reduce_ellipsis(ellipsis):
    '''Reducer for `Ellipsis`.'''
    return (
        _get_ellipsis,
        ()
    )

def _get_quitter():
    '''Get the `Quitter`.'''
    return sys.exit

def reduce_quitter(quitter):
    '''Reducer for `site.Quitter`.'''
    return (
        _get_quitter,
        ()
    )

def _get_not_implemented():
    '''Get the `Quitter`.'''
    return NotImplemented

def reduce_not_implemented(not_implemented):
    '''Reducer for `site.Quitter`.'''
    return (
        _get_not_implemented,
        ()
    )

def pickle_lock(lock):
    return threading.Lock, (lock.locked(),)

def unpickle_lock(locked, *args):
    lock = threading.Lock()
    if locked:
        if not lock.acquire(False):
            raise pickle.UnpicklingError("Cannot acquire lock")


copyreg.pickle(types.MethodType, reduce_method)
copyreg.pickle(types.ModuleType, reduce_module)
copyreg.pickle(type(Ellipsis), reduce_ellipsis)
copyreg.pickle(type(exit), reduce_quitter)
copyreg.pickle(type(NotImplemented), reduce_not_implemented)
