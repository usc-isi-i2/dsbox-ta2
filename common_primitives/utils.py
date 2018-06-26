import copy
import math
import typing
from typing import *

import frozendict  # type: ignore
import torch  # type: ignore
import pandas  # type: ignore
from torch.autograd import Variable  # type: ignore
import numpy as np  # type: ignore

from d3m import container, exceptions
from d3m.metadata import base as metadata_base, hyperparams as metadata_hyperparams


def add_dicts(dict1: typing.Dict, dict2: typing.Dict) -> typing.Dict:
    summation = {}
    for key in dict1:
        summation[key] = dict1[key] + dict2[key]
    return summation


def sum_dicts(dictArray: typing.Sequence[typing.Dict]) -> typing.Dict:
    assert len(dictArray) > 0
    summation = dictArray[0]
    for dictionary in dictArray:
        summation = add_dicts(summation, dictionary)
    return summation


def to_variable(value: Any, requires_grad: bool = False) -> Variable:
    """
    Converts an input to torch Variable object
    input
    -----
    value - Type: scalar, Variable object, torch.Tensor, numpy ndarray
    requires_grad  - Type: bool . If true then we require the gradient of that object

    output
    ------
    torch.autograd.variable.Variable object
    """

    if isinstance(value, Variable):
        return value
    elif torch.is_tensor(value):
        return Variable(value.float(), requires_grad=requires_grad)
    elif isinstance(value, np.ndarray) or isinstance(value, container.ndarray):
        return Variable(torch.from_numpy(value.astype(float)).float(), requires_grad=requires_grad)
    elif value is None:
        return None
    else:
        return Variable(torch.Tensor([float(value)]), requires_grad=requires_grad)


def to_tensor(value: Any) -> torch.FloatTensor:
    """
    Converts an input to a torch FloatTensor
    """
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).float()
    else:
        raise ValueError('Unsupported type: {}'.format(type(value)))


def refresh_node(node: Variable) -> Variable:
    return torch.autograd.Variable(node.data, True)


def log_mvn_likelihood(mean: torch.FloatTensor, covariance: torch.FloatTensor, observation: torch.FloatTensor) -> torch.FloatTensor:
    """
    all torch primitives
    all non-diagonal elements of covariance matrix are assumed to be zero
    """
    k = mean.shape[0]
    variances = covariance.diag()
    log_likelihood = 0
    for i in range(k):
        log_likelihood += - 0.5 * torch.log(variances[i]) \
                          - 0.5 * k * math.log(2 * math.pi) \
                          - 0.5 * ((observation[i] - mean[i])**2 / variances[i])
    return log_likelihood


def covariance(data: torch.FloatTensor) -> torch.FloatTensor:
    """
    input: NxD torch array
    output: DxD torch array

    calculates covariance matrix of input
    """

    N, D = data.size()
    cov = torch.zeros([D, D]).type(torch.DoubleTensor)
    for contribution in (torch.matmul(row.view(D, 1),
                         row.view(1, D))/N for row in data):
        cov += contribution
    return cov


def remove_mean(data: torch.FloatTensor) -> typing.Tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    input: NxD torch array
    output: D-length mean vector, NxD torch array

    takes a torch tensor, calculates the mean of each
    column and subtracts it

    returns (mean, zero_mean_data)
    """

    N, D = data.size()
    mean = torch.zeros([D]).type(torch.DoubleTensor)
    for row in data:
        mean += row.view(D)/N
    zero_mean_data = data - mean.view(1, D).expand(N, D)
    return mean, zero_mean_data


def denumpify(unknown_object: typing.Any) -> typing.Any:
    """
    changes 'numpy.int's and 'numpy.float's etc to standard Python equivalents
    no effect on other data types
    """
    try:
        return unknown_object.item()
    except AttributeError:
        return unknown_object


M = typing.TypeVar('M', bound=metadata_base.Metadata)


# A copy of "Metadata._query" which starts ignoring "ALL_ELEMENTS" only at a certain depth.
# TODO: Make this part of metadata API.
def _query(selector: metadata_base.Selector, metadata_entry: typing.Optional[metadata_base.MetadataEntry], ignore_all_elements: int = None) -> frozendict.FrozenOrderedDict:
    if metadata_entry is None:
        return frozendict.FrozenOrderedDict()
    if len(selector) == 0:
        return metadata_entry.metadata

    segment, selector_rest = selector[0], selector[1:]

    if ignore_all_elements is not None:
        new_ignore_all_elements = ignore_all_elements - 1
    else:
        new_ignore_all_elements = None

    all_elements_metadata = _query(selector_rest, metadata_entry.all_elements, new_ignore_all_elements)
    if segment is metadata_base.ALL_ELEMENTS:
        metadata = all_elements_metadata
    elif segment in metadata_entry.elements:
        segment = typing.cast(metadata_base.SimpleSelectorSegment, segment)
        metadata = _query(selector_rest, metadata_entry.elements[segment], new_ignore_all_elements)
        if ignore_all_elements is None or ignore_all_elements > 0:
            metadata = metadata_base.Metadata()._merge_metadata(all_elements_metadata, metadata)
    elif ignore_all_elements is not None and ignore_all_elements <= 0:
        metadata = frozendict.FrozenOrderedDict()
    else:
        metadata = all_elements_metadata

    return metadata


def _copy_elements_metadata(source_metadata: metadata_base.Metadata, target_metadata: M, from_selector: metadata_base.ListSelector,
                           to_selector: metadata_base.ListSelector, selector: metadata_base.ListSelector, source: typing.Any) -> M:
    # "ALL_ELEMENTS" is always first, if it exists, which works in our favor here.
    # We are copying metadata for both "ALL_ELEMENTS" and elements themselves, so
    # we do not have to merge metadata together for elements themselves.
    elements = source_metadata.get_elements(from_selector + selector)

    for element in elements:
        new_selector = selector + [element]
        metadata = _query(from_selector + new_selector, source_metadata._current_metadata, len(from_selector))
        target_metadata = target_metadata.update(to_selector + new_selector, metadata, source=source)
        target_metadata = _copy_elements_metadata(source_metadata, target_metadata, from_selector, to_selector, new_selector, source)

    return target_metadata


# TODO: Make this part of metadata API.
def copy_elements_metadata(source_metadata: metadata_base.Metadata, target_metadata: M, from_selector: metadata_base.Selector,
                           to_selector: metadata_base.Selector = (), *, source: typing.Any = None) -> M:
    """
    Recursively copies metadata of all elements of ``source_metadata`` to ``target_metadata``, starting at the
    ``from_selector`` and to a selector starting at ``to_selector``.
    It does not copy metadata at the ``from_selector`` itself.
    """

    return _copy_elements_metadata(source_metadata, target_metadata, list(from_selector), list(to_selector), [], source)


# TODO: Make this part of metadata API.
def copy_metadata(source_metadata: metadata_base.Metadata, target_metadata: M, from_selector: metadata_base.Selector,
                 to_selector: metadata_base.Selector = (), *, source: typing.Any = None) -> M:
    """
    Recursively copies metadata of ``source_metadata`` to ``target_metadata``, starting at the
    ``from_selector`` and to a selector starting at ``to_selector``.
    """

    metadata = _query(from_selector, source_metadata._current_metadata, len(from_selector))
    target_metadata = target_metadata.update(to_selector, metadata, source=source)

    return copy_elements_metadata(source_metadata, target_metadata, from_selector, to_selector, source=source)


def select_columns(inputs: container.DataFrame, columns: typing.Sequence[metadata_base.SimpleSelectorSegment], *,
                   source: typing.Any = None) -> container.DataFrame:
    """
    Given a DataFrame, it returns a new DataFrame with data and metadata only for given ``columns``.
    Moreover, columns are renumbered based on the position in ``columns`` list.
    Top-level metadata stays unchanged, except for updating the length of the columns dimension to
    the number of columns.

    So if the ``columns`` is ``[3, 6, 5]`` then output DataFrame will have three columns, ``[0, 1, 2]``,
    mapping data and metadata for columns ``3`` to ``0``, ``6`` to ``1`` and ``5`` to ``2``.

    This allows also duplication of columns.
    """

    if not columns:
        raise exceptions.InvalidArgumentValueError("No columns selected.")

    outputs = inputs.iloc[:, columns]
    outputs.metadata = select_columns_metadata(inputs.metadata, columns, source=source)
    outputs.metadata = outputs.metadata.set_for_value(outputs, source=source)

    return outputs


# TODO: Make this part of metadata API.
# TODO: What happens to the metadata for rows? It should be copied over as well for copied columns.
def select_columns_metadata(inputs_metadata: metadata_base.DataMetadata, columns: typing.Sequence[metadata_base.SimpleSelectorSegment], *,
                            source: typing.Any = None) -> metadata_base.DataMetadata:
    """
    Given metadata, it returns a new metadata object with metadata only for given ``columns``.
    Moreover, columns are renumbered based on the position in ``columns`` list.
    Top-level metadata stays unchanged, except for updating the length of the columns dimension to
    the number of columns.

    So if the ``columns`` is ``[3, 6, 5]`` then output metadata will have three columns, ``[0, 1, 2]``,
    mapping metadata for columns ``3`` to ``0``, ``6`` to ``1`` and ``5`` to ``2``.

    This allows also duplication of columns.
    """

    if not columns:
        raise exceptions.InvalidArgumentValueError("No columns selected.")

    # This makes a copy.
    output_metadata = inputs_metadata.update(
        (metadata_base.ALL_ELEMENTS,),
        {
            'dimension': {
                'length': len(columns),
            },
        },
        source=source,
    )

    # TODO: Do this better. This change is missing an entry in metadata log.
    elements = output_metadata._current_metadata.all_elements.elements
    output_metadata._current_metadata.all_elements.elements = {}
    for i, column_index in enumerate(columns):
        if column_index in elements:
            # If "column_index" is really numeric, we re-enumerate it.
            if isinstance(column_index, int):
                output_metadata._current_metadata.all_elements.elements[i] = elements[column_index]
            else:
                output_metadata._current_metadata.all_elements.elements[column_index] = elements[column_index]

    return output_metadata


# TODO: Make this part of metadata API.
def list_columns_with_semantic_types(metadata: metadata_base.DataMetadata, semantic_types: typing.Sequence[str]) -> typing.Sequence[int]:
    """
    This is similar to ``get_columns_with_semantic_type``, but it returns all column indices
    for a dimension instead of ``ALL_ELEMENTS`` element.

    Moreover, it operates on a list of semantic types, where a column is returned
    if it matches any semantic type on the list.
    """

    columns = []

    for element in metadata.get_elements((metadata_base.ALL_ELEMENTS,)):
        metadata_semantic_types = metadata.query((metadata_base.ALL_ELEMENTS, element)).get('semantic_types', ())
        # TODO: Should we handle inheritance between semantic types here?
        if any(semantic_type in metadata_semantic_types for semantic_type in semantic_types):
            if element is metadata_base.ALL_ELEMENTS:
                return list(range(metadata.query((metadata_base.ALL_ELEMENTS,)).get('dimension', {}).get('length', 0)))
            else:
                columns.append(typing.cast(int, element))

    return columns


def remove_column(input: container.DataFrame, column_index: int, *, source: typing.Any = None) -> container.DataFrame:
    """
    Removes a column from a given ``input`` DataFrame and returns one without, together with all
    metadata for the column removed as well.

    It throws an exception if this would be the last column to remove.
    """

    # We are not using "drop" because we are dropping by the column index (to support columns with same name).
    columns = list(range(input.shape[1]))

    if not columns:
        raise ValueError("No columns to remove.")

    columns.remove(column_index)

    if not columns:
        raise ValueError("Removing a column would have removed the last column.")

    output = input.iloc[:, columns]
    output.metadata = select_columns_metadata(input.metadata, columns, source=source)
    output.metadata = output.metadata.set_for_value(output, generate_metadata=False, source=source)

    return output


def remove_column_metadata(input_metadata: metadata_base.DataMetadata, column_index: int, *, source: typing.Any = None) -> metadata_base.DataMetadata:
    """
    Analogous to ``remove_column`` but operates only on metadata.
    """

    columns = list(range(input_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']))

    if not columns:
        raise ValueError("No columns to remove.")

    columns.remove(column_index)

    if not columns:
        raise ValueError("Removing a column would have removed the last column.")

    return select_columns_metadata(input_metadata, columns, source=source)


def append_columns(left: container.DataFrame, right: container.DataFrame, *, use_right_metadata: bool = False, source: typing.Any = None) -> container.DataFrame:
    """
    Appends all columns from ``right`` to the right of ``left``, together with all metadata.

    Top-level metadata of ``right`` is ignored, not merged, except if ``use_right_metadata``
    is set, in which case top-level metadata of ``left`` is ignored and one from ``right`` is
    used instead.
    """

    outputs = pandas.concat([left, right], axis=1)

    right_metadata = right.metadata
    if use_right_metadata:
        right_metadata = right_metadata.set_for_value(outputs, generate_metadata=False, source=source)
    else:
        outputs.metadata = left.metadata.set_for_value(outputs, generate_metadata=False, source=source)

    outputs.metadata = append_columns_metadata(outputs.metadata, right_metadata, use_right_metadata=use_right_metadata, source=source)

    return outputs


# TODO: What happens to the metadata for rows? It should be copied over as well for copied columns.
def append_columns_metadata(left_metadata: metadata_base.DataMetadata, right_metadata: metadata_base.DataMetadata, use_right_metadata: bool = False, source: typing.Any = None) -> metadata_base.DataMetadata:
    """
    Appends metadata for all columns from ``right_metadata`` to the right of ``left_metadata``.

    Top-level metadata of ``right`` is ignored, not merged, except if ``use_right_metadata``
    is set, in which case top-level metadata of ``left`` is ignored and one from ``right`` is
    used instead.
    """

    left_length = left_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']
    right_length = right_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']

    if not use_right_metadata:
        outputs_metadata = left_metadata

        for column_index in range(right_length):
            outputs_metadata = copy_metadata(right_metadata, outputs_metadata, [metadata_base.ALL_ELEMENTS, column_index], [metadata_base.ALL_ELEMENTS, left_length + column_index], source=source)

    else:
        # This makes a copy.
        outputs_metadata = right_metadata.update(
            (metadata_base.ALL_ELEMENTS,),
            {},
            source=source,
        )
        # TODO: Do this better. Make all this a function which does this change properly.
        outputs_metadata._current_metadata.all_elements.elements = copy.copy(outputs_metadata._current_metadata.all_elements.elements)

        # TODO: Do this better. This change is missing an entry in metadata log.
        # Move columns and make space for left metadata to be prepended.
        # We iterate over a list so that we can change dict while iterating.
        for element in sorted(outputs_metadata._current_metadata.all_elements.elements.keys(), reverse=True):
            metadata = outputs_metadata._current_metadata.all_elements.elements[element]
            del outputs_metadata._current_metadata.all_elements.elements[element]
            outputs_metadata._current_metadata.all_elements.elements[element + left_length] = metadata

        for column_index in range(left_length):
            outputs_metadata = copy_metadata(left_metadata, outputs_metadata, [metadata_base.ALL_ELEMENTS, column_index], [metadata_base.ALL_ELEMENTS, column_index], source=source)

    # There can be only one primary key index.
    index_columns = list_columns_with_semantic_types(outputs_metadata, ('https://metadata.datadrivendiscovery.org/types/PrimaryKey',))
    index_columns = typing.cast(typing.List[int], list(index_columns))
    if len(index_columns) > 1:
        # Remove all except first primary index semantic types, and convert it to unique key.
        for column_index in sorted(index_columns)[1:]:
            outputs_metadata = outputs_metadata.remove_semantic_type([metadata_base.ALL_ELEMENTS, column_index], 'https://metadata.datadrivendiscovery.org/types/PrimaryKey', source=source)
            outputs_metadata = outputs_metadata.add_semantic_type([metadata_base.ALL_ELEMENTS, column_index], 'https://metadata.datadrivendiscovery.org/types/UniqueKey', source=source)

    outputs_metadata = outputs_metadata.update((metadata_base.ALL_ELEMENTS,), {'dimension': {'length': left_length + right_length}}, source=source)

    return outputs_metadata


def replace_columns(inputs: container.DataFrame, columns: container.DataFrame, column_indices: typing.List[int], *, source: typing.Any = None) -> container.DataFrame:
    """
    Replaces columns listed in ``column_indices`` with ``columns``, in order, in ``inputs``.

    Top-level metadata of ``columns`` is ignored.
    """

    if columns.shape[1] != len(column_indices):
        raise exceptions.InvalidArgumentValueError("Columns do not match column indices.")

    if not column_indices:
        return inputs

    outputs = copy.copy(inputs)
    for i, column_index in enumerate(column_indices):
        outputs.iloc[:, column_index] = columns.iloc[:, i]

    outputs.metadata = inputs.metadata.set_for_value(outputs, generate_metadata=False, source=source)
    outputs.metadata = replace_columns_metadata(outputs.metadata, columns.metadata, column_indices, source=source)

    return outputs


# TODO: What happens to the metadata for rows? It should be copied over as well for copied columns.
def replace_columns_metadata(inputs_metadata: metadata_base.DataMetadata, columns_metadata: metadata_base.DataMetadata, column_indices: typing.List[int], *, source: typing.Any = None) -> metadata_base.DataMetadata:
    """
    Analogous to ``replace_columns`` but operates only on metadata.
    """

    if columns_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length'] != len(column_indices):
        raise exceptions.InvalidArgumentValueError("Columns do not match column indices.")

    outputs_metadata = inputs_metadata
    for i, column_index in enumerate(column_indices):
        outputs_metadata = outputs_metadata.remove_column(column_index, source=source)
        outputs_metadata = copy_metadata(columns_metadata, outputs_metadata, (metadata_base.ALL_ELEMENTS, i), (metadata_base.ALL_ELEMENTS, column_index), source=source)

    # A workaround for a bug in d3m core package v2018.6.5, where "remove_column" does not preserve "for_value".
    # TODO: Remove after a release of a newer package.
    outputs_metadata = outputs_metadata.set_for_value(inputs_metadata.for_value, generate_metadata=False, source=source)

    return outputs_metadata


def check_same_height(metadata1: metadata_base.DataMetadata, metadata2: metadata_base.DataMetadata) -> None:
    if metadata1.query(())['dimension']['length'] != metadata2.query(())['dimension']['length']:
        raise ValueError("Data does not match in the number of samples.")


def get_index_column(metadata: metadata_base.DataMetadata) -> typing.Optional[int]:
    """
    Returns column index of the primary index column.
    """

    index_columns = list_columns_with_semantic_types(metadata, ('https://metadata.datadrivendiscovery.org/types/PrimaryKey',))
    assert len(index_columns) < 2
    if index_columns:
        index = index_columns[0]
    else:
        index = None

    return index


def horizontal_concat(left: container.DataFrame, right: container.DataFrame, *, use_index: bool = True,
                      remove_second_index: bool = True, use_right_metadata: bool = False, source: typing.Any = None) -> container.DataFrame:
    """
    Similar to ``append_columns``, but it respects primary index columns, by default.

    It is required that both inputs have the same number of samples.
    """

    check_same_height(left.metadata, right.metadata)

    left_index = get_index_column(left.metadata)
    right_index = get_index_column(right.metadata)

    if left_index is not None and right_index is not None:
        if use_index:
            old_right_metadata = right.metadata
            #       This should be relatively easy because we can just modify
            #       right.metadata._current_metadata.metadata map (and create a new action type for the log).
            right = right.set_index(right.iloc[:, right_index]).reindex(left.iloc[:, left_index]).reset_index(drop=True)
            # TODO: Reorder metadata rows as well.
            right.metadata = old_right_metadata

        # Removing second primary key column.
        if remove_second_index:
            right = remove_column(right, right_index, source=source)

    return append_columns(left, right, use_right_metadata=use_right_metadata, source=source)


def horizontal_concat_metadata(left_metadata: metadata_base.DataMetadata, right_metadata: metadata_base.DataMetadata, *, use_index: bool = True,
                               remove_second_index: bool = True, use_right_metadata: bool = False, source: typing.Any = None) -> metadata_base.DataMetadata:
    """
    Similar to ``append_columns_metadata``, but it respects primary index columns, by default.

    It is required that both inputs have the same number of samples.
    """

    check_same_height(left_metadata, right_metadata)

    left_index = get_index_column(left_metadata)
    right_index = get_index_column(right_metadata)

    if left_index is not None and right_index is not None:
        if use_index:
            # TODO: Reorder metadata rows as well.
            pass

        # Removing second primary key column.
        if remove_second_index:
            right_metadata = remove_column_metadata(right_metadata, right_index, source=source)

    return append_columns_metadata(left_metadata, right_metadata, use_right_metadata=use_right_metadata, source=source)


def get_columns_to_produce(metadata: metadata_base.DataMetadata, hyperparams: metadata_hyperparams.Hyperparams, can_produce_column: typing.Callable) -> typing.Tuple[typing.List[int], typing.List[int]]:
    all_columns = list(hyperparams['use_columns'])
    if not all_columns:
        all_columns = list(range(metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']))

        all_columns = [column_index for column_index in all_columns if column_index not in hyperparams['exclude_columns']]

    columns_to_produce = []
    columns_not_to_produce = []
    for column_index in all_columns:
        if can_produce_column(column_index):
            columns_to_produce.append(column_index)
        else:
            columns_not_to_produce.append(column_index)
    return columns_to_produce, columns_not_to_produce



def get_columns_to_use(metadata: metadata_base.DataMetadata, use_columns: typing.Sequence[int], exclude_columns: typing.Sequence[int],
                       can_use_column: typing.Callable) -> typing.Tuple[typing.List[int], typing.List[int]]:
    """
    A helper function which computes a list of columns to use and a list of columns to ignore
    given ``use_columns``, ``exclude_columns``, and a ``can_use_column`` function which should
    return ``True`` when column can be used.
    """

    all_columns = list(use_columns)

    # If "use_columns" is provided, this is our view of which columns exist.
    if not all_columns:
        # Otherwise, we start with all columns.
        all_columns = list(range(metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']))

        # And remove those in "exclude_columns".
        all_columns = [column_index for column_index in all_columns if column_index not in exclude_columns]

    # Now we create a list of columns for which "can_use_column" returns "True",
    # but also a list of columns for which it does not. The latter can be used
    # to determine if there is an error or warning. For example, when using "use_columns",
    # ideally, "columns_not_to_use" should be empty or a warning should be made.
    # Or, some primitives might require to operate on all columns, so "columns_not_to_use"
    # is empty, an error should be raised.
    columns_to_use = []
    columns_not_to_use = []
    for column_index in all_columns:
        if can_use_column(column_index):
            columns_to_use.append(column_index)
        else:
            columns_not_to_use.append(column_index)

    return columns_to_use, columns_not_to_use

def combine_columns(return_result: str, add_index_column: bool, inputs: container.DataFrame, column_indices: typing.List[int],
                    columns: typing.List[container.DataFrame], *, source: typing.Any = None) -> container.DataFrame:
    """
    Method which appends existing columns, replaces them, or creates new result from them, based on
    ``return_result`` argument, which can be ``append``, ``replace``, or ``new``.

    ``add_index_column`` controls if when creating a new result, a primary index column should be added
    if it is not already among columns.

    ``inputs`` is a DataFrame for which we are appending on replacing columns, or if we are creating new result,
    from where a primary index column can be taken.

    ``column_indices`` controls which columns in ``inputs`` were used to create ``columns``,
    and which columns should be replaced when replacing them.

    ``columns`` is a list of DataFrames which all together should match the columns in ``column_indices``.
    The reason it is a list is to make it easier to operate per-column when preparing ``columns`` and not
    have to concat them all together unnecessarily.

    Top-level metadata in ``columns`` is ignored, except when creating new result.
    In that case top-level metadata from the first element in the list is used.
    """

    all_columns_length = sum(column.shape[1] for column in columns)

    if return_result == 'append':
        outputs = inputs
        for column in columns:
            outputs = append_columns(outputs, column, source=source)

    elif return_result == 'replace':
        if all_columns_length != len(column_indices):
            raise exceptions.InvalidArgumentValueError("Columns do not match column indices.")

        outputs = inputs
        columns_replaced = 0
        for column in columns:
            column_length = column.shape[1]
            outputs = replace_columns(outputs, column, column_indices[columns_replaced:columns_replaced + column_length], source=source)
            columns_replaced += column_length

    elif return_result == 'new':
        if not all_columns_length:
            raise ValueError("No columns produced.")

        outputs = columns[0]
        for column in columns[1:]:
            outputs = append_columns(outputs, column, source=source)

        if add_index_column:
            index_column = get_index_column(inputs.metadata)
            if index_column is not None and index_column not in column_indices:
                outputs = append_columns(select_columns(inputs, [index_column], source=source), outputs, use_right_metadata=True, source=source)

    else:
        raise exceptions.InvalidArgumentValueError("\"return_result\" has an invalid value: {return_result}".format(return_result=return_result))

    return outputs


def combine_columns_metadata(return_result: str, add_index_column: bool, inputs_metadata: metadata_base.DataMetadata, column_indices: typing.List[int],
                             columns_metadata: typing.List[metadata_base.DataMetadata], *, source: typing.Any = None) -> metadata_base.DataMetadata:
    """
    Analogous to ``combine_columns`` but operates only on metadata.
    """

    all_columns_length = sum(column_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length'] for column_metadata in columns_metadata)

    if return_result == 'append':
        outputs_metadata = inputs_metadata
        for column_metadata in columns_metadata:
            outputs_metadata = append_columns_metadata(outputs_metadata, column_metadata, source=source)

    elif return_result == 'replace':
        if all_columns_length != len(column_indices):
            raise exceptions.InvalidArgumentValueError("Columns do not match column indices.")

        outputs_metadata = inputs_metadata
        columns_replaced = 0
        for column_metadata in columns_metadata:
            column_length = column_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']
            outputs_metadata = replace_columns_metadata(outputs_metadata, column_metadata, column_indices[columns_replaced:columns_replaced + column_length], source=source)
            columns_replaced += column_length

    elif return_result == 'new':
        if not all_columns_length:
            raise ValueError("No columns produced.")

        outputs_metadata = columns_metadata[0]
        for column_metadata in columns_metadata[1:]:
            outputs_metadata = append_columns_metadata(outputs_metadata, column_metadata, source=source)

        if add_index_column:
            index_column = get_index_column(inputs_metadata)
            if index_column is not None and index_column not in column_indices:
                outputs_metadata = append_columns_metadata(select_columns_metadata(inputs_metadata, [index_column], source=source), outputs_metadata, use_right_metadata=True, source=source)

    else:
        raise exceptions.InvalidArgumentValueError("\"return_result\" has an invalid value: {return_result}".format(return_result=return_result))

    return outputs_metadata
