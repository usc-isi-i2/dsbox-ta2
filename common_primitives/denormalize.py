import os
import typing

import pandas  # type: ignore

from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives

__all__ = ('DenormalizePrimitive',)

Inputs = container.Dataset
Outputs = container.Dataset


class Hyperparams(hyperparams.Hyperparams):
    starting_resource = hyperparams.Hyperparameter[typing.Union[str, None]](
        None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="From which resource to start denormalizing. If \"None\" then it starts from the dataset entry point.",
    )
    recursive = hyperparams.Hyperparameter[bool](
        True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Denormalize recursively?",
    )
    many_to_many = hyperparams.Hyperparameter[bool](
        True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Denormalize also many-to-many relations?",
    )


class DenormalizePrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which converts a dataset with multiple tabular resources into a dataset with only one tabular resource,
    based on known relations between tabular resources. Any resource which can be joined is joined, and other resources
    are discarded.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'f31f8c1f-d1c5-43e5-a4b2-2ae4a761ef2e',
            'version': '0.2.0',
            'name': "Denormalize datasets",
            'python_path': 'd3m.primitives.datasets.Denormalize',
            'source': {
               'name': common_primitives.__author__,
            },
            'installation': [{
               'type': metadata_base.PrimitiveInstallationType.PIP,
               'package_uri': 'git+https://gitlab.com/datadrivendiscovery/common-primitives.git@{git_commit}#egg=common_primitives'.format(
                   git_commit=utils.current_git_commit(os.path.dirname(__file__)),
               ),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.DATA_DENORMALIZATION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    # TODO: Implement support for M2M relations.
    # TODO: This should work recursively. If any resource being pulled brings more foreign key, they should be resolved as well. Without loops of course.
    # TODO: When copying metadata, copy also all individual metadata for columns and rows, and any recursive metadata for nested data.
    # TODO: Implement can_accept.
    # TODO: This should remove only resources which were joined to the main resource, and not all resources. Do we even want to remove other resources at all?
    # TODO: Add all column names together to "other names" metadata for column.
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        # If only one resource is in the dataset, we do not have anything to do.
        if inputs.metadata.query(())['dimension']['length'] == 1:
            return base.CallResult(inputs)

        main_resource_id = self.hyperparams['starting_resource']

        if main_resource_id is None:
            for resource_id in inputs.keys():
                if 'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint' in inputs.metadata.query((resource_id,)).get('semantic_types', []):
                    main_resource_id = resource_id
                    break

        if main_resource_id is None:
            raise ValueError("A Dataset with multiple resources without an entry point and no starting resource specified as a hyper-parameter.")

        main_data = inputs[main_resource_id]
        main_columns_length = inputs.metadata.query((main_resource_id, metadata_base.ALL_ELEMENTS))['dimension']['length']

        # There is only one resource now.
        top_level_metadata = dict(inputs.metadata.query(()))
        top_level_metadata['dimension'] = dict(top_level_metadata['dimension'])
        top_level_metadata['dimension']['length'] = 1

        metadata = inputs.metadata.clear(top_level_metadata, source=self).set_for_value(None, source=self)

        # Resource is not anymore an entry point.
        entry_point_metadata = dict(inputs.metadata.query((main_resource_id,)))
        entry_point_metadata['semantic_types'] = [
            semantic_type for semantic_type in entry_point_metadata['semantic_types'] if semantic_type != 'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint'
        ]
        metadata = metadata.update((main_resource_id,), entry_point_metadata, source=self)

        data = None

        for column_index in range(main_columns_length):
            column_metadata = inputs.metadata.query((main_resource_id, metadata_base.ALL_ELEMENTS, column_index))

            if 'foreign_key' not in column_metadata:
                # We just copy over data and metadata.
                data, metadata = self._add_column(main_resource_id, data, metadata, self._get_column(main_data, column_index), column_metadata)
            else:
                assert column_metadata['foreign_key']['type'] == 'COLUMN', column_metadata

                if 'column_index' in column_metadata['foreign_key']:
                    data, metadata = self._join_by_index(
                        main_resource_id, inputs, column_index, data, metadata, column_metadata['foreign_key']['resource_id'],
                        column_metadata['foreign_key']['column_index'],
                    )
                elif 'column_name' in column_metadata['foreign_key']:
                    data, metadata = self._join_by_name(
                        main_resource_id, inputs, column_index, data, metadata, column_metadata['foreign_key']['resource_id'],
                        column_metadata['foreign_key']['column_name'],
                    )
                else:
                    assert False, column_metadata

        resources = {}
        resources[main_resource_id] = data

        # Number of columns had changed.
        all_rows_metadata = dict(inputs.metadata.query((main_resource_id, metadata_base.ALL_ELEMENTS)))
        all_rows_metadata['dimension'] = dict(all_rows_metadata['dimension'])
        all_rows_metadata['dimension']['length'] = data.shape[1]
        metadata = metadata.update((main_resource_id, metadata_base.ALL_ELEMENTS), all_rows_metadata, for_value=resources, source=self)
        #print(metadata.pretty_print())
        #print(resources)
        #import pdb
        #pdb.set_trace()
        metadata.check(resources)

        dataset = container.Dataset(resources, metadata)

        return base.CallResult(dataset)

    def _join_by_name(self, main_resource_id: str, inputs: Inputs, inputs_column_index: int, data: typing.Optional[pandas.DataFrame],
                      metadata: metadata_base.DataMetadata, foreign_resource_id: str, foreign_column_name: str) -> typing.Tuple[pandas.DataFrame, metadata_base.DataMetadata]:
        for column_index in range(inputs.metadata.query((foreign_resource_id, metadata_base.ALL_ELEMENTS))['dimension']['length']):
            if inputs.metadata.query((foreign_resource_id, metadata_base.ALL_ELEMENTS, column_index)).get('name', None) == foreign_column_name:
                return self._join_by_index(main_resource_id, inputs, inputs_column_index, data, metadata, foreign_resource_id, column_index)

        raise ValueError(
            "Cannot resolve foreign key with column name '{column_name}' in resource with ID '{resource_id}'.".format(
                resource_id=foreign_resource_id,
                column_name=foreign_column_name,
            ),
        )

    def _join_by_index(self, main_resource_id: str, inputs: Inputs, inputs_column_index: int, data: typing.Optional[pandas.DataFrame],
                       metadata: metadata_base.DataMetadata, foreign_resource_id: str, foreign_column_index: int) -> typing.Tuple[pandas.DataFrame, metadata_base.DataMetadata]:
        main_column_metadata = inputs.metadata.query((main_resource_id, metadata_base.ALL_ELEMENTS, inputs_column_index))

        main_data = inputs[main_resource_id]
        foreign_data = inputs[foreign_resource_id]

        value_to_index = {}
        for value_index, value in enumerate(foreign_data.iloc[:, foreign_column_index]):
            # TODO: Check if values are not unique.
            value_to_index[value] = value_index

        rows = []
        for value in main_data.iloc[:, inputs_column_index]:
            rows.append([foreign_data.iloc[value_to_index[value], j] for j in range(len(foreign_data.columns))])

        if data is None:
            data_columns_length = 0
        else:
            data_columns_length = data.shape[1]

        # Copy over metadata.
        foreign_data_columns_length = inputs.metadata.query((foreign_resource_id, metadata_base.ALL_ELEMENTS))['dimension']['length']
        for column_index in range(foreign_data_columns_length):
            column_metadata = dict(inputs.metadata.query((foreign_resource_id, metadata_base.ALL_ELEMENTS, column_index)))

            # We cannot have duplicate primary keys, so we remove them. It might still be an unique key, but this is not necessary
            # true and should be verified because foreign key can reference same foreign row multiple times.
            if 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' in column_metadata.get('semantic_types', []):
                column_metadata['semantic_types'] = [
                    semantic_type for semantic_type in column_metadata['semantic_types'] if semantic_type != 'https://metadata.datadrivendiscovery.org/types/PrimaryKey'
                ]

            # If the original index column was an attribute, make sure the new index column is as well.
            if 'https://metadata.datadrivendiscovery.org/types/Attribute' in main_column_metadata.get('semantic_types', []):
                if 'https://metadata.datadrivendiscovery.org/types/Attribute' not in column_metadata['semantic_types']:
                    column_metadata['semantic_types'].append('https://metadata.datadrivendiscovery.org/types/Attribute')

            # If the original index column was a suggested target, make sure the new index column is as well.
            if 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget' in main_column_metadata.get('semantic_types', []):
                if 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget' not in column_metadata['semantic_types']:
                    column_metadata['semantic_types'].append('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')

            metadata = metadata.update((main_resource_id, metadata_base.ALL_ELEMENTS, data_columns_length + column_index), column_metadata, source=self)

        selected_data = pandas.DataFrame(rows)

        if data is None:
            data = selected_data
        else:
            data = pandas.concat((data, selected_data), axis=1)

        return data, metadata

    def _get_column(self, data: pandas.DataFrame, column_index: int) -> pandas.DataFrame:
        return data.iloc[:, [column_index]]

    def _add_column(self, main_resource_id: str, data: pandas.DataFrame, metadata: metadata_base.DataMetadata, column_data: pandas.DataFrame,
                    column_metadata: typing.Dict) -> typing.Tuple[pandas.DataFrame, metadata_base.DataMetadata]:
        assert column_data.shape[1] == 1

        if data is None:
            data = column_data
        else:
            data = pandas.concat((data, column_data), axis=1)

        metadata = metadata.update((main_resource_id, metadata_base.ALL_ELEMENTS, data.shape[1] - 1), column_metadata, source=self)

        return data, metadata
