from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.container.dataset import D3MDatasetLoader, Dataset
from d3m.metadata import base as metadata_base
from d3m.metadata.base import Metadata
import networkx as nx
import json
import os


class Runtime:
    """
    Class to run the build and run a Pipeline.

    Attributes
    ----------
    pipeline_description : Pipeline
        A pipeline description to be executed.
    primitives_arguments
        List of indexes reference to the arguments for each step.
    execution_order
        List of indexes that contains the execution order.
    pipeline
        List of different models generated by the primitives.
    outputs
        List of indexes reference how to build the the output.

    Parameters
    ----------
    pipeline_description : Pipeline
        A pipeline description to be executed.
    """

    def __init__(self, pipeline_description: Pipeline) -> None:
        self.pipeline_description = pipeline_description
        n_steps = len(self.pipeline_description.steps)

        self.primitives_arguments = {}
        for i in range(0, n_steps):
            self.primitives_arguments[i] = {}

        self.execution_order = None

        self.pipeline = [None] * n_steps
        self.outputs = []

        # Getting the outputs
        for output in self.pipeline_description.outputs:
            origin = output['data'].split('.')[0]
            source = output['data'].split('.')[1]
            self.outputs.append((origin, int(source)))

        # Constructing DAG to determine the execution order
        execution_graph = nx.DiGraph()
        for i in range(0, n_steps):
            for argument, data in self.pipeline_description.steps[i].arguments.items():
                argument_edge = data['data']
                origin = argument_edge.split('.')[0]
                source = argument_edge.split('.')[1]

                self.primitives_arguments[i][argument] = {'origin': origin, 'source': int(source)}

                if origin == 'steps':
                    execution_graph.add_edge(str(source), str(i))
                else:
                    execution_graph.add_edge(origin, str(i))

        execution_order = list(nx.topological_sort(execution_graph))

        # Removing non-step inputs from the order
        execution_order = list(filter(lambda x: x.isdigit(), execution_order))
        self.execution_order = [int(x) for x in execution_order]

        # Creating set of steps to be call in produce
        self.produce_order = set()
        for output in self.pipeline_description.outputs:
            origin = output['data'].split('.')[0]
            source = output['data'].split('.')[1]
            if origin != 'steps':
                continue
            else:
                current_step = int(source)
                self.produce_order.add(current_step)
                for i in range(0, len(execution_order)):
                    step_origin = self.primitives_arguments[current_step]['inputs']['origin']
                    step_source = self.primitives_arguments[current_step]['inputs']['source']
                    if step_origin != 'steps':
                        break
                    else:
                        self.produce_order.add(step_source)
                        current_step = step_source
        
        # kyao!!!!
        self.produce_order = set(self.execution_order)
        self.fit_outputs = []
        self.produce_outputs = []
    
    def fit(self, **arguments) -> None:
        """
        Train all steps in the pipeline.

        Paramters
        ---------
        arguments
            Arguments required to train the Pipeline
        """

        primitives_outputs = [None] * len(self.execution_order)

        for i in range(0, len(self.execution_order)):
            primitive_arguments = {}
            n_step = self.execution_order[i]

            for argument, value in self.primitives_arguments[n_step].items():
                if value['origin'] == 'steps':
                    primitive_arguments[argument] = primitives_outputs[value['source']]
                else:
                    primitive_arguments[argument] = arguments[argument][value['source']]
            import pdb
            if isinstance(self.pipeline_description.steps[n_step], PrimitiveStep):
                # print('-'*100)
                primitive = self.pipeline_description.steps[n_step].primitive
                # print('step', n_step, 'primitive', primitive)
                #pdb.set_trace()
                primitives_outputs[n_step] = self._primitive_step_fit(n_step, self.pipeline_description.steps[n_step], primitive_arguments)
                #print("output of no",n_step," is:::")
                #print(primitives_outputs[n_step])

        # kyao!!!!
        self.fit_outputs = primitives_outputs

    def _primitive_step_fit(self, n_step: int, step: PrimitiveStep, primitive_arguments):
        """
        Execute a step and train it with primitive arguments.

        Paramters
        ---------
        n_step: int
            An integer of the actual step.
        step: PrimitiveStep
            A primitive step.
        primitive_arguments
            Arguments for set_training_data, fit, produce of the primitive for this step.

        """

        primitive = step.primitive

        primitive_hyperparams = primitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        custom_hyperparams = dict()

        # print("primitive hyperparams:", primitive_hyperparams)

        if bool(step.hyperparams):
            for hyperparam, value in step.hyperparams.items():
                if isinstance(value, dict):
                    custom_hyperparams[hyperparam] = value['data']
                else:
                    custom_hyperparams[hyperparam] = value

        # print("custom hyperparams:", custom_hyperparams)

        training_arguments_primitive = self._primitive_arguments(primitive, 'set_training_data')
        training_arguments = {}
        produce_params_primitive = self._primitive_arguments(primitive, 'produce')
        produce_params = {}

        for param, value in primitive_arguments.items():
            if param in produce_params_primitive:
                produce_params[param] = value
            if param in training_arguments_primitive:
                training_arguments[param] = value

        #FIXME: once hyperparameters work, simplify code below
        model = primitive(hyperparams=primitive_hyperparams(primitive_hyperparams.defaults()))
        try:
            model = primitive(hyperparams=primitive_hyperparams(
                        primitive_hyperparams.defaults(), **custom_hyperparams))
        except:
            print("******************\n[ERROR]Hyperparameters unsuccesfully set - using defaults")
            model = primitive(hyperparams=primitive_hyperparams(primitive_hyperparams.defaults()))

        #print('-'*100)
        # print('step', n_step, 'primitive', primitive)
        # print('training_arguments', training_arguments)
        model.set_training_data(**training_arguments)
        model.fit()
        self.pipeline[n_step] = model
        # print('produce_params', produce_params)
        return model.produce(**produce_params).value

    def _primitive_arguments(self, primitive, method: str) -> set:
        """
        Get the arguments of a primitive given a function.

        Paramters
        ---------
        primitive
            A primitive.
        method
            A method of the primitive.
        """
        return set(primitive.metadata.query()['primitive_code']['instance_methods'][method]['arguments'])

    def produce(self, **arguments):
        """
        Train all steps in the pipeline.

        Paramters
        ---------
        arguments
            Arguments required to execute the Pipeline
        """
        steps_outputs = [None] * len(self.execution_order)

        # print('-'*100)
        for i in range(0, len(self.execution_order)):
            n_step = self.execution_order[i]
            primitive = self.pipeline_description.steps[n_step].primitive
            # primitive = self.pipeline[i]
            produce_arguments_primitive = self._primitive_arguments(primitive, 'produce')
            produce_arguments = {}

            for argument, value in self.primitives_arguments[n_step].items():
                if argument in produce_arguments_primitive:
                    if value['origin'] == 'steps':
                        produce_arguments[argument] = steps_outputs[value['source']]
                    else:
                        produce_arguments[argument] = arguments[argument][value['source']]
                    if produce_arguments[argument] is None:
                        continue


            if isinstance(self.pipeline_description.steps[n_step], PrimitiveStep):
                if n_step in self.produce_order:
                    # print('step', n_step, 'primitive', primitive)
                    #import pdb
                    #pdb.set_trace()
                    steps_outputs[n_step] = self.pipeline[n_step].produce(**produce_arguments).value
                else:
                    steps_outputs[n_step] = None

        # print('-'*100)
        # kyao!!!!
        self.produce_outputs = steps_outputs

        # Create output
        pipeline_output = []
        for output in self.outputs:
            if output[0] == 'steps':
                pipeline_output.append(steps_outputs[output[1]])
            else:
                pipeline_output.append(arguments[output[0][output[1]]])
        return pipeline_output


def load_problem_doc(problem_doc_uri: str):
    """
    Load problem_doc from problem_doc_uri

    Paramters
    ---------
    problem_doc_uri
        Uri where the problemDoc.json is located
    """
    with open(problem_doc_uri) as file:
        problem_doc = json.load(file)
    problem_doc_metadata = Metadata(problem_doc)
    return problem_doc_metadata


def add_target_columns_metadata(dataset: 'Dataset', problem_doc_metadata: 'Metadata'):
    """
    Add metadata to the dataset from problem_doc_metadata

    Paramters
    ---------
    dataset
        Dataset
    problem_doc_metadata:
        Metadata about the problemDoc
    """
    for data in problem_doc_metadata.query(())['inputs']['data']:
        targets = data['targets']
        for target in targets:
            semantic_types = list(dataset.metadata.query(
                (target['resID'], metadata_base.ALL_ELEMENTS, target['colIndex'])).get('semantic_types', []))
            if 'https://metadata.datadrivendiscovery.org/types/Target' not in semantic_types:
                semantic_types.append('https://metadata.datadrivendiscovery.org/types/Target')
                dataset.metadata = dataset.metadata.update(
                    (target['resID'], metadata_base.ALL_ELEMENTS, target['colIndex']), {'semantic_types': semantic_types})
            if 'https://metadata.datadrivendiscovery.org/types/TrueTarget' not in semantic_types:
                semantic_types.append('https://metadata.datadrivendiscovery.org/types/TrueTarget')
                dataset.metadata = dataset.metadata.update(
                    (target['resID'], metadata_base.ALL_ELEMENTS, target['colIndex']), {'semantic_types': semantic_types})
    return dataset


def generate_pipeline(pipeline_uri: str, dataset_uri: str, problem_doc_uri: str):
    """
    Simplified interface that fit a pipeline with a dataset

    Paramters
    ---------
    pipeline_uri
        Uri to the pipeline description
    dataset_uri:
        Uri to the datasetDoc.json
    problem_doc_uri:
        Uri to the problemDoc.json
    """
    # Pipeline description
    pipeline_description = None
    if '.json' in pipeline_uri:
        with open(pipeline_uri) as pipeline_file:
            pipeline_description = Pipeline.from_json_content(string_or_file=pipeline_file)
    else:
        with open(pipeline_uri) as pipeline_file:
            pipeline_description = Pipeline.from_yaml_content(string_or_file=pipeline_file)

    # Problem Doc
    problem_doc = load_problem_doc(problem_doc_uri)

    # Dataset
    if 'file:' not in dataset_uri:
        dataset_uri = 'file://{dataset_uri}'.format(dataset_uri=os.path.abspath(dataset_uri))
    dataset = D3MDatasetLoader()
    dataset = dataset.load(dataset_uri=dataset_uri)
    # Adding Metadata to Dataset
    dataset = add_target_columns_metadata(dataset, problem_doc)

    # Pipeline
    pipeline_runtime = Runtime(pipeline_description)
    # Fitting Pipeline
    pipeline_runtime.fit(inputs=[dataset])
    return pipeline_runtime


def test_pipeline(pipeline_runtime: Runtime, dataset_uri: str):
    """
    Simplified interface test a pipeline with a dataset

    Paramters
    ---------
    pipeline_runtime
        Runtime object
    dataset_uri:
        Uri to the datasetDoc.json
    """
    # Dataset
    if 'file:' not in dataset_uri:
        dataset_uri = 'file://{dataset_uri}'.format(dataset_uri=os.path.abspath(dataset_uri))
    dataset = D3MDatasetLoader()
    dataset = dataset.load(dataset_uri=dataset_uri)

    return pipeline_runtime.produce(inputs=[dataset])


# # TESTING
# # BBN Pipeline

# # Simplified interfaces
# from runtime import generate_pipeline, test_pipeline
# pipeline_uri = 'bbn_pipe_v3.json'
# dataset_uri = '../../datasets/seed_datasets_current/31_urbansound/31_urbansound_dataset/datasetDoc.json'
# problem_doc_uri = '../../datasets/seed_datasets_current/31_urbansound/31_urbansound_problem/problemDoc.json'
# # Fit pipeline
# pipeline_runtime = generate_pipeline(pipeline_uri=pipeline_uri, dataset_uri=dataset_uri, problem_doc_uri=problem_doc_uri)
# # Testing
# path_test = '../../datasets/seed_datasets_current/31_urbansound/TEST/dataset_TEST/datasetDoc.json'
# results = test_pipeline(pipeline_runtime, path_test)
