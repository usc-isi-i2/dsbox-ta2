import logging
import copy
import typing
import os
import json
import d3m.exceptions as exceptions

from dsbox.pipeline.fitted_pipeline import FittedPipeline
from dsbox.pipeline.utils import larger_is_better
from dsbox.combinatorial_search.ConfigurationSpaceBaseSearch import calculate_score, SpecialMetric
from dsbox.combinatorial_search.search_utils import get_target_columns
from d3m.metadata.problem import parse_problem_description, TaskType
from d3m import runtime as runtime_module, container
from d3m.metadata import pipeline as pipeline_module
from d3m.metadata.base import ALL_ELEMENTS, Metadata
from d3m import index as d3m_index


class EnsembleTuningPipeline:
    """
    Class specially used to do ensemble tuning.

    Attributes
    ----------
    voting_pipeline: Pipeline
        A D3M Pipeline object for ensemble tuning (including subpipelines)
    fitted_pipeline: FittedPipeline
        A DSBOX FittedPipeline object for saving and fitting

    Parameters
    ----------
    pipeline_files_dir : str
        The path to fitted pipelines
    log_dir: str
        The path to log files
    pids : typing.List[str]
        The ids of candidate pipelines
        If it was not given, you should call 'generate_candidate_pids' to generate it
    train_dataset / test_dataset: Dataset
        The training dataset and testing dataset
    candidate_choose_method: str
        Method to use for choose candidate pids, can be not given
    report: dict
        The ensemble report for tuning (from ta2 system), if it is not given, you have to set pids by yourself
    problem: dict
        Problem description
    problem_doc_metadata: Metadata
        Problem doc in metadata format
    """
    def __init__(self, pipeline_files_dir: str, log_dir: str, 
                 train_dataset: container.Dataset, 
                 test_dataset: container.Dataset, 
                 pids: typing.List[str] = None,
                 candidate_choose_method: str = 'lastStep',
                 report = None, problem = None, 
                 problem_doc_metadata = None):

        self.pipeline_files_dir = pipeline_files_dir
        self.log_dir = log_dir
        self.pids = pids
        self.candidate_choose_method = candidate_choose_method
        self.report = report
        self.test_dataset = test_dataset
        self.train_dataset = train_dataset
        self.problem_doc_metadata = problem_doc_metadata

        self.problem = problem
        if problem:
            performance_metrics = problem['problem']['performance_metrics']
            self.performance_metrics = list(map(
                lambda d: {'metric': d['metric'].unparse(), 'params': d['params']},
                performance_metrics
                ))

            self.task_type = self.problem['problem']['task_type']
            self.dataset_id = self.problem['problem']['id']
        else:
            self.dataset_id = ""
        self._logger = logging.getLogger(__name__)
        # self.ensemble_dataset = ensemble_dataset

        self.voting_pipeline = None
        self.fitted_pipeline = None

    def generate_ensemble_pipeline(self):
        """
            Function used to generate the Pipeline for ensemble tuning
        """
        if not self.pids:
            raise ValueError("No candidate pipeline ids found, unable to generate the ensemble pipeline.")
        elif len(self.pids) == 1:
            raise ValueError("Only 1 candidate pipeline id found, unable to generate the ensemble pipeline.")

        step_outputs = []
        self.voting_pipeline = pipeline_module.Pipeline('voting', context=pipeline_module.PipelineContext.TESTING)
        pipeline_input = self.voting_pipeline.add_input(name='inputs')

        for each_pid in self.pids:
            each_dsbox_fitted, each_runtime = FittedPipeline.load(self.pipeline_files_dir, each_pid, self.log_dir)
            each_fitted = runtime_module.FittedPipeline(each_pid, each_runtime, context=pipeline_module.PipelineContext.TESTING)
            each_step = pipeline_module.FittedPipelineStep(each_fitted.id, each_fitted)
            each_step.add_input(pipeline_input)
            self.voting_pipeline.add_step(each_step)
            step_outputs.append(each_step.add_output('output'))

        concat_step = pipeline_module.PrimitiveStep({
            "python_path": "d3m.primitives.dsbox.VerticalConcat",
            "id": "dsbox-vertical-concat",
            "version": "1.3.0",
            "name": "DSBox vertically concat"})

        for i in range(len(self.pids) - 1):
            each_concact_step = copy.deepcopy(concat_step)
            if i == 0:
                each_concact_step.add_argument(name='inputs1', argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=step_outputs[i])
            else:
                each_concact_step.add_argument(name='inputs1', argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=concat_step_output)
            each_concact_step.add_argument(name='inputs2', argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=step_outputs[i+1])


            self.voting_pipeline.add_step(each_concact_step)
            # update concat_step_output
            concat_step_output = each_concact_step.add_output('produce')

        vote_step = pipeline_module.PrimitiveStep({
            "python_path": "d3m.primitives.dsbox.EnsembleVoting",
            "id": "dsbox-ensemble-voting",
            "version": "1.3.0",
            "name": "DSBox ensemble voting"})

        vote_step.add_argument(name='inputs', argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=concat_step_output)
        self.voting_pipeline.add_step(vote_step)
        voting_output = vote_step.add_output('produce')

        self.voting_pipeline.add_output(name='Metafeatures', data_reference=voting_output)
        self._logger.info("Ensemble pipeline created successfully")

    def fit_and_produce(self):
        """
            Generate the FittedPipeline object 
            And run fit and produce steps to get the metric score of this ensemble pipeline
        """
        if not self.voting_pipeline:
            raise ValueError("No voting pipeline found, please run generate_ensemble_pipeline first")

        self.fitted_pipeline = FittedPipeline(pipeline = self.voting_pipeline, dataset_id = self.dataset_id, log_dir = self.log_dir, metric_descriptions = "pass")

        # if we are given performance metrics and task type, we can conduct prediction operations
        if self.performance_metrics and self.task_type:
            self._logger.info("Will calculate the metric scores")
            # In ensemble tuning, we should not use cache
            self.fitted_pipeline.runtime.set_not_use_cache()
            if self.test_dataset:
                self.fitted_pipeline.fit(inputs = [self.train_dataset])
                self.fitted_pipeline.produce(inputs = [self.test_dataset])

            prediction = self.fitted_pipeline.get_produce_step_output(0)
            ground_truth = get_target_columns(self.test_dataset, self.problem_doc_metadata)
            score_metric = calculate_score(ground_truth, prediction, self.performance_metrics, self.task_type, SpecialMetric().regression_metric)

            if type(score_metric) is list:
                    score_metric = score_metric[0]
            self.fitted_pipeline.set_metric(score_metric)

        if self.problem:
            self.fitted_pipeline.problem = self.problem_doc_metadata
        self._logger.info("Ensemble pipeline fitted and produced successfully")

    def save(self):
        '''
            use fitted_pipeline to save this ensemble voting pipeline
        '''
        if not self.fitted_pipeline:
            raise ValueError("Ensemble voting pipeline must be fitted before saving.")
        else:
            self.fitted_pipeline.save(folder_loc = self.pipeline_files_dir)
        self._logger.info("Save ensemble pipeline successfully")

    def generate_candidate_pids(self) -> None:
        if self.pids:
            self._logger.warning("There already exist candidate pipeline ids")

        # TODO: add ability to deal with the condition when there are multiple prediction columns
        if not self.report:
            raise ValueError("No ensemble tuning report found, unable to generate candidate pipeline ids")

        # way 1: check the model step of pipeline, only choose the pipelines with different model step
        if self.candidate_choose_method == 'lastStep':
            memo = {}
            all_predictions = {}
            all_predictions_id = {}
            '''
                These 3 dictionary saves the correspondinng best pipelines of each model method
                The key is the model step's name, e.g.: "d3m.primitives.sklearn_wrap.SKSGDClassifier"
                memo: save the test metric scores
                all_predicionts: save the detail prediction results on ensemble_dataset
                all_predicionts_id: save the pipeline id of the best pipelines
            '''
            for key, value in self.report['report']['ensemble_dataset_predictions'].items():
                pipeline_description = value['pipeline']
                each_prediction = value['ensemble_tuning_result']
                # each_prediction = self.add_d3m_index_and_prediction_class_name(value['ensemble_tuning_result'], self.ensemble_dataset)
                if "confidence" not in each_prediction.columns:
                    each_prediction['confidence'] = 1.0

                if 'model_step' in pipeline_description:
                    model_step_name = pipeline_description['model_step']['primitive']
                    print(model_step_name)
                    # if not first time see,choose the pipelines with better test matrics scores
                    if model_step_name in memo:
                        if larger_is_better(value['ensemble_tuning_metrics'][0]['metric']):
                            if value['ensemble_tuning_metrics'][0]['value'] > memo[model_step_name]['value']:
                                memo[model_step_name] = value['ensemble_tuning_metrics'][0]
                                all_predictions[model_step_name] = each_prediction
                                all_predictions_id[model_step_name] = key
                        else:
                            if value['ensemble_tuning_metrics'][0]['value'] < memo[model_step_name]['value']:
                                memo[model_step_name] = value['ensemble_tuning_metrics'][0]
                                all_predictions[model_step_name] = each_prediction
                                all_predictions_id[model_step_name] = key
                    else:  # if first time see, add to memo directly
                        memo[model_step_name] = value['ensemble_tuning_metrics'][0]
                        all_predictions[model_step_name] = each_prediction
                        all_predictions_id[model_step_name] = key
                else:
                    self._logger.error("No model step found for pipeline" + key)
            # finally get a list of pids for ensemble tuning pipelines
            self.pids = list(all_predictions_id.values())

        # way 2: check the similarity of each prediction results, only choose the low similarity predictions
        elif self.candidate_choose_method == 'resultSimilarity':
            # TODO: add a method to check the similarity of the predictions
            ensemble_predicts = self.report['report']['ensemble_dataset_predictions']
            pipeline_ids = list(ensemble_predicts.keys())
            pipelines_count = len(pipeline_ids)

            target_len_ensemble_pipeline_pids = 5
            if pipelines_count < target_len_ensemble_pipeline_pids:
                target_len_ensemble_pipeline_pids = pipelines_count

            similarity_matrix = {}
            # calculate the similarity of each predictions

            for i in range(pipelines_count):
                for j in range(i + 1, pipelines_count):
                    temp1 = ensemble_predicts[pipeline_ids[i]]['ensemble_tuning_result']
                    temp2 = ensemble_predicts[pipeline_ids[j]]['ensemble_tuning_result']

                    if self.task_type == TaskType.CLASSIFICATION:
                        # if classification problem, use accuracy instead of f1marco
                        temp_metric = copy.deepcopy(self.performance_metrics)
                        temp_metric[0]['metric'] = 'accuracy'
                        temp_score = calculate_score(temp1, temp2, temp_metric, self.task_type, SpecialMetric().regression_metric)
                    elif self.task_type == TaskType.REGRESSION:
                        # if regression problem, use MSE instead of f1marco
                        temp_metric = copy.deepcopy(self.performance_metrics)
                        temp_metric[0]['metric'] = 'meanSquaredError'
                        temp_score = calculate_score(temp1, temp2, temp_metric, self.task_type, SpecialMetric().regression_metric)

                    similarity_matrix[(i,j)] = temp_score[0]['value']

            similarity_matrix_list = []
            for k, v in similarity_matrix.items():
                similarity_matrix_list.append([v,k])
            similarity_matrix_list = sorted(similarity_matrix_list,key=lambda x: x[0])
            similarity_matrix_list.reverse()

            self.pids = []
            while len(self.pids) < target_len_ensemble_pipeline_pids:
                temp = similarity_matrix_list.pop()
                for each in temp[1]:
                    pid_each = pipeline_ids[each]
                    if pid_each not in self.pids:
                        self.pids.append(pid_each)

class HorizontalTuningPipeline(EnsembleTuningPipeline):
    def __init__(self, pipeline_files_dir: str, log_dir: str, 
                 train_dataset: container.Dataset, 
                 test_dataset: container.Dataset, 
                 pids: typing.List[str] = None,
                 candidate_choose_method: str = 'lastStep',
                 report = None, problem = None, 
                 problem_doc_metadata = None, 
                 final_step_primitive: str = "d3m.primitives.sklearn_wrap.SKBernoulliNB"):
        super().__init__(pipeline_files_dir, log_dir, 
                 train_dataset, test_dataset, 
                 pids, candidate_choose_method, report, 
                 problem, problem_doc_metadata)
        self.final_step_primitive = final_step_primitive

    def generate_ensemble_pipeline(self):
        if not self.pids:
            raise ValueError("No candidate pipeline ids found, unable to generate the ensemble pipeline.")
        elif len(self.pids) == 1:
            raise ValueError("Only 1 candidate pipeline id found, unable to generate the ensemble pipeline.")        
        step_outputs = []
        self.big_pipeline, pipeline_output, pipeline_input, target = self.preprocessing_pipeline()
        for each_pid in self.pids:
            each_dsbox_fitted, each_runtime = FittedPipeline.load(self.pipeline_files_dir, each_pid, self.log_dir)
            each_fitted = runtime_module.FittedPipeline(each_pid, each_runtime, context=pipeline_module.PipelineContext.TESTING)
            each_step = pipeline_module.FittedPipelineStep(each_fitted.id, each_fitted)
            each_step.add_input(pipeline_input)
            self.big_pipeline.add_step(each_step)
            step_outputs.append(each_step.add_output('output'))

        concat_step = pipeline_module.PrimitiveStep({
            "python_path": "d3m.primitives.dsbox.HorizontalConcat", 
            "id": "dsbox-horizontal-concat", 
            "version": "1.3.0",
            "name": "DSBox horizontal concat"
            })
        for i in range(len(self.pids) - 1):
            each_concact_step = copy.deepcopy(concat_step)
            if i == 0:
                each_concact_step.add_argument(name='inputs1', argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=step_outputs[i])
            else:
                each_concact_step.add_argument(name='inputs1', argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=concat_step_output)
            each_concact_step.add_argument(name='inputs2', argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=step_outputs[i+1])
            each_concact_step.add_hyperparameter(name="column_name", argument_type=pipeline_module.ArgumentType.VALUE, data=i)

            self.big_pipeline.add_step(each_concact_step)
            # update concat_step_output
            concat_step_output = each_concact_step.add_output('produce')

        encode_res_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.dsbox.Encoder").metadata.query()))
        encode_res_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=concat_step_output)
        self.big_pipeline.add_step(encode_res_step)
        encode_res_step_output = encode_res_step.add_output("produce")

        concat_step1 = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.data.HorizontalConcat").metadata.query()))
        concat_step1.add_argument(name="left", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=encode_res_step_output)
        concat_step1.add_argument(name="right", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=pipeline_output)
        concat_step1.add_hyperparameter(name="use_index", argument_type=pipeline_module.ArgumentType.VALUE, data=False)
        self.big_pipeline.add_step(concat_step1)
        concat_output1 = concat_step1.add_output("produce")

        model_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive(self.final_step_primitive).metadata.query()))
        model_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=concat_output1)
        model_step.add_argument(name="outputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=target)
        self.big_pipeline.add_step(model_step)
        big_output = model_step.add_output("produce")
        final_output = self.big_pipeline.add_output(name="final", data_reference=big_output)
        self._logger.info("Ensemble pipeline created successfully")

    def generate_candidate_pids(self) -> None: # select top 3 from the pipeline directory
        if self.pids:
            self._logger.warn("There already exist candidate pipeline ids")
            return

        pipeline_dir = os.path.join(self.pipeline_files_dir, "pipelines")
        pip_candidate = {}
        for p_name in os.listdir(pipeline_dir):
            with open(pipeline_dir+"/"+p_name, "r") as f:
                valid = True
                if p_name != ".DS_Store":
                    data = json.load(f)
                else:
                    continue
                if "pipeline_rank" in data:
                    for s in data["steps"]:
                        if s["type"] == "SUBPIPELINE":
                            valid = False
                            break
                    if valid:
                        pip_candidate[data["id"]] = data["pipeline_rank"]
        candidate = list(pip_candidate.keys())
        candidate.sort(key=lambda x: pip_candidate[x])
        self.pids = candidate[:3]
        return


    def fit_and_produce(self):
        self.fitted_pipeline = FittedPipeline(pipeline=self.big_pipeline, dataset_id=self.dataset_id,
                                              log_dir=self.log_dir, metric_descriptions="pass")
        if self.performance_metrics and self.task_type:
            self._logger.info("Will calculate the metric scores")
            # In ensemble tuning, we should not use cache
            self.fitted_pipeline.runtime.set_not_use_cache()
            if self.test_dataset:
                self.fitted_pipeline.fit(inputs=[self.train_dataset])
                self.fitted_pipeline.produce(inputs=[self.test_dataset])

            prediction = self.fitted_pipeline.get_produce_step_output(0)
            ground_truth = get_target_columns(self.test_dataset, self.problem_doc_metadata)
            score_metric = calculate_score(ground_truth, prediction, self.performance_metrics, self.task_type, SpecialMetric().regression_metric)

            if type(score_metric) is list:
                    score_metric = score_metric[0]
            self.fitted_pipeline.set_metric(score_metric)

        if self.problem:
            self.fitted_pipeline.problem = self.problem_doc_metadata
        self._logger.info("Ensemble pipeline fitted and produced successfully")


    def preprocessing_pipeline(self):
        preprocessing_pipeline = pipeline_module.Pipeline('big', context=pipeline_module.PipelineContext.TESTING)
        initial_input = preprocessing_pipeline.add_input(name="inputs")
        denormalize_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.dsbox.Denormalize").metadata.query()))
        denormalize_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=initial_input)
        preprocessing_pipeline.add_step(denormalize_step)
        denormalize_step_output = denormalize_step.add_output('produce')
        to_dataframe_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.datasets.DatasetToDataFrame").metadata.query()))
        to_dataframe_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=denormalize_step_output)
        preprocessing_pipeline.add_step(to_dataframe_step)
        to_dataframe_step_output = to_dataframe_step.add_output("produce")
        extract_attribute_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.data.ExtractColumnsBySemanticTypes").metadata.query()))
        extract_attribute_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=to_dataframe_step_output)
        preprocessing_pipeline.add_step(extract_attribute_step)
        extract_attribute_step_output = extract_attribute_step.add_output("produce")
        extract_attribute_step.add_hyperparameter(name='semantic_types',argument_type=pipeline_module.ArgumentType.VALUE, data=(
                                        'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                        'https://metadata.datadrivendiscovery.org/types/Attribute',
                                        ))
        profiler_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.dsbox.Profiler").metadata.query()))
        profiler_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=extract_attribute_step_output)
        preprocessing_pipeline.add_step(profiler_step)
        profiler_step_output = profiler_step.add_output("produce")
        clean_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.dsbox.CleaningFeaturizer").metadata.query()))
        clean_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=profiler_step_output)
        preprocessing_pipeline.add_step(clean_step)
        clean_step_output = clean_step.add_output("produce")
        corex_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.dsbox.CorexText").metadata.query()))
        corex_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=clean_step_output)
        preprocessing_pipeline.add_step(corex_step)
        corex_step_output = corex_step.add_output("produce")
        encoder_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.dsbox.Encoder").metadata.query()))
        encoder_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=corex_step_output)
        preprocessing_pipeline.add_step(encoder_step)
        encoder_step_output = encoder_step.add_output("produce")
        impute_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.dsbox.MeanImputation").metadata.query()))
        impute_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=encoder_step_output)
        preprocessing_pipeline.add_step(impute_step)
        impute_step_output = impute_step.add_output("produce")
        scalar_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.dsbox.IQRScaler").metadata.query()))
        scalar_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=impute_step_output)
        preprocessing_pipeline.add_step(scalar_step)
        scalar_step_output = scalar_step.add_output("produce")
        extract_target_step = pipeline_module.PrimitiveStep(dict(d3m_index.get_primitive("d3m.primitives.data.ExtractColumnsBySemanticTypes").metadata.query()))
        extract_target_step.add_argument(name="inputs", argument_type=pipeline_module.ArgumentType.CONTAINER, data_reference=to_dataframe_step_output)
        preprocessing_pipeline.add_step(extract_target_step)
        extract_target_step_output = extract_target_step.add_output("produce")
        extract_target_step.add_hyperparameter(name='semantic_types',argument_type=pipeline_module.ArgumentType.VALUE, data=(
                                                  'https://metadata.datadrivendiscovery.org/types/Target',
                                                  'https://metadata.datadrivendiscovery.org/types/TrueTarget'
                                                  ))
        # preprocessing_pipeline.add_output(name="produce", data_reference=scalar_step_output)
        return preprocessing_pipeline, scalar_step_output, initial_input, extract_target_step_output


def set_target_column(dataset):
    """
        Function used for unit test
    """
    for index in range(
            dataset.metadata.query(('0', ALL_ELEMENTS))['dimension']['length'] - 1,
            -1, -1):
        column_semantic_types = dataset.metadata.query(
            ('0', ALL_ELEMENTS, index))['semantic_types']
        if 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget' in \
                column_semantic_types:
            column_semantic_types = list(column_semantic_types) + [
                'https://metadata.datadrivendiscovery.org/types/Target',
                'https://metadata.datadrivendiscovery.org/types/TrueTarget']
            dataset.metadata = dataset.metadata.update(
                ('0', ALL_ELEMENTS, index), {'semantic_types': column_semantic_types})
            return

    raise exceptions.InvalidArgumentValueError(
        'At least one column should have semantic type SuggestedTarget')


# unit test part
# sys.path.append('/Users/minazuki/Desktop/studies/master/2018Summer/DSBOX_new/dsbox-ta2/python')
if __name__ == "__main__":
    # data_dir = '/Users/minazuki/Desktop/studies/master/2018Summer/data'
    # log_dir = '/Users/minazuki/Desktop/studies/master/2018Summer/data/log'
    # pids = ['3c5f6bfa-4d3b-43b4-a371-af7be9e2a938','bcdab3e5-eb82-438c-83cb-d7f851754536',
    #         'c0b4d68e-16ca-4042-9d51-8df706a7ecf1','7370ab30-c7cf-461e-a94b-f7a50ebbaaf0']

    # dataset = container.Dataset.load('file:///Users/minazuki/Desktop/studies/master/2018Summer/data/datasets/seed_datasets_current/38_sick/38_sick_dataset/datasetDoc.json')
    # set_target_column(dataset)

    # problem_doc_path = os.path.abspath('/Users/minazuki/Desktop/studies/master/2018Summer/data/datasets/seed_datasets_current/38_sick/38_sick_problem/problemDoc.json')

    # problem = parse_problem_description(problem_doc_path)
    # choose_method = 'lastStep'
    # with open(problem_doc_path) as file:
    #     problem_doc = json.load(file)
    # problem_doc_metadata = Metadata(problem_doc)

    # pp = EnsembleTuningPipeline(pipeline_files_dir = data_dir, log_dir = log_dir,
    #              pids = pids, candidate_choose_method = choose_method, report = None, problem = problem, 
    #              test_dataset = dataset, train_dataset = dataset, problem_doc_metadata = problem_doc_metadata)
    # pp.generate_ensemble_pipeline()
    # pp.fit_and_produce()
    # pp.save()
    data_dir = "/Users/muxin/Desktop/ISI/dsbox-env/output/seed/38_sick/"
    log_dir = '/Users/muxin/Desktop/studies/master/2018Summer/data/log'
    pids = ['32b24d72-44c6-4956-bc21-835cb42f0f2e', 'a8f4001a-64f4-4ff1-a89d-3548f4dfeb88', '5e1d9723-ec02-46d2-abdf-46389fba8e52']
    dataset = container.Dataset.load('file:///Users/muxin/Desktop/ISI/dsbox-env/data/datasets/seed_datasets_current/38_sick/38_sick_dataset/datasetDoc.json')
    set_target_column(dataset)
    problem_doc_path = os.path.abspath("/Users/muxin/Desktop/ISI/dsbox-env/data/datasets/seed_datasets_current/38_sick/38_sick_problem/problemDoc.json")
    problem = parse_problem_description(problem_doc_path)
    with open(problem_doc_path) as file:
        problem_doc = json.load(file)
    problem_doc_metadata = Metadata(problem_doc)
    qq = HorizontalTuningPipeline(pipeline_files_dir=data_dir, log_dir=log_dir,
                                  pids=None, problem=problem, train_dataset=dataset,
                                  test_dataset=dataset, problem_doc_metadata=problem_doc_metadata
                                 )
    qq.generate_candidate_pids()
    print(qq.pids)
    qq.generate_ensemble_pipeline()
    qq.fit_and_produce() 
    print(qq.fitted_pipeline.get_produce_step_output(0))
    qq.save()




