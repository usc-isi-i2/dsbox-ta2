'''Hyperparameter tunning'''

from dsbox.planner.common.pipeline import Pipeline


class RandomHyperparamTuning(object):
    '''Generate random hyperparameters'''

    def generate_new_pipelines(self, pipeline: Pipeline, num_pipelines=1):
        new_pipelines = []
        for i in range(num_pipelines):
            new_pipe = pipeline.clone()
            new_pipe.planner_result = None
            new_pipe.test_result = None
            new_pipe.finished = False
            for primitive in new_pipe.primitives:
                if primitive.task == "Modeling" and primitive.hasHyperparamClass():
                    hyperparams_class = primitive.getHyperparamClass()
                    primitive.setHyperparams(hyperparams_class.sample())
            new_pipelines.append(new_pipe)
        print('RandomHyperparamTuning: {}'.format(new_pipelines))
        return new_pipelines
