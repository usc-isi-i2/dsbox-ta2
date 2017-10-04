class PlannerEventHandler(object):
    pass

    def ProblemNotImplemented(self):
        return False

    def StartedPlanning(self):
        return True

    def SubmittedPipeline(self, pipeline):
        return True

    def RunningPipeline(self, pipeline):
        return True

    def CompletedPipeline(self, pipeline, result):
        return True

    def StartExecutingPipeline(self, pipeline):
        return True

    def ExecutedPipeline(self, pipeline, result):
        return True

    def EndedPlanning(self):
        return True
