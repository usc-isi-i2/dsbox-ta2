class PlannerEventHandler(object):
    pass

    def StartedPlanning(self):
        pass

    def SubmittedPipeline(self, pipeline):
        pass

    def RunningPipeline(self, pipeline):
        pass

    def CompletedPipeline(self, pipeline, result):
        pass

    def StartExecutingPipeline(self, pipeline):
        pass

    def ExecutedPipeline(self, pipeline, result):
        pass

    def EndedPlanning(self):
        pass
