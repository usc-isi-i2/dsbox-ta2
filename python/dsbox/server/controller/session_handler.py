import os
import uuid
import json
import shutil
import tempfile
import hashlib

class Session:
    sessions = {}

    def __init__(self):
        self.id = str(uuid.uuid4())
        self.controller = None
        self.pipelines = {}
        self.test_results = {}
        self.planner_results = {}
        self.outputdir = tempfile.gettempdir() + os.sep + "dsbox-ta2" + os.sep + self.id
        self.config = self.create_config_from_env('ENV_D3M_CONFIG_FILEPATH')
        #print(self.outputdir)

    def get_pipeline(self, pipelineid):
        return self.pipelines.get(pipelineid, None)

    def add_pipeline(self, pipeline):
        self.pipelines[pipeline.id] = pipeline
        return pipeline.id

    def update_pipeline(self, pipeline):
        self.pipelines[pipeline.id] = pipeline

    def delete_pipeline(self, pipelineid):
        self.pipelines.pop(pipelineid, None)

    def save_prediction_file(self, result):
        resultfile = "%s%s%s.csv" % (self.outputdir, os.sep, str(uuid.uuid4()))
        result.predictions.to_csv(resultfile, index_label=result.predictions.index.name)
        return resultfile

    def cache_planner_result(self, pipeline, result):
        self.planner_results[pipeline.id] = result

    def cache_test_result(self, pipeline, result):
        self.test_results[pipeline.id] = result

    def create_config_from_env(self, env):
        path = os.environ.get(env)
        if path is not None:
            config = {}
            with open(path) as conf_data:
                config = json.load(conf_data)
                conf_data.close()
            return config
        return None

    @staticmethod
    def new():
        session = Session()
        Session.sessions[session.id] = session
        return session

    @staticmethod
    def get(id):
        return Session.sessions.get(id, None)

    @staticmethod
    def delete(sessionid):
        session = Session.get(sessionid)
        if session is not None:
            Session.sessions.pop(session.id, None)
            shutil.rmtree(session.outputdir)
