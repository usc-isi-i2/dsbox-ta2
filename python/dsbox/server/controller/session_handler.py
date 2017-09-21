import os
import uuid
import shutil
import tempfile
import hashlib

class Session:
    sessions = {}

    def __init__(self):
        self.id = str(uuid.uuid4())
        self.controller = None
        self.pipelines = {}
        self.outputdir = tempfile.gettempdir() + os.sep + "dsbox-ta2" + os.sep + self.id
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
