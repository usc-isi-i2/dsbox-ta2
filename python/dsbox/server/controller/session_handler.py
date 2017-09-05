import uuid
import shutil
import tempfile
import hashlib

class Session:
    sessions = {}

    def __init__(self):
        self.id = str(uuid.uuid1())
        self.controller = None
        self.pipelines = {}
        self.outputdir = tempfile.gettempdir() + "/dsbox-ta2/" + self.id
        #print(self.outputdir)

    def get_pipeline_id(self, pipeline, key):
        m = hashlib.md5()
        m.update((str(pipeline)+":"+key).encode('utf-8'))
        pipelineid = str(m.hexdigest())
        return pipelineid

    def get_pipeline(self, pipelineid):
        return self.pipelines.get(pipelineid, None)

    def add_pipeline(self, pipeline, key):
        pipelineid = self.get_pipeline_id(pipeline, key)
        self.pipelines[pipelineid] = pipeline
        return pipelineid

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
