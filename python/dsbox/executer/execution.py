import pickle
import zerorpc

client = zerorpc.Client()
#client.connect("tcp://127.0.0.1:4242")
#client.connect("tcp://varuns-mbp.local:4242")
client.connect("tcp://192.168.0.105:4242")

class Execution():
    def execute(self, function_call, obj=None, args=None, kwargs=None, objreturn=False):
        return pickle.loads(client.execute(function_call, pickle.dumps(obj), pickle.dumps(args), pickle.dumps(kwargs), objreturn))
