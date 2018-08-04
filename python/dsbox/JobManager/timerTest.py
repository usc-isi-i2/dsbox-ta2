from threading import Timer
import os
import time
import traceback



class foo:
    def __init__(self):
        self.a = 100

    def long_running_job(self):
        print("[long_running_job] long_running_job started :",self.a)
        print("[long_running_job] pid:", os.getpid())
        time.sleep(self.a)

    def kill_me(self):
        print("[kill_me] Killing the process")
        os.kill(os.getpid(), 9)

    def hello(self):
        print("[hello] hello, world")
        print("[hello] pid:", os.getpid())


        t = Timer(10, self.kill_me)
        t.start()

        print("[hello] pid:", os.getpid())
        self.long_running_job()


pid = os.fork()
if pid == 0:
    try:
        o = foo()
        o.hello()
    except:
        traceback.print_exc()
else:
    print("[MAIN] pid:", os.getpid())
    os.wait()
print("[INFO] I am still alive")
