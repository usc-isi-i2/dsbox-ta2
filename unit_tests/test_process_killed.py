import unittest
import subprocess


class TestProcessKilled(unittest.TestCase):

    def setUp(self):
        self.ps_command = subprocess.Popen("ps -ax | grep ta2-search", shell=True, stdout=subprocess.PIPE)

    def test_process_killed(self):
        ps_output = self.ps_command.stdout.read().decode("utf-8")
        command_out = [x for x in ps_output.split("\n") if x and "grep ta2-search" not in x]
        self.assertEqual(command_out, [], "child process not killed or parent process not exit")
