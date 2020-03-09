import os
import sys


class Logger:
    def __init__(self, logfpath):
        self._file = open(logfpath, 'w')

    def write(self, s):
        print(s)
        self._file.write(s)


class Tee:
    """
    copied from https://stackoverflow.com/questions/616645/how-do-i-duplicate-sys-stdout-to-a-log-file-in-python
    """
    def __init__(self, logfpath, overwrite=False):
        self._terminal = sys.stdout
        self._file = None

        if os.path.isfile(logfpath):
            if overwrite:
                os.remove(logfpath)
            else:
                raise IOError('log file exists: %s' % logfpath)
        self._file = open(logfpath, 'a')
        sys.stdout = self

    def stop(self):
        self.flush()
        sys.stdout = self._terminal
        if self._file is not None:
            self._file.close()

    def write(self, s):
        self._file.write(s)
        self._terminal.write(s)

    def flush(self):
        self._file.flush()
        # self._terminal.flush()
