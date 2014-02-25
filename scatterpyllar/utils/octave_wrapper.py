import os
import tempfile
import commands
from scipy.io import loadmat


def octave_code_to_raw_octave_code(octave_code):

    persistence_line = 'save -v7 %(__tmpfile__)s'
    code = ";".join(octave_code.split(";") +
                                [persistence_line])
    return code


def default_postprocessing(tmpfile):
    result = loadmat(tmpfile, squeeze_me=True)
    return result


class OctaveCommand(object):

    def __init__(self, octave_code=None, postprocessing=None,
                 raw_octave_code=None, pathvar=None):
        self.octave_code = octave_code
        self.postprocessing = postprocessing
        self.raw_octave_code = raw_octave_code
        self.pathvar = pathvar

    def __call__(self, **kwargs):

        raw_code = self.raw_octave_code or \
            octave_code_to_raw_octave_code(self.octave_code)

        command = 'octave -f --no-window-system '
        if self.pathvar:
            command = command + '--path %s ' % self.pathvar
        command = command + '--eval "' + raw_code + '"'

        fd, fname = tempfile.mkstemp(suffix='.mat')
        os.close(fd)

        kwargs["__tmpfile__"] = fname

        self.output = commands.getoutput(command % kwargs)

        if self.postprocessing:
            result = self.postprocessing(fname)
        else:
            result = default_postprocessing(fname)

        os.remove(fname)
        return result


if __name__ == "__main__":
    oc = OctaveCommand(octave_code="%(var)s = 10")

    res = oc(var="my_y")

    print repr(res['my_y'])

    
