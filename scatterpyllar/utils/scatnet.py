from octave_wrapper import OctaveCommand
import os
from scipy.io import loadmat
scatnet_path = os.environ.get("SCATNET_PATH", None)


def get_args_out(fname, *args):
    f = loadmat(fname, squeeze_me=True)
    return dict([(arg, f[arg]) for arg in args])


def octave_build_struct(struct_name, struct_content):
    command = "\n".join(["%(sn)s.%(k)s = %(v)s;" %
                         dict(sn=struct_name, k=k, v=str(v))
        for k, v in struct_content.items()])
    return command


def option_preproc(args, options=None):
    if options is None:
        options = dict()

    arguments = map(str, args) + ["options"]
    prep_code = octave_build_struct("options", options)

    return arguments, prep_code


class ScatNetFunction(object):

    def __init__(self, function_name, nargs_out=1,
                 pathvar=None,
                 postprocessing=None,
                 preprocessing=None):
        self.function_name = function_name
        self.nargs_out = nargs_out
        self.postprocessing = postprocessing
        self.preprocessing = preprocessing
        self.pathvar = pathvar

    def __call__(self, *args, **kwargs):
        arguments, prep_code = self.preprocessing(args, kwargs) if \
            self.preprocessing is not None else (args, "")

        args_out = ["arg%d" % i for i in range(self.nargs_out)]

        def post_proc(fname):
            return get_args_out(fname, *args_out)

        code = ("addpath_scatnet;\n" +
                prep_code + "\n" +
                ", ".join(args_out) +
                " = " +
                "%s(%s)" % (self.function_name, ", ".join(arguments)))

        oc = OctaveCommand(octave_code=code,
                           postprocessing=post_proc,
                           pathvar=self.pathvar or scatnet_path
            )
        self.oc = oc
        return oc()


def morlet_filter_bank_2d(shape, J=4, L=8, Q=1,
                          sigma_phi=0.8,
                          sigma_psi=0.8,
                          xi_psi=None,
                          slant_psi=None,
                          verbose=False):
    def preproc(args, kwargs):
        return option_preproc([args[0]], options=args[1])

    snf = ScatNetFunction(function_name="morlet_filter_bank_2d",
                          preprocessing=preproc)

    options = dict(J=J, L=L, Q=Q,
                   sigma_phi=sigma_phi,
                   sigma_psi=sigma_psi)
    if xi_psi is not None:
        options['xi_psi'] = xi_psi
    if slant_psi is not None:
        options['slant_psi'] = slant_psi

    result = snf(shape, options)

    if verbose:
        print snf.oc.output

    result = dict([(k, result['arg0'][k].item())
                   for k in ['phi', 'psi', 'meta']])
    result['psi'] = dict([(k, result['psi'][k].item())
                          for k in ['filter', 'meta']])
    result['psi']['filter'] = [dict([(n, fil[n]) for n in fil.dtype.names])
                               for fil in result['psi']['filter']]

    return result

