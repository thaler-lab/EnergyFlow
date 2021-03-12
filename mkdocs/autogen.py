import inspect
import os
import re

import energyflow as ef
import energyflow.archs as archs

template_dir = 'sources/templates'
example_dir = '../examples'
example_files = ef.utils.data_utils.ALL_EXAMPLES

template_dict = {
    'current_version': ef.__version__,
    'num_example_files': len(example_files)
}

DEFAULT_CLASSLEVEL = 2
DEFAULT_FUNCLEVEL = 3

docs = {

    'measures': [
        {
            'module': ef.measure,
            'classes': [
                {
                    'name': 'Measure',
                    'methods': ['evaluate'],
                }
            ]
        }
    ],

    'gen': [
        {
            'module': ef.gen,
            'classes': [
                {
                    'name': 'Generator',
                    'methods': ['save'], 
                    'properties': ['specs']
                }
            ]
        }
    ],

    'efp': [
        {
            'module': ef.efp,
            'classes': [
                {
                    'name': 'EFP',
                    'methods': ['compute', 'batch_compute'],
                    'properties': ['graph', 'simple_graph', 'weights', 'weight_set', 'einstr', 'einpath',
                                   'efm_spec', 'efm_einstr', 'efm_einpath', 'efmset', 'np_optimize',
                                   'n', 'e', 'd', 'v', 'k', 'c', 'p', 'h', 'spec', 'ndk']
                },
                {
                    'name': 'EFPSet',
                    'methods': ['compute', 'batch_compute', 'calc_disc', 'sel', 'csel',
                                'count', 'graphs', 'simple_graphs'],
                    'properties': ['efps', 'efmset', 'specs', 'cspecs', 'weight_set', 'cols']
                }
            ]
        }
    ],

    'efm': [
        {
            'module': ef.efm,
            'functions': ['efp2efms'],
            'classlevel': 3,
            'classes': [
                {
                    'name': 'EFM',
                    'methods': ['compute', 'batch_compute'],
                    'properties': ['nup', 'nlow', 'spec', 'v']
                },
                {
                    'name': 'EFMSet',
                    'methods': ['compute', 'batch_compute'],
                    'properties': ['efms', 'rules']
                }
            ]
        }
    ],

    'utils': [
        {
            'module': ef.utils.particle_utils,
            'functions': ef.utils.particle_utils.__all__
        },
        {
            'module': ef.utils.data_utils,
            'modpath': 'energyflow.utils.',
            'functions': ef.utils.data_utils.__all__
        },
        {
            'module': ef.utils.image_utils,
            'modpath': 'energyflow.utils.',
            'functions': ef.utils.image_utils.__all__
        },
        {
            'module': ef.utils.fastjet_utils,
            'functions': ef.utils.fastjet_utils.__all__
        },
        {
            'module': ef.utils.random_utils,
            'functions': ef.utils.random_utils.__all__[:-1]
        },
    ],

    'archs': [
        {
            'module': archs.archbase,
            'modpath': 'energyflow.archs.archbase.',
            'classes': [
                {
                    'name': 'ArchBase',
                    'methods': ['fit', 'predict'],
                    'properties': ['model'],
                    'postdoc': inspect.getdoc(archs.archbase.NNBase._process_hps)
                }
            ]
        },
        {
            'module': archs,
            'modpath': 'energyflow.archs.',
            'classes': [
                {
                    'name': 'EFN',
                    'maindocfuncname': '_process_hps',
                    'methods': ['eval_filters'],
                    'properties': ['inputs', 'weights', 'Phi', 'latent', 'F', 'output', 'layers', 'tensors']
                },
                {
                    'name': 'PFN',
                    'maindocfuncname': '_construct_point_cloud_inputs',
                    'properties': ['inputs', 'weights', 'Phi', 'latent', 'F', 'output', 'layers', 'tensors']
                },
                {
                    'name': 'CNN',
                    'maindocfuncname': '_process_hps',
                },
                {
                    'name': 'DNN',
                    'maindocfuncname': '_process_hps',
                },
            ]
        },
        {
            'module': archs.linear,
            'modpath': 'energyflow.archs.',
            'classes': [
                {
                    'name': 'LinearClassifier',
                    'maindocfuncname': '_process_hps'
                }
            ]
        },
        {
            'module': archs.utils,
            'modpath': 'energyflow.archs.',
            'functions': archs.utils.__all__
        },
    ],

    'datasets': [
        {
            'module': ef.datasets.mod,
            'classlevel': 3,
            'functions': ['load', 'filter_particles', 'kfactors'],
            'modpath': 'energyflow.mod.',
            'classes': [
                {
                    'name': 'MODDataset',
                    'methods': ['sel', 'apply_mask', 'save', 'close'],
                    'properties': ['jets_i', 'jets_f', 'pfcs', 'gens', 'particles', 'filenames', 'hf']
                }
            ]
        },
        {
            'module': ef.datasets.zjets_delphes,
            'funclevel': 4,
            'functions': ef.datasets.zjets_delphes.__all__,
            'modpath': 'energyflow.zjets_delphes.'
        },
        {
            'module': ef.datasets.qg_jets,
            'funclevel': 4,
            'functions': ef.datasets.qg_jets.__all__,
            'modpath': 'energyflow.qg_jets.'
        },
        {
            'module': ef.datasets.qg_nsubs,
            'funclevel': 4,
            'functions': ef.datasets.qg_nsubs.__all__,
            'modpath': 'energyflow.qg_nsubs.'
        },
    ],

    'emd': [
        {
            'module': ef.emd,
            'functions': ['emd4doc', 'emds4doc'] + ef.emd.__all__[2:],
            'modpath': 'energyflow.emd.'
        }
    ],

    'obs': [
        {
            'module': ef.obs,
            'funclevel': 2,
            'functions': ['image_activity', 'zg', 'zg_from_pj'],
            'classes': [
                {
                    'name': 'D2',
                    'methods': ['compute', 'batch_compute'],
                    'properties': ['efpset']
                },
                {
                    'name': 'C2',
                    'methods': ['compute', 'batch_compute'],
                    'properties': ['efpset']
                },
                {
                    'name': 'C3',
                    'methods': ['compute', 'batch_compute'],
                    'properties': ['efpset']
                },
            ]
        }
    ]
}

def get_function_signature(fname, func, argskip=0, modpath=''):

    comments = inspect.getcomments(func)
    s = '```python\n'
    s += modpath
    if comments is None:
        sig = inspect.signature(func).parameters
        args = []
        for name, param in sig.items():
            default = param.default
            if default is inspect.Parameter.empty:
                args.append(name)
            else:
                if isinstance(default, str):
                    val = '=\'{}\''.format(default)
                else:
                    val = '=' + str(default)
                args.append(name + val)
        s += '{}({})'.format(fname, ', '.join(args[argskip:]))
    else:
        #args = comments.replace('\n', '').replace(' ', '').replace('#', '').split(',')
        #s += ', '.join(args)
        args = comments.replace('# ', '').split('\n')[:-1]
        s += ('\n' + len(modpath)*' ').join(args)

    return s + '\n```\n'

def write_class(f, name, obj, attrs, modpath, **kwargs):

    classlevel = kwargs.get('classlevel', DEFAULT_CLASSLEVEL)
    funclevel = classlevel + 1

    f.write('#'*classlevel + ' {}\n'.format(name))
    f.write('\n')
    f.write(inspect.getdoc(obj))
    f.write('\n\n')
    write_function(f, name, getattr(obj, attrs.get('maindocfuncname', '__init__')),
                   header=False, inclass=True, modpath=modpath, postdoc=attrs.get('postdoc', ''))
    for method in attrs.get('methods', []):
        write_function(f, method, getattr(obj, method), funclevel, inclass=True, modpath='')

    if len(properties := attrs.get('properties', [])):
        f.write('#'*funclevel + ' properties\n\n')
    for prop in properties:
        write_property(f, prop, getattr(obj, prop), classlevel + 2)

def write_function(f, name, func, funclevel=None, header=True, inclass=False, modpath='energyflow.', postdoc='',):

    try:

        if name.startswith('_'):
            return False
        if header:
            if name.endswith('4doc'):
                name = name[:-4]
            f.write('#'*funclevel + ' ' + name + '\n')
            f.write('\n')
        f.write(get_function_signature(name, func, 1 if inclass else 0, modpath))
        f.write('\n')
        f.write(inspect.getdoc(func))
        f.write('\n\n')
        if len(postdoc):
            f.write(postdoc)
            f.write('\n\n')
        return True

    except:
        print(name)
        raise

def write_property(f, name, func, proplevel, modpath=''):

    f.write('#'*proplevel + ' ' + name + '\n')
    f.write('\n')
    f.write('```python\n')
    f.write(modpath + name + '\n')
    f.write('```\n')
    f.write('\n')
    f.write(inspect.getdoc(func))
    f.write('\n\n')

def write_hline(f):
    f.write('\n----\n\n')

def write_module(f, module=None, classes=[], functions=[], modpath='energyflow.', **kwargs):

    module_docstring = inspect.getdoc(module)
    f.write(('' if module_docstring is None else module_docstring) + '\n')
    if module_docstring is not None:
        write_hline(f)

    class_names = set(cl['name'] for cl in classes)

    funclevel = kwargs.get('funclevel', DEFAULT_FUNCLEVEL)
    for fname in functions:
        if fname not in class_names:
            if write_function(f, fname, getattr(module, fname), funclevel, modpath=modpath):
                write_hline(f)

    for cl in classes:
        write_class(f, cl['name'], getattr(module, cl['name']), cl, modpath, **kwargs)
        write_hline(f)

def main():

    # handle getting examples
    doc_code_re = re.compile(r'"""([\w\W]*?)"""([\w\W]*)')
    examples_str = ''
    for example_file in example_files:
        with open(os.path.join(example_dir, example_file), 'r') as f:
            docstring, code = doc_code_re.match(f.read()).group(1, 2)
            examples_str += '### {}\n\n{}\n```python\n{}\n```\n\n'.format(example_file, 
                                                                          docstring, 
                                                                          code.lstrip('\n'))

    template_dict['example_files'] = examples_str

    # handle reading changelog
    with open('../CHANGELOG', 'r') as f:
        template_dict['changelog'] = f.read()

    # handle templates
    for template_file in os.listdir(template_dir):
        new_file = template_file.replace('_template', '')
        with open(os.path.join(template_dir, template_file), 'r') as fr: 
            with open(os.path.join('sources', new_file), 'w') as fw:
                fw.write(fr.read().format(**template_dict))

    for mdname, modlist in docs.items():
        with open(os.path.join('sources', 'docs', mdname + '.md'), 'w') as f:
            for moddict in modlist:
                write_module(f, **moddict)

if __name__ == '__main__':
    main()
