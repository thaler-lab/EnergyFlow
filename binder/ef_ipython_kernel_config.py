
# EnergyFlow jupyter/matplotlib defaults
c.InteractiveShellApp.matplotlib = 'inline'
c.InlineBackend.figure_formats = {'retina'}
c.InlineBackend.rc.update({'figure.figsize': (4,4),
                           'figure.dpi': 120,
                           'text.usetex': True,
                           'text.latex.preview': True,
                           'text.latex.preamble': [r'\usepackage{amsmath}'],
                           'mathtext.fontset': 'cm',
                           'mathtext.rm': 'serif',
                           'font.family': 'serif',
                           })
