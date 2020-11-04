
# EnergyFlow jupyter/matplotlib defaults
c.InteractiveShellApp.matplotlib = 'inline'
c.InlineBackend.figure_formats = {'retina'}
c.InlineBackend.rc.update({'figure.figsize': (4,4),
                           'figure.dpi': 120,
                           'text.usetex': True,
                           'text.latex.preamble': r'\usepackage{amsmath}',
                           'font.family': 'serif',
                           })
