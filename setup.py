from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name             = 'tl',
    version          = '0.1',
    description      = 'Transfer Learning Tremor Location',
    long_description =  readme(),
    url              = 'http://github.com/lvanderlaat/tl',
    author           = 'Leonardo van der Laat',
    author_email     = 'laat@umich.edu',
    packages         = ['tl'],
    install_requires = [
    ],
    scripts          = [
        'bin/tl-clean',
        'bin/tl-feature-extraction',
        'bin/tl-feature-engineering',
        'bin/tl-feature-selection',
        'bin/tl-get-wfs-eq',
        'bin/tl-help',
        'bin/tl-predict',
        'bin/tl-scan',
        'bin/tl-train',
    ],
    zip_safe         = False
)
