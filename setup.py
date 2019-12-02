from setuptools import setup

setup(name='datautils',
      version='0.0',
      description='Data utilities',
      url='https://github.com/robertibatullin/datautils',
      author='Robert Ibatullin',
      author_email='r.ibatullin@celado-media.ru',
      packages=['datautils'],
      scripts=['bin/annotate'],
      install_requires=[
          'numpy',
          'opencv-python',
          'python-magic'
      ],
      zip_safe=False)
