from setuptools import setup


setup(
    name='LISAflow',
    version='0.1',
    description='Implementation of Normalising flow for LISA data analysis',
    url='https://gitlab.in2p3.fr/ai_for_lisa/lisaflow',
    author='Natalia Korsakova',
    author_email='korsakova@apc.in2p3.fr',
    license='MIT',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'Programming Language :: Python :: 3',
    ],
    keywords=['LISA','normalising flow','parameter estimation'],
    packages=setuptools.find_packages(),
    install_requires=['torch','numpy','matplotlib','corner','h5py','scipy']
)


