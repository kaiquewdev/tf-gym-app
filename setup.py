from setuptools import setup
from setuptools import find_packages

setup(name='qlearning',
      version='1.0.0',
      description='Q-learning algorithms based on app structure',
      author='Kaique da Silva',
      author_email='kaique.silva.dev@gmail.com',
      url='https://github.com/kaiquewdev/tf-gym-app',
      download_url='https://github.com/kaiquewdev/tf-gym-app/tarball/1.0.0',
      license='BSD',
      entry_points={
            'console_scripts': [
                  'qlearning=app.app:run_main'
            ]
      },
      install_requires=['numpy>=1.14.2',
                        'tensorflow>=1.7.0',
                        'Keras>=2.1.4',
                        'pandas>=0.22.0',
                        'gym>=0.10.5',
                        'gym_gomoku>=0.0.1',
                        'nesgym-super-mario-bros>=0.3.1'],
      extras_require={
      	'h5py': ['h5py']
      },
	classifiers=[
      	'Development Status :: 5 - Production/Stable',
        	'Intended Audience :: Developers',
        	'Intended Audience :: Education',
        	'Intended Audience :: Science/Research',
        	'License :: OSI Approved :: BSD License',
        	'Programming Language :: Python :: 3.6',
        	'Topic :: Software Development :: Libraries',
        	'Topic :: Software Development :: Libraries :: Python Modules'
	],
	packages=find_packages())