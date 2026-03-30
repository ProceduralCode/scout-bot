from setuptools import setup, Extension
from Cython.Build import cythonize
setup(
	ext_modules=cythonize([
		Extension("fast_game", ["fast_game.pyx"]),
		Extension("fast_encoding", ["fast_encoding.pyx"]),
	], language_level="3"),
)
