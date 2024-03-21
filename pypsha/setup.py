from builtins import *
from shutil import copytree, copy, rmtree
from setuptools import setup
import os
import sys
from distutils.command.build_ext import build_ext
import glob
from setuptools.command.install import install
from pathlib import Path

def remove_if_exists(file_path):
    if os.path.exists(file_path):
        if os.path.islink(file_path) or os.path.isfile(file_path):
            os.remove(file_path)
        else:
            assert os.path.isdir(file_path)
            rmtree(file_path)


def find_file_path(pattern):
    files = glob.glob(pattern)
    if len(files) < 1:
        print("Failed to find the file %s." % pattern)
        exit(-1)
    if len(files) > 1:
        print("The file pattern %s is ambiguous: %s" % (pattern, files))
        exit(-1)
    return files[0]


current_dir = os.path.abspath(os.path.dirname(__file__))
long_description = Path(os.path.join(current_dir,"README.md")).read_text()

JAR_PATH = "jars"

in_source_dir = os.path.isfile("../pom.xml")

try:
  if in_source_dir:
      try:
        os.mkdir(JAR_PATH)
      except:
        print("Jar path already exists {0}".format(JAR_PATH),
              file=sys.stderr)

      copy("../java/target/IM_EventSetCalc_v3_0_ASCII.jar", os.path.join(JAR_PATH, "IM_EventSetCalc_v3_0_ASCII.jar"))

  PACKAGES = ["pypsha", "pypsha.jars"]
  PACKAGE_DIR = {"pypsha.jars" : "jars"}
  PACKAGE_DATA = {"pypsha.jars" : ["*.jar"]}

  setup(
          name='pypsha',
          version="0.0.4",
          packages=PACKAGES,
          include_package_data=True,
          package_dir=PACKAGE_DIR,
          package_data=PACKAGE_DATA,
          author='Neetesh Sharma',
          author_email='neetesh@stanford.edu',
		  url='https://github.com/neetesh-nks/pypsha',
          description='Package for regional probabilistic seismic hazard analysis',
		  long_description = long_description,
		  long_description_content_type='text/markdown',
          python_requires='>=3.5',
          zip_safe=False,
          classifiers=[
              'Programming Language :: Python :: 3.8']
      )
finally:
    if in_source_dir:
      remove_if_exists(JAR_PATH)


