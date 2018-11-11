#!/usr/bin/env bash

set -e

# Find where this script lives and then the Tensorflow root.
MY_DIRECTORY=`dirname $0`
export TENSORFLOW_SRC_ROOT=`realpath $MY_DIRECTORY/../../../..`

# Build a pip build tree.
BUILD_ROOT=/tmp/tflite_pip
rm -rf $BUILD_ROOT
mkdir -p $BUILD_ROOT/tflite_runtime/lite
mkdir -p $BUILD_ROOT/tflite_runtime/lite/python

# Build an importable module tree
cat > $BUILD_ROOT/tflite_runtime/__init__.py <<EOF;
import tflite_runtime.lite.interpreter
EOF

cat > $BUILD_ROOT/tflite_runtime/lite/__init__.py <<EOF;
from interpreter import Interpreter as Interpreter
EOF

cat > $BUILD_ROOT/tflite_runtime/lite/python/__init__.py <<EOF;

EOF

# Copy necessary source files
TFLITE_ROOT=$TENSORFLOW_SRC_ROOT/tensorflow/lite
cp -r  $TFLITE_ROOT/python/interpreter_wrapper $BUILD_ROOT
cp  $TFLITE_ROOT/python/interpreter.py $BUILD_ROOT/tflite_runtime/lite/
cp   $TFLITE_ROOT/tools/pip_package/setup.py $BUILD_ROOT
cp   $TFLITE_ROOT/tools/pip_package/MANIFEST.in $BUILD_ROOT

# Build the Pip
cd $BUILD_ROOT
python setup.py bdist_wheel
