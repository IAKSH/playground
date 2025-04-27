#!/bin/bash
javac ./Demo.java -h .
java -Djava.library.path=/home/lain/Desktop/playground/cpp_playground/opencv_body_estimate_dagon/build/dagong/ Demo
