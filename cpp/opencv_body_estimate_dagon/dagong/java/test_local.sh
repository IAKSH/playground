#!/bin/bash
javac ./Demo.java -h .
java -Djava.library.path=./ Demo
