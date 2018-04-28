# Tf Gym App

[![Build Status](https://travis-ci.com/kaiquewdev/tf-gym-app.svg?token=fP2MzeqGP5sWPBqwVGGZ&branch=master)](https://travis-ci.com/kaiquewdev/tf-gym-app)

## Description

Small definition of a gym tensorflow app

## Installation

Remote install

```
[sudo] [-H] pip install qlearning
```

The primary can be dued like that

```
python setup.py install
```

Dependencies for that program work well

```
pip install gym gym[atari] tensorflow keras gym-super-mario-bros
```

or you can made an python environment with anaconda

```
conda create -n tf python=3.6 gym gym[atari] tensorflow keras gym-super-mario-bros
```

## Basic usage with example for a pre-defined case or a suited one

```
python app.py
```

or you can build a binary

```
bazel build :app
```

After that you can execute

```
./bazel-bin/app
```

### Properties available to activate as arguments

First small example

```
python app.py --environment_name "Gomoku19x19-v0" --render "presented" --action_type "dqn" --pre_defined_state_size "gym-gomoku"
```

## Thinking on relations like test

```
python app_test.py
```

or just doing the same to produce a binary in another way

```
bazel build :app_test
```

For tests too, you can have an access of that binary

```
./bazel-bin/app_test
```