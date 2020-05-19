## Introduction

MSCV is a foundational python library for computer vision research.

It provides the following functionalities.

- Image IO and processing
- Checkpoint processing
- Test time Argumentation
- Useful Meters(Avergae Meter, EMA Meter,...)
- Model structure printing
- TensorBoard Summary

NOTE: MSCV requires Python 3.6+.

## Installation

**For pip**  

```bash
pip install mscv
```

**For source**

Clone the repo, cd into it and run `pip install .` command.

```bash
git clone https://github.com/misads/mscv.git
cd mscv
pip install -e .
```

**For conda**

```bash
source ~/anaconda3/bin/activate
conda activate <env>
python setup.py install
```

A configure file `mscv.egg-info` will be generated in the repo directory. Copy `mscv` and `mscv.egg-info` to your `site-packages` folder.


