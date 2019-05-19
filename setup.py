import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

modules = [
    CppExtension(
        'pth_nms.nms_cpu',
        ['pth_nms/src/nms.cpp']
        )
]

if torch.cuda.is_available():
    modules.append(
        CUDAExtension(
            'pth_nms.nms_gpu',
            ['pth_nms/src/nms_cuda.cpp',
             'pth_nms/src/cuda/nms_kernel.cu'],
            extra_compile_args={'cxx': ['-g'],
                                'nvcc': ['-O2']}
        )
    )

setup(
    name='pth_nms',
    description='PyTorch version of non maximum suppression from: https://github.com/rbgirshick/py-faster-rcnn/tree/master/lib/nms',
    author='Ross Girshick, Shaoqing Ren',
    url='https://github.com/meikuam/pth_nms',
    version='0.0.1',
    packages=find_packages(exclude=('tests',)),

    ext_modules=modules,
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch']
)
