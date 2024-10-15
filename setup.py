import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_git_commit_number():
    if not os.path.exists('.git'):
        return '0000000'

    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        f.write(f'__version__ = "{version}"')


if __name__ == '__main__':
    version = f'0.1.0+{get_git_commit_number()}'
    write_version_to_file(version, 'pccls/version.py')

    setup(
        name='pccls',
        version=version,
        description='OpenPCCls is a general codebase for 3D object classification from point cloud',
        install_requires=[
            'numpy',
            'llvmlite',
            'numba',
            'easydict',
            'pyyaml',
            'SharedArray',
        ],

        author='Cheng Mei',
        author_email='meicheng@smail.nju.edu.cn',
        license='Apache License 2.0',
        packages=find_packages(exclude=[]),
        cmdclass={
            'build_ext': BuildExtension,
        },
        ext_modules=[
            make_cuda_ext(
                name='bpa_point_aggregation_cuda',
                module='pccls.ops.bpa_point_aggregation_ops',
                sources=[
                    'src/bpa_point_aggregation_api.cpp',
                    'src/bpa_furthest_point_sampling/bpa_furthest_point_sampling.cpp',
                    'src/bpa_furthest_point_sampling/bpa_furthest_point_sampling_gpu.cu',
                    'src/bpa_knn_query/bpa_knn_query.cpp',
                    'src/bpa_knn_query/bpa_knn_query_gpu.cu',
                    'src/bpa_ball_query/bpa_ball_query.cpp',
                    'src/bpa_ball_query/bpa_ball_query_gpu.cu',
                    'src/bpa_gather_query/bpa_gather_query.cpp',
                    'src/bpa_gather_query/bpa_gather_query_gpu.cu',
                    'src/bpa_gather_group/bpa_gather_group.cpp',
                    'src/bpa_gather_group/bpa_gather_group_gpu.cu',
                    'src/bpa_three_interpolate/bpa_three_interpolate.cpp',
                    'src/bpa_three_interpolate/bpa_three_interpolate_gpu.cu',
                ]
            ),
        ],
    )
