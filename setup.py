import sys
import platform
from setuptools import setup, find_packages

python_version = '.'.join(map(str, sys.version_info[:2]))
os_name = platform.system().lower()

torch_package_urls = {
    '3.10': {
        'linux': 'torch-2.0.1%2Bcu118-cp310-cp310-linux_x86_64.whl',
        'windows': 'torch-2.0.1%2Bcu118-cp310-cp310-win_amd64.whl'
    },
    '3.11': {
        'linux': 'torch-2.0.1%2Bcu118-cp311-cp311-linux_x86_64.whl',
        'windows': 'torch-2.0.1%2Bcu118-cp311-cp311-win_amd64.whl'
    },
    '3.8': {
        'linux': 'torch-2.0.1%2Bcu118-cp38-cp38-linux_x86_64.whl',
        'windows': 'torch-2.0.1%2Bcu118-cp38-cp38-win_amd64.whl'
    },
    '3.9': {
        'linux': 'torch-2.0.1%2Bcu118-cp39-cp39-linux_x86_64.whl',
        'windows': 'torch-2.0.1%2Bcu118-cp39-cp39-win_amd64.whl'
    }
}

torchvision_package_urls = {
    '3.10': {
        'linux': 'torchvision-0.15.2%2Bcu118-cp310-cp310-linux_x86_64.whl',
        'windows': 'torchvision-0.15.2%2Bcu118-cp310-cp310-win_amd64.whl'
    },
    '3.11': {
        'linux': 'torchvision-0.15.2%2Bcu118-cp311-cp311-linux_x86_64.whl',
        'windows': 'torchvision-0.15.2%2Bcu118-cp311-cp311-win_amd64.whl'
    },
    '3.8': {
        'linux': 'torchvision-0.15.2%2Bcu118-cp38-cp38-linux_x86_64.whl',
        'windows': 'torchvision-0.15.2%2Bcu118-cp38-cp38-win_amd64.whl'
    },
    '3.9': {
        'linux': 'torchvision-0.15.2%2Bcu118-cp39-cp39-linux_x86_64.whl',
        'windows': 'torchvision-0.15.2%2Bcu118-cp39-cp39-win_amd64.whl'
    }
}

if python_version in torch_package_urls:
    torch_url = torch_package_urls[python_version][os_name]
    torchvision_url = torchvision_package_urls[python_version][os_name]
else:
    sys.exit(f"Unsupported Python version: {python_version}")

torch_path = f"https://download.pytorch.org/whl/cu118/{torch_url}"
torchvision_path = f"https://download.pytorch.org/whl/cu118/{torchvision_url}"

setup(
    name='deforum',
    version='0.7a',
    packages=find_packages(),
    package_data={'deforum': ['test/test.png']},
    install_requires=[
        #f'torch@{torch_path}',
        #f'torchvision@{torchvision_path}',
        'einops==0.6.0',
        'numexpr==2.8.4',
        'matplotlib==3.7.1',
        'pandas==1.5.3',
        'av==10.0.0',
        'pims==0.6.1',
        'imageio-ffmpeg==0.4.8',
        'rich==13.3.2',
        'gdown==4.7.1',
        'py3d==0.0.87',
        'librosa==0.10.0.post2',
        'numpy==1.24.2',
        'opencv-contrib-python==4.7.0.72',
        'basicsr==1.4.2',
        'timm==0.6.13',
    ],
    entry_points={
        'console_scripts': [
            'deforum=deforum.cmd:main',
        ],
    },
)