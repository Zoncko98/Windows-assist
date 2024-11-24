from setuptools import setup, find_packages

setup(
    name='ai-assistant',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'transformers>=4.32.0',
        'datasets>=2.13.0',
        'peft',
        'requests',
        'beautifulsoup4',
        'lxml',
        'tk',
        'pyttsx3',
        'speechrecognition',
        'psutil',
        'torch>=2.0.0',
        'torchvision',
        'torchaudio',
        'virtualenv'
    ],
    entry_points={
        'console_scripts': [
            'ai-assistant=main:main',  # Assuming `main.py` contains a `main` function
        ],
    },
)
