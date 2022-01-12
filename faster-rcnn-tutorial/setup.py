import setuptools

setuptools.setup(
    name="faster_rcnn_tutorial",
    version="0.0.1",
    author="ifding",
    author_email="",
    url="https://github.com/ifding/faster-rcnn-tutorial",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scikit-image',
        'sklearn',
        'neptune-contrib',
        'python-dotenv',        
        'albumentations==0.5.2',
        'pytorch-lightning==1.3.5',
        'torch==1.8.1',
        'torchvision==0.9.1',
        'torchsummary==1.5.1',
        'torchmetrics==0.2.0'
    ]
)
