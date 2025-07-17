from setuptools import setup, find_packages

setup(
    name='healthybot',
    version='0.1.0',
    author='Sumit Dhar',
    author_email='sumiths.0015@gmail.com',
    description='A medical chatbot using LLMs and LangChain',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
)