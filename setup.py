import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='wroclaw_taxonomy',
    version='0.0.1',
    author='Tomasz Matuszek',
    author_email='tom.mateuszek@gmail.com',
    description='Function for creating dendrite using Wroclaw taxonomy',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/TomekMatuszek/wroclaw-taxonomy',
    project_urls = {
        #"Bug Tracker": "https://github.com/mike-huls/toolbox/issues"
    },
    license='MIT',
    packages=['wroclaw_taxonomy'],
    install_requires=['numpy', 'shapely', 'scipy', 'pandas', 'fiona', 'pyproj'],
)