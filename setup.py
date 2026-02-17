import setuptools

setuptools.setup(
    name='audio-speech-to-sign-language-converter',
    version='0.1.0',
    description='Python project',
    packages=setuptools.find_packages(),
    install_requires=[
        'nltk',
        'django',
        'gTTS>=2.3',
        'indic-nlp-library>=0.81',
    ],
    setup_requires=['nltk', 'joblib', 'click', 'regex', 'sqlparse', 'setuptools'],
)