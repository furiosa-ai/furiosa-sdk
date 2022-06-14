# Building Sphinx Documents

Let's install the required packages as follows:
```
$ pip install -r requirements.txt
```

Generate HTML documents and copy the output to a specific furiosa-doc path:
```sh

  DOCS_PATH=../../furiosa-docs/ DOCS_VERSION=v0.7.0 make en deploy-en
  DOCS_PATH=../../furiosa-docs/ DOCS_VERSION=v0.7.0 make ko deploy-ko
```
