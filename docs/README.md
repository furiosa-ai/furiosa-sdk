# Building Sphinx Documents

Let's install the required packages as the following:
```
$ pip install -r requirements
```

Generate HTML documents and copy the output to a specific furiosa-doc path:
```sh

  DOCS_PATH=../../furiosa-docs/ DOCS_VERSION=v0.6.0 make en deploy-en
  DOCS_PATH=../../furiosa-docs/ DOCS_VERSION=v0.6.0 make ko deploy-ko
```