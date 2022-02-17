- jupyter notebook --generate-config

```python
from notebook.auth import passwd

passwd()

c.NotebookApp.ip='*'
c.NotebookApp.password = u'xxx'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888

```

- jupyter notebook
