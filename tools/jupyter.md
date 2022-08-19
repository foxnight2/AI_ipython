- jupyter notebook --generate-config

```python

jupyter notebook --generate-config

c.NotebookApp.ip='*'
c.NotebookApp.password = u'xxx'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888



jupyter notebook --generate-config
echo " 
c.NotebookApp.ip='*' 
c.NotebookApp.password = u'' 
c.NotebookApp.open_browser = False 
c.NotebookApp.port = 8888 
" >>  /root/.jupyter/jupyter_notebook_config.py



from notebook.auth import passwd
passwd()

```

- jupyter notebook
- python -m http.server 