# zebrafish_passivity
`FILE=both_no_train_0919.out; nohup python main.py 0 freeze > $FILE & tail -f $FILE`

##
```
nvidia-docker build -t bf .
nvidia-docker run -v ~/code:/code -v /data:/data -v ~/Notebooks:/Notebooks -p 8880:8880 -it bf /bin/bash
```


## nix
- switch to moviepy over scikit-video
- add environment.yml
