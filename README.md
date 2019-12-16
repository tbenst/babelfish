## setup
need to set following variables in .bashrc or similar, with actual username / password:
```
export MLFLOW_TRACKING_URI="https://mlflow.tylerbenster.com";
export MLFLOW_TRACKING_USERNAME="USERNAME";
export MLFLOW_TRACKING_PASSWORD="PASSWORD";
```


# zebrafish_passivity
`FILE=both_no_train_0919.out; nohup python scripts/main.py 0 freeze > $FILE & tail -f $FILE`

##
```
nvidia-docker build -t bf .
nvidia-docker run -v ~/code:/code -v /data:/data -v ~/Notebooks:/Notebooks -p 8880:8880 -it bf /bin/bash
```


## nix
- switch to moviepy over scikit-video
- add environment.yml


## building with Docker
- we use `Dockerfile-bootstrap` to get latest versions of python packages. However, this is not guaranteed to be reproducible
- `Dockerfile` aims to be reproducible by using version number in `strict_environment`
- can update `strict_environment` by:
```
> docker build -t bf_bootstrap -f Dockerfile-bootstrap .
> docker run -it bf_bootstrap /bin/bash
$ conda list --explicit
```

Now copy output to strict_environment



## copy nix-cache
```
nix copy --to 's3://nix-science?profile=nix-cache&region=us-west-2' nixpkgs.hello 
```

# TODO
- use PyTables for recording pipeline DAG per-file?
- fix environment.yml to use requirements.txt where possible?
- delete S3 artifacts when removed from mlflow server
- allow scripts to change while being able to retrieve default.nix for particular run_id (store default.nix in mlflow?? change to fetchFromGithub for babelfish, using niv)
    - 
- mypy linting