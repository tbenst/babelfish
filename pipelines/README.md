## setup

```
export AIRFLOW_HOME="~/code/lensman-airflow"
airflow initdb
```

## start
```
airflow webserver -p 8080
airflow scheduler
```