import requests, mlflow, os
import mlflow.pytorch
import babelfish as bf
import babelfish.helpers
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD")

def load_model_from_run_info(run_info, cuda=True, device_id=None,
    artifact_path="/models"):
    # TODO cache this / wrap all mlflow with some sort of memoization?
    # have babelfish-specific cache folder
    """Load pytorch model from mlflow URI
    
    Arguments:
        run_id {run_id}
    
    Keyword Arguments:
        device {str} -- GPU int / device ID (default: {None})
    
    Returns:
        pytorch module
    """    
    artifact_uri = run_info.artifact_uri
    model_uri = artifact_uri + artifact_path

    print("loading ", model_uri)
    model = mlflow.pytorch.load_model(model_uri, map_location=device_id)
    print("converting to cuda")
    if cuda:
        model.cuda(device_id)
    print("finished loading")
    return model

def get_run_info(run_id):
    """Load run info from mlflow URI
    
    Arguments:
        run_id {run_id}
    
    Returns:
        dict
    """    
    try:
        run_info = requests.get(MLFLOW_TRACKING_URI+"/api/2.0/mlflow/runs/get",
            auth=(MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD),
            params={"run_id": run_id})
        run_info = mlflow.utils.proto_json_utils.message_to_json(run_info)
    except Exception as e:
        print("""make sure `echo $MLFLOW_TRACKING_URI \
            $MLFLOW_TRACKING_USERNAME $MLFLOW_TRACKING_PASSWORD` returns \
            values""")
        raise(e)
    # print(run_info) # TODO why is status: FAILED?
    return run_info