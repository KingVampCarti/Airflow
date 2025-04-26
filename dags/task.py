import pendulum
from airflow.decorators import dag, task
from astronomer.providers.amazon.aws.sensors.s3 import S3KeySensorAsync
from airflow.operators.email import EmailOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from kubernetes.client import models as k8s
from mlflow_provider.operators.registry import GetLatestModelVersionsOperator
import os

DATA_PATH = os.environ["STORAGE_PATH"]
PVC = os.environ["PVC_NAME"]
K8S_CONTEXT = os.environ["CLUSTER_CONTEXT"]
RAY_HOST = os.environ["RAY_SERVER"]
MLFLOW_HOST = os.environ["MLFLOW_SERVER"]

mount = k8s.V1VolumeMount(
    name='data-vol',
    mount_path=DATA_PATH,
    read_only=False
)

vol = k8s.V1Volume(
    name='data-vol',
    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
        claim_name=PVC),
)

gpu_resources = k8s.V1ResourceRequirements(
    limits={'nvidia.com/gpu': '1'},
    requests={'nvidia.com/gpu': '1'}    
)

k8s_config = {
    "cluster_context": K8S_CONTEXT,
    "namespace": "default",
    "labels": {"k8s_pod": "dannnn_isuct666"},
    "get_logs": True,
    "delete_pod": True,
    "in_cluster": False,
    "config_file": "/home/astro/config",
    "volume_mounts": [mount],
    "volumes": [vol]
}

@dag(
    schedule_interval=None,
    start_date=pendulum.datetime(2022, 7, 20, tz="UTC"),
    catchup=False,
    tags=['dannnn_isuct666_xray'],
)
def dannnn_xray_flow():

    s3_check = S3KeySensorAsync(
        bucket_key="s3://ce-ml-data/xray_data_2_class.tgz",
        task_id="s3_check",
        aws_conn_id="my_aws_conn"
    )

    get_data = KubernetesPodOperator(
        task_id="get_data",
        name="dannnn_data_pod",
        image="astro/xray_services:0.0.3",
        cmds=["/bin/bash", "-c", "--",
              f"cd {DATA_PATH} && curl -O 'https://astro-datasets.s3.eu-central-1.amazonaws.com/xray_data_2_class.tgz' && tar -xzf xray_data_2_class.tgz"],
        **k8s_config
    )

    @task.kubernetes(
        image="fletchjeffastro/tfmlflow:0.0.6",
        name="dannnn_train_pod",
        task_id="train_model",
        startup_timeout_seconds=600,
        container_resources=gpu_resources,
        env_vars={
            "DATA_PATH": DATA_PATH,
            "RUN_ID": "{{dag_run.logical_date.strftime('%Y%m%d-%H%M%S')}}",
            "MLFLOW_HOST": MLFLOW_HOST
        },
        **k8s_config
    )
    def train_model():
        import tensorflow as tf
        from tensorflow.keras.preprocessing import image_dataset_from_directory
        import os
        import mlflow

        data_dir = os.path.join(os.environ["DATA_PATH"], "data/train/")
        batch_size = 32
        img_size = (224, 224)

        train_ds = image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=img_size,
            batch_size=batch_size
        )

        val_ds = image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=img_size,
            batch_size=batch_size
        )

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=img_size + (3,),
            include_top=False,
            weights='imagenet'
        )

        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0001),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        mlflow.set_tracking_uri(f"http://{os.environ['MLFLOW_HOST']}:5000")
        mlflow.keras.autolog()
        
        with mlflow.start_run():
            model.fit(train_ds, epochs=10, validation_data=val_ds)

        os.makedirs(f"{os.environ['DATA_PATH']}/models/{os.environ['RUN_ID']}", exist_ok=True)
        model.save(f"{os.environ['DATA_PATH']}/models/{os.environ['RUN_ID']}/model.h5")

    @task()
    def setup_ray(run_id):
        import ray
        from ray import serve

        ray.init(f"ray://{RAY_HOST}:10001")
        serve.start(detached=True)

        @serve.deployment
        async def xray_predictor(request):
            from PIL import Image
            import tensorflow as tf
            from io import BytesIO
            import base64
            import numpy as np
            
            img_data = await request.json()
            model = tf.keras.models.load_model(f"{DATA_PATH}/models/{run_id}/model.h5")
            img = Image.open(BytesIO(base64.b64decode(img_data['image'][22:])))
            img = img.resize((224, 224)).convert("RGB")
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)
            return {'result': 'normal' if prediction[0][0] < 0.5 else 'pneumonia'}

        if 'xray_predictor' in serve.list_deployments():
            xray_predictor.delete()
        xray_predictor.deploy()

    @task.kubernetes(
        image="astro/xray_services:0.0.3",
        name="dannnn_streamlit_pod",
        task_id="update_streamlit",
        env_vars={
            "DATA_PATH": DATA_PATH,
            "RUN_ID": "{{dag_run.logical_date.strftime('%Y%m%d-%H%M%S')}}",
            "RAY_HOST": RAY_HOST
        },
        **k8s_config
    )
    def update_streamlit():
        import os
        from urllib.request import urlopen
        
        url = 'https://raw.githubusercontent.com/KingVanpCarti/Airflow/main/include/code/streamlit_app.py'
        with urlopen(url) as f:
            content = f.read().decode('utf-8')
        
        content = content.replace("RAY_SERVER=''", f"RAY_SERVER='{os.environ['RAY_HOST']}'")
        content = content.replace("STORAGE_PATH=''", f"STORAGE_PATH='{os.environ['DATA_PATH']}'")
        content = content.replace("CURRENT_RUN=''", f"CURRENT_RUN='{os.environ['RUN_ID']}'")
        
        with open(f"{os.environ['DATA_PATH']}/streamlit_app.py", 'w') as f:
            f.write(content)

    model_check = GetLatestModelVersionsOperator(
        mlflow_conn_id='my_mlflow',
        task_id='model_check',
        name='xray_model'
    )

    notify = EmailOperator(
        conn_id='smtp_default',
        task_id="notify",
        to='dannnn_isuct666@astronomer.io',
        subject='[dannnn_isuct666] Training Done',
        html_content=f"Xray model trained at {{{{dag_run.logical_date.strftime('%Y%m%d-%H%M%S')}}}}"
    )

    s3_check >> get_data >> train_model() >> [
        setup_ray("{{dag_run.logical_date.strftime('%Y%m%d-%H%M%S')}}"), 
        update_streamlit()
    ] >> model_check >> notify

dannnn_xray_flow()