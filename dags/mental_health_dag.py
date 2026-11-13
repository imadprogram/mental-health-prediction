import os
import sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#
from src.pipeline import evaluate_model, prepare_data, train_model


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


with DAG(
    dag_id='mental_health_ml_pipeline',
    default_args=default_args,
    description='Automated pipeline for student mental health risk detection',
    schedule='@weekly',
    catchup=False,
    tags=['mlops', 'mental_health'],
) as dag:


    task_prepare = PythonOperator(
        task_id='prepare_data', python_callable=prepare_data
    )

    task_train = PythonOperator(
        task_id='train_model', python_callable=train_model
    )

    task_evaluate = PythonOperator(
        task_id='evaluate_model', python_callable=evaluate_model
    )


    task_prepare >> task_train >> task_evaluate