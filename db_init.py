
import boto3
from logger_setup import setup_logger
import os

from dotenv import load_dotenv
import logging
from botocore.exceptions import ClientError
import yaml

boto3.set_stream_logger('botocore', level=logging.DEBUG)

logger = setup_logger()
load_dotenv()


aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
region_name = os.getenv('AWS_DEFAULT_REGION')

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)
    
    
dynamodb = boto3.resource(
    'dynamodb',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)


table_name = config['dynamo_tbl_name']

try:
    # Create the DynamoDB table
    table = dynamodb.create_table(
        TableName=table_name,
        KeySchema=[
            {
                'AttributeName': 'PK',
                'KeyType': 'HASH'  # Partition key
            },
            {
                'AttributeName': 'SK',
                'KeyType': 'RANGE'  # Sort key
            }
        ],
        AttributeDefinitions=[
            {'AttributeName': 'PK', 'AttributeType': 'S'},
            {'AttributeName': 'SK', 'AttributeType': 'S'}
        ],
        BillingMode='PAY_PER_REQUEST'
    )

    # Wait until the table exists before proceeding
    table.meta.client.get_waiter('table_exists').wait(TableName=table_name)

    logger.info(f"Table '{table_name}' created successfully!")

except ClientError as e:
    if e.response['Error']['Code'] == 'ResourceInUseException':
        logger.warning(f"Table '{table_name}' already exists.")
    else:
        logger.error(f"Unexpected error: {e}")

