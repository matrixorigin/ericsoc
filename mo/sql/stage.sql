use taxi;

CREATE STAGE fengttt_public
    URL = 's3://fengttt-public-data/'
    CREDENTIALS = {
        'AWS_KEY_ID' = 'your_access_key',
        'AWS_SECRET_KEY' = 'your_secret_key',
        'AWS_REGION' = 'us-east-2',
        'PROVIDER' = 'amazon',
        'ENDPOINT' = 'https://s3.us-east-2.amazonaws.com'
    };

