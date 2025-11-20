def load_from_azure(container, blob_name, conn_str):
    """Download blob to dataframe"""
    blob_service = BlobServiceClient.from_connection_string(conn_str)
    blob_client = blob_service.get_blob_client(container=container, blob=blob_name)

    # Download blob as stream
    stream = blob_client.download_blob().readall()
    return pd.read_csv(pd.io.common.BytesIO(stream))  # or pd.read_parquet if parquet