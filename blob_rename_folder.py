from azure.storage.blob import BlobServiceClient

conn_str = "DefaultEndpointsProtocol=https;AccountName=oslotestvideo;AccountKey=zOevIegkZjld6ciTY+alA+YkzZ2gdAWVP7rkuhty5NAZ67AtiBB3fRTaa+eE3UbqhgwHZvWkOM0L+ASt4zCx6g==;EndpointSuffix=core.windows.net"
container_name = "instavideo"
old_prefix = "hydra_plus_tar_file/"
new_prefix = "hydra_plus_tar_files/"

blob_service = BlobServiceClient.from_connection_string(conn_str)
container_client = blob_service.get_container_client(container_name)

for blob in container_client.list_blobs(name_starts_with=old_prefix):
    old_blob = container_client.get_blob_client(blob.name)
    new_name = blob.name.replace(old_prefix, new_prefix, 1)
    new_blob = container_client.get_blob_client(new_name)

    # Copy to new location
    new_blob.start_copy_from_url(old_blob.url)
    # Optionally delete old blob after copy
    old_blob.delete_blob()
