provider "google" {
  project     = var.project_id
  region      = "us-central1"
  zone        = "us-central1-a"
}

module "zip-upload-gcs" {
  source = "./modules/zip-upload-gcs"

  bucket_name = var.bucket_name
  function_source_location = var.function_source_location
  function_source_zip = var.function_source_zip
  project_id = var.project_id
}

# see https://registry.terraform.io/modules/GoogleCloudPlatform/cloud-functions/google/latest
module "cloud_functions2" {
  source  = "GoogleCloudPlatform/cloud-functions/google"
  version = "~> 0.9"
  location = "us-central1"
  project_id        = var.project_id
  function_name     = "cal-http"
  function_location = var.function_location
  runtime           = "python310"
  entrypoint        = "cal_http"
  storage_source = {
    bucket     = var.bucket_name
    object     =  module.zip-upload-gcs.function-zip-uploaded
    generation = null
  }
}