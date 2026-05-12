# Output variable definitions

output "function-zip-uploaded" {
  description = "Name of the zip file uploaded"
  value = google_storage_bucket_object.function-source.name
}