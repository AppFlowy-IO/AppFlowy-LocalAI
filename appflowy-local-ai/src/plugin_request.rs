use anyhow::anyhow;
use reqwest::Client;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio_stream::StreamExt;
use tokio_util::sync::CancellationToken;
use tracing::trace;

type ProgressCallback = Arc<dyn Fn(u64, u64) + Send + Sync>;

pub async fn download_plugin(
  url: &str,
  plugin_dir: &PathBuf,
  file_name: &str,
  cancel_token: Option<CancellationToken>,
  progress_callback: Option<ProgressCallback>,
) -> Result<PathBuf, anyhow::Error> {
  let client = Client::new();
  let response = client.get(url).send().await?;

  if !response.status().is_success() {
    return Err(anyhow!("Failed to download file: {}", response.status()));
  }

  let total_size = response
    .content_length()
    .ok_or(anyhow!("Failed to get content length"))?;

  // Create paths for the partial and final files
  let partial_path = plugin_dir.join(format!("{}.part", file_name));
  let final_path = plugin_dir.join(file_name);
  let mut part_file = File::create(&partial_path).await?;
  let mut stream = response.bytes_stream();
  let mut downloaded: u64 = 0;

  while let Some(chunk) = stream.next().await {
    if let Some(cancel_token) = &cancel_token {
      if cancel_token.is_cancelled() {
        trace!("Download canceled");
        fs::remove_file(&partial_path).await?;
        return Err(anyhow!("Download canceled"));
      }
    }

    let bytes = chunk?;
    part_file.write_all(&bytes).await?;
    downloaded += bytes.len() as u64;

    // Call the progress callback
    if let Some(progress_callback) = &progress_callback {
      progress_callback(downloaded, total_size);
    }
  }

  // Ensure all data is written to disk
  part_file.sync_all().await?;

  // Move the temporary file to the final destination
  fs::rename(&partial_path, &final_path).await?;
  trace!("Plugin downloaded to {:?}", final_path);
  Ok(final_path)
}

#[cfg(test)]
mod test {
  use super::*;
  use std::env::temp_dir;
  use zip_extensions::zip_extract;

  #[tokio::test]
  async fn download_plugin_test() {
    let url = "https://appflowy-local-ai.s3.amazonaws.com/macos-latest/AppFlowyLLM_release.zip?AWSAccessKeyId=AKIAVQA4ULIFKSXHI6PI&Signature=OUMmaMPSSbzBJoJu6KLSG0woTf4%3D&Expires=1720877298";
    if url.is_empty() {
      return;
    }

    let temp_dir = temp_dir().join("download_plugin");
    if !temp_dir.exists() {
      std::fs::create_dir(&temp_dir).unwrap();
    }

    let path = download_plugin(url, &temp_dir, "AppFlowyLLM.zip", None, None)
      .await
      .unwrap();
    println!("Downloaded plugin to {:?}", path);
    zip_extract(&path, &temp_dir).unwrap();

    // remove all files in temp_dir
    // std::fs::remove_dir_all(&temp_dir).unwrap();
  }
}
