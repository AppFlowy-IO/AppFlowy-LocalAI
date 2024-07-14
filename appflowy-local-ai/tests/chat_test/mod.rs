use crate::util::{get_asset_path, setup_log, LocalAITest};
use appflowy_local_ai::chat_plugin::{ChatPluginConfig, LocalChatLLMChat};
use appflowy_local_ai::plugin_request::download_plugin;

use appflowy_plugin::core::plugin::{handle_macos_security_check, PluginInfo};
use appflowy_plugin::manager::PluginManager;
use std::env::temp_dir;
use std::path::PathBuf;
use std::sync::Arc;
use tokio_stream::StreamExt;
use zip_extensions::zip_extract;

#[tokio::test]
async fn load_chat_model_test() {
  let test = LocalAITest::new().unwrap();
  test.init_chat_plugin().await;
  test.init_embedding_plugin().await;

  let chat_id = uuid::Uuid::new_v4().to_string();
  let resp = test
    .send_chat_message(&chat_id, "translate 你好 to english")
    .await;
  eprintln!("chat response: {:?}", resp);

  let score = test.calculate_similarity(&resp, "Hello").await;
  assert!(score > 0.9, "score: {}", score);
}

#[tokio::test]
async fn ci_chat_stream_test() {
  let test = LocalAITest::new().unwrap();
  test.init_chat_plugin().await;
  test.init_embedding_plugin().await;
  let chat_id = uuid::Uuid::new_v4().to_string();

  let mut resp = test.stream_chat_message(&chat_id, "what is banana?").await;
  let mut list = vec![];
  while let Some(s) = resp.next().await {
    list.push(String::from_utf8(s.unwrap().to_vec()).unwrap());
  }

  let answer = list.join("");
  eprintln!("response: {:?}", answer);

  let expected = r#"banana is a fruit that belongs to the genus _______, which also includes other fruits such as apple and pear. It has several varieties with different shapes, colors, and flavors depending on where it grows. Bananas are typically green or yellow in color and have smooth skin that peels off easily when ripe. They are sweet and juicy, often eaten raw or roasted, and can also be used for cooking and baking. In some cultures, banana is considered a symbol of good luck, fertility, and prosperity. Bananas originated in Southeast Asia, where they were cultivated by early humans thousands of years ago. They are now grown around the world as a major crop, with significant production in many countries including the United States, Brazil, India, and China#"#;
  let score = test.calculate_similarity(&answer, expected).await;
  assert!(score > 0.7, "score: {}", score);
}

#[tokio::test]
async fn ci_chat_with_pdf() {
  let test = LocalAITest::new().unwrap();
  test.init_chat_plugin().await;
  test.init_embedding_plugin().await;
  let chat_id = uuid::Uuid::new_v4().to_string();
  let pdf = get_asset_path("AppFlowy_Values.pdf");
  test.chat_manager.index_file(&chat_id, pdf).await.unwrap();

  let resp = test
    .chat_manager
    .ask_question(
      &chat_id,
      // "what is the meaning of Aim High and Iterate in AppFlowy?",
      "what is AppFlowy Values?",
    )
    .await
    .unwrap();

  println!("chat with pdf response: {}", resp);

  let expected = r#"
1. **Mission Driven**: Our mission is to enable everyone to unleash their potential and achieve more with secure workplace tools.
2. **Collaboration**: We pride ourselves on being a great team. We foster collaboration, value diversity and inclusion, and encourage sharing.
3. **Honesty**: We are honest with ourselves. We admit mistakes freely and openly. We provide candid, helpful, timely feedback to colleagues with respect, regardless of their status or whether they disagree with us.
4. **Aim High and Iterate**: We strive for excellence with a growth mindset. We dream big, start small, and move fast. We take smaller steps and ship smaller, simpler features.
5. **Transparency**: We make information about AppFlowy public by default unless there is a compelling reason not to. We are straightforward and kind with ourselves and each other.
"#;
  let score = test.calculate_similarity(&resp, expected).await;
  assert!(score > 0.8, "score: {}", score);
}

#[tokio::test]
async fn load_aws_chat_bin_test() {
  setup_log();
  let plugin_manager = PluginManager::new();
  let llm_chat = LocalChatLLMChat::new(Arc::new(plugin_manager));

  let chat_bin = chat_bin_path().await;
  // clear_extended_attributes(&chat_bin).await.unwrap();

  let mut chat_config = ChatPluginConfig::new(chat_bin, chat_model()).unwrap();
  handle_macos_security_check(&PluginInfo {
    name: "".to_string(),
    exec_path: chat_config.chat_bin_path.clone(),
  });

  chat_config = chat_config.with_device("gpu");
  llm_chat.init_chat_plugin(chat_config).await.unwrap();

  let chat_id = uuid::Uuid::new_v4().to_string();
  let resp = llm_chat
    .ask_question(&chat_id, "what is banana?")
    .await
    .unwrap();
  assert!(!resp.is_empty());
  eprintln!("response: {:?}", resp);
}

async fn chat_bin_path() -> PathBuf {
  let url = "https://appflowy-local-ai.s3.amazonaws.com/macos-latest/AppFlowyLLM_release.zip?AWSAccessKeyId=AKIAVQA4ULIFKSXHI6PI&Signature=gfafCIkenNJpB351HIkYqDUMvqs%3D&Expires=1720914632";
  // let url = "";
  let temp_dir = temp_dir().join("download_plugin");
  if !temp_dir.exists() {
    std::fs::create_dir(&temp_dir).unwrap();
  }
  let path = download_plugin(url, &temp_dir, "AppFlowyLLM.zip", None, None)
    .await
    .unwrap();
  println!("Downloaded plugin to {:?}", path);

  zip_extract(&path, &temp_dir).unwrap();
  temp_dir.join("chat_plugin")
}

fn chat_model() -> PathBuf {
  let model_dir = PathBuf::from(dotenv::var("LOCAL_AI_MODEL_DIR").unwrap());
  let chat_model = dotenv::var("LOCAL_AI_CHAT_MODEL_NAME").unwrap();
  model_dir.join(chat_model)
}
