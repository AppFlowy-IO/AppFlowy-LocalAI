use crate::util::LocalAITest;
use tokio_stream::StreamExt;
#[tokio::test]
async fn generate_embedding_test() {
  let test = LocalAITest::new().unwrap();
  test.init_embedding_plugin().await;

  let embedding = test.generate_embedding("hello world").await;
  assert_eq!(embedding.len(), 1);
}

#[tokio::test]
async fn load_chat_model_test() {
  let test = LocalAITest::new().unwrap();
  test.init_chat_plugin().await;
  test.init_embedding_plugin().await;

  let chat_id = uuid::Uuid::new_v4().to_string();
  let resp = test.send_chat_message(&chat_id, "hello world").await;
  eprintln!("chat response: {:?}", resp);

  let score = test.calculate_similarity(&resp, "Hello! How can I help you today? Is there something specific you would like to know or discuss").await;
  assert!(score > 0.9, "score: {}", score);
}

#[tokio::test]
async fn chat_stream_test() {
  let test = LocalAITest::new().unwrap();
  test.init_chat_plugin().await;
  test.init_embedding_plugin().await;
  let chat_id = uuid::Uuid::new_v4().to_string();

  let mut resp = test.stream_chat_message(&chat_id, "hello world").await;
  let mut list = vec![];
  while let Some(s) = resp.next().await {
    list.push(String::from_utf8(s.unwrap().to_vec()).unwrap());
  }

  let answer = list.join("");
  eprintln!("response: {:?}", answer);

  let score = test.calculate_similarity(&answer, "Hello! How can I help you today? Is there something specific you would like to know or discuss").await;
  assert!(score > 0.9, "score: {}", score);
}
