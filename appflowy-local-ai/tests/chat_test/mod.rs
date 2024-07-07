use crate::util::LocalAITest;
use std::time::Duration;
use tokio_stream::StreamExt;

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

  // let questions = test
  //   .chat_manager
  //   .get_related_question(&chat_id)
  //   .await
  //   .unwrap();
  // println!("related questions: {:?}", questions);

  tokio::time::sleep(Duration::from_secs(5)).await;
}
