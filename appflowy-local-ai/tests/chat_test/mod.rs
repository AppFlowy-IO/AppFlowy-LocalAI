use crate::util::{get_asset_path, LocalAITest};
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
