use ordered_float::NotNan;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use typed_builder::TypedBuilder;

#[derive(Debug, Serialize, Deserialize)]
pub enum EmbeddingModel {
    #[serde(rename = "text-embedding-ada-002")]
    #[serde(alias = "text-embedding-ada-002-v2")]
    Ada,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    String(String),
    Array(Vec<String>),
}

#[derive(Debug, Serialize, Deserialize, TypedBuilder)]
pub struct EmbeddingRequest {
    #[builder(default = EmbeddingModel::Ada)]
    pub model: EmbeddingModel,
    pub input: EmbeddingInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub user: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingItem {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingItem>,
    pub model: EmbeddingModel,
    pub usage: EmbeddingUsage,
}

impl EmbeddingRequest {
    pub async fn request(
        self,
        client: &Client,
        api_key: &str,
    ) -> Result<EmbeddingResponse, Box<dyn std::error::Error + Send + Sync>> {
        let response = client
            .post("https://api.openai.com/v1/embeddings")
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&self)
            .send()
            .await?
            .error_for_status()?;

        Ok(response.json::<EmbeddingResponse>().await?)
    }
}

pub async fn string_embeddings(
    strings: impl Iterator<Item = impl Into<String>>,
    client: &Client,
    key: &str,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
    let request = EmbeddingRequest::builder()
        .input(EmbeddingInput::Array(strings.map(|s| s.into()).collect()))
        .build();

    let response = request.request(client, key).await?;

    Ok(response.data.into_iter().map(|i| i.embedding).collect())
}

// fn dot_product_fixed<T, const LEN: usize>(a: &[T; LEN], b: &[T; LEN]) -> T
// where
//     T: std::default::Default + std::ops::Mul<Output = T> + std::ops::AddAssign + Copy,
// {
//     let mut sum = T::default();
//     for i in 0..LEN {
//         sum += a[i] * b[i];
//     }
//     sum
// }

fn dot_product<T>(a: &[T], b: &[T]) -> T
where
    T: std::ops::Mul<Output = T> + std::iter::Sum + Copy,
{
    a.iter().zip(b.iter()).map(|(a, b)| *a * *b).sum()
}

pub trait Embedding {
    fn embedding(&self) -> &[f32];
}

impl Embedding for Vec<f32> {
    fn embedding(&self) -> &[f32] {
        return &self;
    }
}

impl Embedding for &Vec<f32> {
    fn embedding(&self) -> &[f32] {
        return &self;
    }
}

impl Embedding for &[f32] {
    fn embedding(&self) -> &[f32] {
        return self;
    }
}

pub struct EmbeddingDistance<T> {
    item: T,
    distance: NotNan<f32>,
}

impl<T> PartialEq for EmbeddingDistance<T> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<T> PartialOrd for EmbeddingDistance<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.distance.cmp(&other.distance))
    }
}

impl<T> Eq for EmbeddingDistance<T> {}

impl<T> Ord for EmbeddingDistance<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.cmp(&other.distance)
    }
}

pub fn knn_search<'a, T, U>(
    query: &T,
    content: impl Iterator<Item = &'a U>,
    k: usize,
) -> Vec<(&'a U, f32)>
where
    T: Embedding,
    U: Embedding,
{
    let mut heap = std::collections::BinaryHeap::with_capacity(k);
    for item in content {
        let distance = dot_product(query.embedding(), item.embedding());
        if heap.len() < k {
            heap.push(Reverse(EmbeddingDistance {
                item,
                distance: NotNan::new(distance).unwrap(),
            }));
        } else if heap.peek().unwrap().0.distance.into_inner() < distance {
            heap.pop();
            heap.push(Reverse(EmbeddingDistance {
                item,
                distance: NotNan::new(distance).unwrap(),
            }));
        }
    }
    heap.into_sorted_vec()
        .into_iter()
        .map(|item| (item.0.item, item.0.distance.into_inner()))
        .collect()
}

#[cfg(test)]
mod test {
    // use super::*;

    // #[tokio::test]
    // async fn test_simple_embeddings() {
    //     let env = crate::Environment::new();

    //     let client = Client::new();

    //     let requests = EmbeddingRequest::new(EmbeddingInput::Array(vec![
    //         "Do you remember talking with my cat about U.S. history yesterday?".to_string(),
    //         "Hello, my dog is cute".to_string(),
    //         "Hello, my cat is cute".to_string(),
    //         "We choose to go to the Moon in this decade and do the other things, not because they are easy, but because they are hard.".to_string(),
    //     ]));

    //     let mut embeddings = requests
    //         .request(&client, &env.openai_key)
    //         .await
    //         .unwrap()
    //         .data
    //         .into_iter()
    //         .map(|item| {
    //             println!("{:#?}", item.object);
    //             // (item.embedding, item.index)
    //             ExchangeVector {
    //                 embedding: item
    //                     .embedding
    //                     .into_iter()
    //                     .map(|f| NotNan::new(f))
    //                     .collect::<Result<Vec<_>, _>>()
    //                     .unwrap()
    //                     .try_into()
    //                     .unwrap(),
    //                 id: item.index,
    //             }
    //         })
    //         .collect::<Vec<_>>();

    //     let query = embeddings.remove(0);
    //     let results = knn_search(&query, &embeddings, 8);

    //     panic!("{:#?}", results);
    // }
}
