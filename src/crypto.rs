use hmac::{Hmac, Mac};
use sha2::Sha256;
use hex;
use std::env;
use subtle::ConstantTimeEq;

type HmacSha256 = Hmac<Sha256>;

pub struct CryptoEngine {
    secret: String,
}

impl CryptoEngine {
    pub fn new() -> Self {
        let secret = env::var("CUEMAP_SECRET_KEY")
            .unwrap_or_else(|_| "default_ephemeral_secret_change_me".to_string());
        Self { secret }
    }

    pub fn sign(&self, content: &str) -> String {
        let mut mac = HmacSha256::new_from_slice(self.secret.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(content.as_bytes());
        let result = mac.finalize();
        hex::encode(result.into_bytes())
    }

    pub fn verify(&self, content: &str, signature: &str) -> bool {
        let expected = self.sign(content);
        // Constant-time comparison to prevent timing attacks
        expected.as_bytes().ct_eq(signature.as_bytes()).into()
    }
}
