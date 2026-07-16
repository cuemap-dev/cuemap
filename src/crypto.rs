use chacha20poly1305::{
    aead::{Aead, KeyInit},
    ChaCha20Poly1305, Key, Nonce,
};
use hmac::Hmac;
use pbkdf2::pbkdf2;
use rand::{thread_rng, RngCore};
use ring::signature::{Ed25519KeyPair, KeyPair};
use sha2::Sha256;
use zeroize::{Zeroize, ZeroizeOnDrop};

#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct EncryptionKey(Vec<u8>);

impl EncryptionKey {
    pub fn new(key: Vec<u8>) -> Self {
        Self(key)
    }

    pub fn from_passphrase(passphrase: &str, salt: &[u8]) -> Self {
        let mut key = vec![0u8; 32]; // ChaCha20 key size is 32 bytes
                                     // Use Hmac<Sha256> as PRF
        let _ = pbkdf2::<Hmac<Sha256>>(passphrase.as_bytes(), salt, 100_000, &mut key);
        Self(key)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

pub fn compress(data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
    // 3 is default compression level, usually good balance
    zstd::stream::encode_all(std::io::Cursor::new(data), 3)
}

pub fn decompress(data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
    zstd::stream::decode_all(std::io::Cursor::new(data))
}

pub fn is_compressed(data: &[u8]) -> bool {
    // Zstd Magic Number: 0xFD2FB528 (Little Endian: 28 B5 2F FD)
    if data.len() < 4 {
        return false;
    }
    data[0] == 0x28 && data[1] == 0xB5 && data[2] == 0x2F && data[3] == 0xFD
}

pub fn encrypt(data: &[u8], key: &EncryptionKey) -> Result<Vec<u8>, String> {
    let cipher = ChaCha20Poly1305::new(Key::from_slice(key.as_bytes()));

    // Generate random 12-byte nonce
    let mut nonce_bytes = [0u8; 12];
    thread_rng().fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);

    // Encrypt
    let ciphertext = cipher
        .encrypt(nonce, data)
        .map_err(|e| format!("Encryption failed: {}", e))?;

    // Prepend nonce to ciphertext: [Nonce (12B) | Ciphertext]
    let mut result = Vec::with_capacity(12 + ciphertext.len());
    result.extend_from_slice(&nonce_bytes);
    result.extend(ciphertext);

    Ok(result)
}

pub fn decrypt(data: &[u8], key: &EncryptionKey) -> Result<Vec<u8>, String> {
    if data.len() < 12 {
        return Err("Data too short to contain nonce".to_string());
    }

    let (nonce_bytes, ciphertext) = data.split_at(12);
    let nonce = Nonce::from_slice(nonce_bytes);

    let cipher = ChaCha20Poly1305::new(Key::from_slice(key.as_bytes()));

    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|e| format!("Decryption failed: {}", e))
}

#[derive(Clone, Debug)]
pub struct SignedContext {
    pub algorithm: &'static str,
    pub signature: String,
    pub public_key: Option<String>,
}

/// Context signing for Grounded Recall.
///
/// Ed25519 is preferred because verifiers only need the public key. HMAC is
/// kept as a compatibility fallback for existing CUEMAP_SECRET_KEY deployments.
pub enum ContextSigner {
    Ed25519 { key_pair: Ed25519KeyPair },
    HmacSha256 { secret: Vec<u8> },
}

impl ContextSigner {
    pub fn from_ed25519_seed_hex(seed_hex: &str) -> Result<Self, String> {
        let normalized = seed_hex
            .trim()
            .strip_prefix("ed25519:")
            .unwrap_or(seed_hex.trim());
        let seed = hex::decode(normalized).map_err(|e| format!("invalid Ed25519 seed hex: {}", e))?;
        if seed.len() != 32 {
            return Err(format!(
                "Ed25519 signing private key must be a 32-byte hex seed, got {} bytes",
                seed.len()
            ));
        }
        let key_pair = Ed25519KeyPair::from_seed_unchecked(&seed)
            .map_err(|_| "invalid Ed25519 signing seed".to_string())?;
        Ok(Self::Ed25519 { key_pair })
    }

    pub fn from_hmac_secret(secret: Vec<u8>) -> Self {
        Self::HmacSha256 { secret }
    }

    pub fn sign(&self, data: &str) -> SignedContext {
        match self {
            Self::Ed25519 { key_pair } => {
                let signature = key_pair.sign(data.as_bytes());
                SignedContext {
                    algorithm: "ed25519",
                    signature: hex::encode(signature.as_ref()),
                    public_key: Some(format!("ed25519:{}", hex::encode(key_pair.public_key().as_ref()))),
                }
            }
            Self::HmacSha256 { secret } => {
                use hmac::{Hmac, Mac};
                use sha2::Sha256;

                type HmacSha256 = Hmac<Sha256>;

                let mut mac = <HmacSha256 as Mac>::new_from_slice(secret)
                    .expect("HMAC can take key of any size");
                mac.update(data.as_bytes());
                let result = mac.finalize();
                SignedContext {
                    algorithm: "hmac-sha256",
                    signature: hex::encode(result.into_bytes()),
                    public_key: None,
                }
            }
        }
    }
}
