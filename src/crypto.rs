
use chacha20poly1305::{
    aead::{Aead, KeyInit},
    ChaCha20Poly1305, Key, Nonce
};
use rand::{RngCore, thread_rng};
use zeroize::{Zeroize, ZeroizeOnDrop};
use sha2::Sha256;
use hmac::Hmac;
use pbkdf2::pbkdf2;

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
    let ciphertext = cipher.encrypt(nonce, data)
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
    
    cipher.decrypt(nonce, ciphertext)
        .map_err(|e| format!("Decryption failed: {}", e))
}

/// Helper struct for signing context (used by Grounded Recall)
pub struct CryptoEngine;

impl CryptoEngine {
    pub fn new() -> Self {
        Self
    }

    /// Sign data using HMAC-SHA256 with a compiled-in secret
    pub fn sign(&self, data: &str) -> String {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;
        
        type HmacSha256 = Hmac<Sha256>;
        
        // Use a static secret for context signing proof
        let secret = b"cuemap-context-signature-secret-v1";
        let mut mac = <HmacSha256 as Mac>::new_from_slice(secret).expect("HMAC can take key of any size");
        mac.update(data.as_bytes());
        let result = mac.finalize();
        hex::encode(result.into_bytes())
    }
}
