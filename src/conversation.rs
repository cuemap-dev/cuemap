//! Conversation structure used at recall time.
//!
//! A memory is part of a conversation when it carries a session key and a message
//! order (`source_session_id` / `source_turn_index` and their aliases, see
//! `CueMapEngine::source_order_for_memory`) plus a role (`source_role` and aliases).
//! Memories that share a session and an order are parts of one message.
//! Nothing here looks at memory text or at ingestion order: without that metadata,
//! the conversation features are a no-op.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Most messages to look back through when searching for the message a result replies to.
pub const ROUND_LOOKBACK_MESSAGES: usize = 32;
/// Most source-order entries (message parts) scanned for that lookback.
pub const ROUND_LOOKBACK_ENTRIES: usize = 256;
/// The attached message is cut after this many approx-tokens.
pub const ROUND_PARTNER_TOKEN_CAP: usize = 512;
/// How many of the top results also get the message that replies to them, by default
/// (`CUEMAP_ROUND_REPLY_TOP` overrides it; 0 turns it off). One keeps the context small
/// while the best match still comes back as a whole exchange.
pub const ROUND_REPLY_TOP: usize = 1;

/// `expansion_depth` in recall requests: a number (1 = no expansion; n > 1 adds
/// neighbours within radius n - 1 in both directions), or a named mode. The default is
/// `conversation_round`, which leaves memories without conversation metadata unchanged,
/// so it behaves like 1 for them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ExpansionDepth {
    Depth(usize),
    Mode(ExpansionMode),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpansionMode {
    /// Attach the message each result replies to (see [`round_partner_index`]).
    ConversationRound,
}

impl Default for ExpansionDepth {
    fn default() -> Self {
        Self::Mode(ExpansionMode::ConversationRound)
    }
}

impl From<usize> for ExpansionDepth {
    fn from(depth: usize) -> Self {
        Self::Depth(depth)
    }
}

impl ExpansionDepth {
    /// Depth handed to the engine's neighbour expansion. Named modes run after ranking
    /// instead, so the engine sees 1 (no neighbour expansion).
    pub fn engine_depth(self) -> usize {
        match self {
            Self::Depth(depth) => depth,
            Self::Mode(_) => 1,
        }
    }

    pub fn conversation_round(self) -> bool {
        matches!(self, Self::Mode(ExpansionMode::ConversationRound))
    }
}

impl FromStr for ExpansionDepth {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.trim();
        if let Ok(depth) = value.parse::<usize>() {
            return Ok(Self::Depth(depth));
        }
        match value.to_ascii_lowercase().replace('-', "_").as_str() {
            "conversation_round" => Ok(Self::Mode(ExpansionMode::ConversationRound)),
            _ => Err(format!(
                "expansion depth must be a number or \"conversation_round\", got {value:?}"
            )),
        }
    }
}

impl fmt::Display for ExpansionDepth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Depth(depth) => write!(f, "{depth}"),
            Self::Mode(ExpansionMode::ConversationRound) => f.write_str("conversation_round"),
        }
    }
}

/// Who wrote a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConversationRole {
    User,
    Assistant,
    Tool,
    System,
    /// A named participant in a human-to-human conversation.
    Speaker(String),
}

/// Maps a stored role value to a [`ConversationRole`]. Unknown names are speakers.
pub fn classify_role(raw: &str) -> Option<ConversationRole> {
    let role = raw.trim().to_lowercase();
    if role.is_empty() {
        return None;
    }
    Some(match role.as_str() {
        "user" | "human" => ConversationRole::User,
        "assistant" | "ai" | "model" | "bot" => ConversationRole::Assistant,
        "tool" | "function" => ConversationRole::Tool,
        "system" | "developer" => ConversationRole::System,
        _ => ConversationRole::Speaker(role),
    })
}

/// The message a message replies to.
///
/// `earlier` holds the roles of the preceding messages in the same conversation,
/// nearest first; `None` marks a message without a role and stops the search.
/// Returns an index into `earlier`.
///
/// - An assistant message replies to the user message that opened its round: earlier
///   assistant, tool and system messages are skipped (a tool loop between the prompt
///   and the answer stays inside the round).
/// - In human-to-human chats, a speaker's message replies to the nearest earlier
///   message by someone else; that speaker's own preceding messages are skipped.
/// - User, tool and system messages have no partner: the user message starts a round.
pub fn round_partner_index(
    hit: &ConversationRole,
    earlier: &[Option<ConversationRole>],
) -> Option<usize> {
    for (index, role) in earlier.iter().enumerate() {
        let role = role.as_ref()?;
        let partner = match (hit, role) {
            (ConversationRole::User | ConversationRole::Tool | ConversationRole::System, _) => {
                return None
            }
            (_, ConversationRole::Tool | ConversationRole::System) => false,
            (ConversationRole::Assistant, ConversationRole::User) => true,
            (ConversationRole::Assistant, ConversationRole::Assistant) => false,
            (ConversationRole::Assistant, ConversationRole::Speaker(_)) => return None,
            (ConversationRole::Speaker(name), ConversationRole::Speaker(other)) => name != other,
            (ConversationRole::Speaker(_), _) => true,
        };
        if partner {
            return Some(index);
        }
    }
    None
}

/// The message that replies to a message: the mirror of [`round_partner_index`].
///
/// `later` holds the roles of the following messages in the same conversation, nearest
/// first; `None` marks a message without a role and stops the search. Returns an index
/// into `later`.
///
/// - A user message's reply is the first assistant message after it; tool and system
///   messages in between are skipped. The next user message ends the round without one.
/// - In human-to-human chats, a speaker's message gets the nearest later message by
///   someone else; that speaker's own following messages are skipped.
/// - Assistant, tool and system messages get no reply.
pub fn round_reply_index(
    hit: &ConversationRole,
    later: &[Option<ConversationRole>],
) -> Option<usize> {
    for (index, role) in later.iter().enumerate() {
        let role = role.as_ref()?;
        let reply = match (hit, role) {
            (ConversationRole::Assistant | ConversationRole::Tool | ConversationRole::System, _) => {
                return None
            }
            (_, ConversationRole::Tool | ConversationRole::System) => false,
            (ConversationRole::User, ConversationRole::Assistant) => true,
            (ConversationRole::User, _) => return None,
            (ConversationRole::Speaker(name), ConversationRole::Speaker(other)) => name != other,
            (ConversationRole::Speaker(_), _) => true,
        };
        if reply {
            return Some(index);
        }
    }
    None
}

/// Approximate token count used by the eval harnesses: each run of word characters
/// is one token, and so is each other non-space character.
pub fn approx_token_count(text: &str) -> usize {
    cap_approx_tokens(text, usize::MAX).1
}

/// Keeps the first `cap` approx-tokens of `text` (see [`approx_token_count`]).
/// Returns the kept text and the number of tokens kept; the text is unchanged when it
/// fits.
pub fn cap_approx_tokens(text: &str, cap: usize) -> (&str, usize) {
    let mut count = 0usize;
    let mut in_word = false;
    for (offset, ch) in text.char_indices() {
        let word = ch.is_alphanumeric() || ch == '_';
        let starts_token = if word { !in_word } else { !ch.is_whitespace() };
        in_word = word;
        if starts_token {
            if count == cap {
                return (text[..offset].trim_end(), count);
            }
            count += 1;
        }
    }
    (text, count)
}

#[cfg(test)]
#[path = "../tests/unit/conversation.rs"]
mod tests;
