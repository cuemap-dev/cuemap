    use super::*;
    use serde_json::json;

    fn summary(messages: &[ConversationMessage]) -> Vec<(String, usize, String, String)> {
        messages
            .iter()
            .map(|m| (m.conversation_id.clone(), m.turn, m.role.clone(), m.text.clone()))
            .collect()
    }

    fn row(conversation: &str, turn: usize, role: &str, text: &str) -> (String, usize, String, String) {
        (conversation.to_string(), turn, role.to_string(), text.to_string())
    }

    #[test]
    fn claude_export_keeps_each_conversation_and_text_blocks() {
        let export = json!([
            {"uuid": "c-1", "name": "Dog collar", "chat_messages": [
                {"uuid": "m1", "sender": "human", "text": "Which collar suits a golden retriever?", "created_at": "2024-05-01T10:00:00Z"},
                {"uuid": "m2", "sender": "assistant", "text": "", "content": [
                    {"type": "thinking", "thinking": "hidden"},
                    {"type": "text", "text": "A flat leather collar."},
                    {"type": "tool_use", "name": "search"}
                ]}
            ]},
            {"uuid": "c-2", "chat_messages": [
                {"uuid": "m3", "sender": "human", "text": "Second chat"}
            ]}
        ]);
        let messages = parse_conversations(&export.to_string(), "upload").unwrap();
        assert_eq!(summary(&messages), vec![
            row("c-1", 0, "user", "Which collar suits a golden retriever?"),
            row("c-1", 1, "assistant", "A flat leather collar."),
            row("c-2", 0, "user", "Second chat"),
        ]);
        assert_eq!(messages[0].title.as_deref(), Some("Dog collar"));
        assert_eq!(messages[0].timestamp.as_deref(), Some("2024-05-01T10:00:00Z"));
        assert_eq!(messages[1].message_id.as_deref(), Some("m2"));
    }

    #[test]
    fn claude_export_follows_the_current_leaf_when_messages_branch() {
        let export = json!({"uuid": "c-9", "current_leaf_message_uuid": "a2", "chat_messages": [
            {"uuid": "u1", "sender": "human", "text": "Question"},
            {"uuid": "a1", "sender": "assistant", "text": "First answer", "parent_message_uuid": "u1"},
            {"uuid": "a2", "sender": "assistant", "text": "Regenerated answer", "parent_message_uuid": "u1"}
        ]});
        let messages = parse_conversations(&export.to_string(), "upload").unwrap();
        assert_eq!(summary(&messages), vec![
            row("c-9", 0, "user", "Question"),
            row("c-9", 1, "assistant", "Regenerated answer"),
        ]);
    }

    #[test]
    fn chatgpt_export_keeps_the_displayed_branch_without_tools_or_system() {
        let node = |id: &str, parent: Option<&str>, children: &[&str], message: serde_json::Value| {
            json!({"id": id, "parent": parent, "children": children, "message": message})
        };
        let msg = |role: &str, text: &str, extra: serde_json::Value| {
            let mut message = json!({"id": format!("msg-{text}"), "author": {"role": role}, "create_time": 1714557600.5,
                "content": {"content_type": "text", "parts": [text]}, "recipient": "all"});
            if let (Some(target), Some(source)) = (message.as_object_mut(), extra.as_object()) {
                for (key, value) in source { target.insert(key.clone(), value.clone()); }
            }
            message
        };
        let export = json!([{
            "id": "g-1", "title": "Trip", "current_node": "n6",
            "mapping": {
                "root": node("root", None, &["n0"], serde_json::Value::Null),
                "n0": node("n0", Some("root"), &["n1"], msg("system", "be helpful", json!({}))),
                "n1": node("n1", Some("n0"), &["n2"], msg("user", "Plan a trip to Lisbon", json!({}))),
                "n2": node("n2", Some("n1"), &["n3", "n3b"], msg("assistant", "searching", json!({"recipient": "browser"}))),
                "n3": node("n3", Some("n2"), &["n4"], msg("tool", "search results", json!({}))),
                "n3b": node("n3b", Some("n2"), &[], msg("assistant", "Old branch answer", json!({}))),
                "n4": node("n4", Some("n3"), &["n5"], msg("assistant", "Day one: Alfama.", json!({}))),
                "n5": node("n5", Some("n4"), &["n6"], msg("user", "context", json!({"metadata": {"is_visually_hidden_from_conversation": true}}))),
                "n6": node("n6", Some("n5"), &[], msg("user", "Add a food tour", json!({})))
            }
        }]);
        let messages = parse_conversations(&export.to_string(), "upload").unwrap();
        assert_eq!(summary(&messages), vec![
            row("g-1", 0, "user", "Plan a trip to Lisbon"),
            row("g-1", 1, "assistant", "Day one: Alfama."),
            row("g-1", 2, "user", "Add a food tour"),
        ]);
        assert_eq!(messages[0].timestamp.as_deref(), Some("2024-05-01T10:00:00+00:00"));
        assert_eq!(messages[0].title.as_deref(), Some("Trip"));
    }

    #[test]
    fn json_message_lists_use_the_default_conversation() {
        let list = json!([
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "assistant", "content": "Hello!"}
        ]);
        let messages = parse_conversations(&list.to_string(), "api-log").unwrap();
        assert_eq!(summary(&messages), vec![row("api-log", 0, "user", "Hi"), row("api-log", 1, "assistant", "Hello!")]);
        let wrapped = json!({"id": "thread-4", "messages": [{"speaker": "Alice", "text": "Lunch?"}]});
        let messages = parse_conversations(&wrapped.to_string(), "api-log").unwrap();
        assert_eq!(summary(&messages), vec![row("thread-4", 0, "Alice", "Lunch?")]);
    }

    #[test]
    fn role_transcripts_only_split_on_role_labels() {
        let transcript = "User: My dog is Max.\nHe is a golden retriever.\nAssistant: Nice!\nNote: walk him daily.\nHuman: Thanks\n";
        let messages = parse_conversations(transcript, "paste").unwrap();
        assert_eq!(summary(&messages), vec![
            row("paste", 0, "user", "My dog is Max.\nHe is a golden retriever."),
            row("paste", 1, "assistant", "Nice!\nNote: walk him daily."),
            row("paste", 2, "user", "Thanks"),
        ]);
    }

    #[test]
    fn speaker_transcripts_need_repeating_speakers() {
        let chat = "Caroline: I went to a support group.\nMelanie: How was it?\nCaroline: Powerful.\n";
        let messages = parse_conversations(chat, "dm").unwrap();
        assert_eq!(summary(&messages), vec![
            row("dm", 0, "Caroline", "I went to a support group."),
            row("dm", 1, "Melanie", "How was it?"),
            row("dm", 2, "Caroline", "Powerful."),
        ]);
        for prose in [
            "Note: this is a document.\nExample: not a chat.\nWarning: still not.",
            "Intro line first.\nAlice: hi\nBob: hello\nAlice: bye",
            "Just text with a time 10:30 and a link https://example.com",
        ] {
            assert!(parse_conversations(prose, "doc").is_err(), "{prose}");
        }
        assert!(parse_conversations("not json {", "doc").is_err());
        assert!(parse_conversations("{\"title\": \"no messages\"}", "doc").is_err());
    }

    #[test]
    fn long_messages_become_ordered_parts_with_metadata() {
        let long = format!("{}\n\n{}", "a".repeat(30), "b ".repeat(20).trim());
        let transcript = format!("User: {long}\nAssistant: short");
        let chunks = segment_conversations(&transcript, "paste", 32).unwrap();
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].content, format!("user: {}", "a".repeat(30)));
        assert!(chunks[1].content.starts_with("user: b b"));
        assert_eq!(chunks[0].metadata["source_part_index"], 0);
        assert_eq!(chunks[2].metadata["source_part_index"], 2);
        assert_eq!(chunks[2].metadata["source_part_count"], 3);
        assert_eq!(chunks[2].metadata["source_turn_index"], 0);
        let answer = &chunks[3];
        assert_eq!(answer.content, "assistant: short");
        assert_eq!(answer.metadata["source_session_id"], "paste");
        assert_eq!(answer.metadata["source_turn_index"], 1);
        assert_eq!(answer.metadata["source_role"], "assistant");
        assert!(answer.metadata.get("source_part_index").is_none());
        assert_eq!(answer.structural_cues, vec!["source_type:chat_message", "source_role:assistant"]);
        assert_eq!(answer.key, "paste/1/0");
        // Every chunk key is distinct, even for repeated text.
        let repeated = segment_conversations("User: ok\nAssistant: ok\nUser: ok", "p", 100).unwrap();
        let keys: std::collections::HashSet<_> = repeated.iter().map(|chunk| chunk.key.clone()).collect();
        assert_eq!(keys.len(), 3);
    }

    #[test]
    fn split_message_prefers_natural_boundaries() {
        assert_eq!(split_message("short", 10), vec!["short"]);
        assert_eq!(split_message("One. Two. Three.", 10), vec!["One. Two.", "Three."]);
        assert_eq!(split_message("abcdefghij", 4), vec!["abcd", "efgh", "ij"]);
        assert_eq!(split_message("line one\nline two", 9), vec!["line one", "line two"]);
    }
