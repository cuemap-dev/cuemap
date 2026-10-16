    use super::*;
    use ConversationRole::*;

    fn speaker(name: &str) -> ConversationRole {
        Speaker(name.to_string())
    }

    #[test]
    fn expansion_depth_accepts_numbers_and_conversation_round() {
        assert_eq!("3".parse::<ExpansionDepth>().unwrap(), ExpansionDepth::Depth(3));
        assert_eq!(
            "conversation-round".parse::<ExpansionDepth>().unwrap(),
            ExpansionDepth::Mode(ExpansionMode::ConversationRound)
        );
        assert!("round".parse::<ExpansionDepth>().is_err());

        let number: ExpansionDepth = serde_json::from_str("10").unwrap();
        assert_eq!(number, ExpansionDepth::Depth(10));
        assert_eq!(number.engine_depth(), 10);
        let mode: ExpansionDepth = serde_json::from_str("\"conversation_round\"").unwrap();
        assert!(mode.conversation_round());
        assert_eq!(mode.engine_depth(), 1);
        assert_eq!(serde_json::to_string(&mode).unwrap(), "\"conversation_round\"");
        assert!(serde_json::from_str::<ExpansionDepth>("\"sideways\"").is_err());
    }

    #[test]
    fn conversation_round_is_the_default_expansion() {
        let default = ExpansionDepth::default();
        assert!(default.conversation_round());
        assert_eq!(default.engine_depth(), 1);
        assert_eq!(ROUND_REPLY_TOP, 1);
    }

    #[test]
    fn roles_are_classified_with_unknown_names_as_speakers() {
        assert_eq!(classify_role(" Human "), Some(User));
        assert_eq!(classify_role("assistant"), Some(Assistant));
        assert_eq!(classify_role("tool"), Some(Tool));
        assert_eq!(classify_role("Caroline"), Some(speaker("caroline")));
        assert_eq!(classify_role("  "), None);
    }

    #[test]
    fn assistant_replies_to_the_user_message_that_opened_its_round() {
        assert_eq!(round_partner_index(&Assistant, &[Some(User)]), Some(0));
        // A tool loop and an earlier assistant message in the same round are skipped.
        let earlier = [Some(Tool), Some(Assistant), Some(System), Some(User), Some(Assistant)];
        assert_eq!(round_partner_index(&Assistant, &earlier), Some(3));
        // No user message within the lookback, or a message without a role first.
        assert_eq!(round_partner_index(&Assistant, &[Some(Assistant), Some(Tool)]), None);
        assert_eq!(round_partner_index(&Assistant, &[None, Some(User)]), None);
        assert_eq!(round_partner_index(&Assistant, &[]), None);
    }

    #[test]
    fn user_tool_and_system_messages_have_no_partner() {
        for hit in [User, Tool, System] {
            assert_eq!(round_partner_index(&hit, &[Some(Assistant), Some(User)]), None);
        }
    }

    #[test]
    fn speaker_replies_to_the_nearest_message_by_someone_else() {
        let earlier = [Some(speaker("a")), Some(speaker("a")), Some(speaker("b"))];
        assert_eq!(round_partner_index(&speaker("a"), &earlier), Some(2));
        assert_eq!(round_partner_index(&speaker("b"), &earlier), Some(0));
        assert_eq!(round_partner_index(&speaker("a"), &[Some(System), Some(speaker("b"))]), Some(1));
        assert_eq!(round_partner_index(&speaker("a"), &[Some(speaker("a"))]), None);
    }

    #[test]
    fn user_message_gets_the_assistant_reply_of_its_round() {
        assert_eq!(round_reply_index(&User, &[Some(Assistant)]), Some(0));
        // A tool loop before the answer is skipped; the answer that follows it counts.
        assert_eq!(round_reply_index(&User, &[Some(Tool), Some(System), Some(Assistant)]), Some(2));
        // The next user message, or a message without a role, ends the search.
        assert_eq!(round_reply_index(&User, &[Some(User), Some(Assistant)]), None);
        assert_eq!(round_reply_index(&User, &[None, Some(Assistant)]), None);
        assert_eq!(round_reply_index(&User, &[]), None);
        for hit in [Assistant, Tool, System] {
            assert_eq!(round_reply_index(&hit, &[Some(User), Some(Assistant)]), None);
        }
    }

    #[test]
    fn speaker_gets_the_nearest_later_message_by_someone_else() {
        let later = [Some(speaker("a")), Some(speaker("a")), Some(speaker("b"))];
        assert_eq!(round_reply_index(&speaker("a"), &later), Some(2));
        assert_eq!(round_reply_index(&speaker("b"), &later), Some(0));
        assert_eq!(round_reply_index(&speaker("a"), &[Some(Tool), Some(speaker("b"))]), Some(1));
        assert_eq!(round_reply_index(&speaker("a"), &[Some(speaker("a"))]), None);
    }

    #[test]
    fn approx_tokens_match_the_harness_tokenizer() {
        // Same split as Python's re.findall(r"\w+|[^\w\s]", ...).
        assert_eq!(approx_token_count("user: Hi, I'm Max!"), 9);
        assert_eq!(approx_token_count(""), 0);
        let (kept, count) = cap_approx_tokens("one two, three four", 3);
        assert_eq!((kept, count), ("one two,", 3));
        let (kept, count) = cap_approx_tokens("short text", 512);
        assert_eq!((kept, count), ("short text", 2));
    }
