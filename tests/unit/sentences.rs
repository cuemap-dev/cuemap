    use super::*;

    fn check(text: &str, expected: &[&str]) {
        assert_eq!(split(text), expected, "{text:?}");
    }

    #[test]
    fn rejoins_hard_wrapped_prose() {
        check(
            "My work is creative. I have a small design agency. My responsibilities \nare making design, execution of documents.",
            &["My work is creative.", "I have a small design agency.", "My responsibilities are making design, execution of documents."],
        );
        check(
            "I live in Ukraine and before the war I worked as Child \npsychologist in usual school and studied at the University.",
            &["I live in Ukraine and before the war I worked as Child psychologist in usual school and studied at the University."],
        );
    }

    #[test]
    fn keeps_lines_lists_and_headings_apart() {
        check(
            "Degree: \nBachelor of Science in Information Technology\n\nSKILLS:\n C19: Fully Vaccinated",
            &["Degree:", "Bachelor of Science in Information Technology", "SKILLS:", "C19: Fully Vaccinated"],
        );
        check(
            "- Flavorful Fish and Seafood Recipes\n- Quick and Tasty Soups",
            &["- Flavorful Fish and Seafood Recipes", "- Quick and Tasty Soups"],
        );
    }

    #[test]
    fn splits_glued_sentences_but_not_abbreviations_or_initials() {
        check(
            "Our small business plan is perfect for busy websites or online stores.SPECIAL PRICE\n$4.95/month",
            &["Our small business plan is perfect for busy websites or online stores.", "SPECIAL PRICE", "$4.95/month"],
        );
        check(
            "My wife, Huan Chen, holder of passport No. EJ, has been accepted. Dr. Smith agreed. See Fig. 3 for details.",
            &["My wife, Huan Chen, holder of passport No.", "EJ, has been accepted.", "Dr. Smith agreed.", "See Fig. 3 for details."],
        );
        check(
            "We moved to the U.S. in 2010. It was great! Really? Yes.",
            &["We moved to the U.S. in 2010.", "It was great!", "Really?", "Yes."],
        );
        check(
            "Nared's bladder is stronger than Jasmine's. They are both very susceptible. J. K. Rowling wrote it.",
            &["Nared's bladder is stronger than Jasmine's.", "They are both very susceptible.", "J. K. Rowling wrote it."],
        );
        check("I said no. Then he left. Use Python 3.10 for this.", &["I said no.", "Then he left.", "Use Python 3.10 for this."]);
    }

    #[test]
    fn trims_unbalanced_quotes_and_drops_code_fences() {
        check(
            "Cyan: [translation: “Anything from Ventoraptor kidneys. But I also tend to eat some farm animals too.”]",
            &["Cyan: [translation: “Anything from Ventoraptor kidneys.", "But I also tend to eat some farm animals too."],
        );
        check(
            "\"I am a hardworking student with a strong passion for computer science. Academics have always been a strength.",
            &["I am a hardworking student with a strong passion for computer science.", "Academics have always been a strength."],
        );
        check(
            "Here is code:\n```python\nprint('x. Y')\n```\nPlease keep every answer under 100 words. Thanks!",
            &["Here is code:", "Please keep every answer under 100 words.", "Thanks!"],
        );
    }

    #[test]
    fn sentences_keeps_three_to_sixty_words_of_mostly_letters() {
        assert_eq!(
            sentences("Ok. I prefer tea over coffee.\n$4.95 / 12 / 30 / 99"),
            vec!["I prefer tea over coffee."]
        );
    }

    /// Parity with the Python splitter that built the training data. Point
    /// CUEMAP_SENTENCE_PARITY at a JSON list of `[text, [sentences...]]` pairs written by
    /// `segment.split`; without it the test does nothing.
    #[test]
    fn matches_the_python_splitter_on_a_fixture() {
        let Ok(path) = std::env::var("CUEMAP_SENTENCE_PARITY") else { return };
        let data: Vec<(String, Vec<String>)> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let mismatches: Vec<_> = data.iter().filter(|(text, expected)| &split(text) != expected).collect();
        for (text, expected) in mismatches.iter().take(5) {
            eprintln!("text {text:?}\n  python {expected:?}\n  rust   {:?}", split(text));
        }
        assert!(mismatches.is_empty(), "{} of {} texts differ", mismatches.len(), data.len());
    }
