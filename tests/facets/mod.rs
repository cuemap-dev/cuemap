use cuemap::engine::CueMapEngine;
use cuemap::facets::{
    compile_query_intent, compile_query_intent_with_reference_time, extract_memory_facets,
};
use cuemap::nl::tokenize_to_cues;
use cuemap::structures::MainStats;
use serde_json::json;
use std::collections::{HashMap, HashSet};

fn compile_weighted_query(engine: &CueMapEngine<MainStats>, query: &str) -> Vec<(String, f64)> {
    compile_weighted_query_at(engine, query, None)
}

fn compile_weighted_query_at(
    engine: &CueMapEngine<MainStats>,
    query: &str,
    reference_time: Option<&str>,
) -> Vec<(String, f64)> {
    let mut weighted_cues: Vec<(String, f64)> = tokenize_to_cues(query)
        .into_iter()
        .map(|cue| (cue, 1.0))
        .collect();
    let total_memories = engine.total_memories().max(1);
    let intent = compile_query_intent_with_reference_time(query, reference_time, |cue| {
        let df = engine.get_cue_frequency(cue);
        df > 0 && (df <= 16 || df * 5 <= total_memories)
    });

    for (cue, multiplier) in &intent.cue_weight_adjustments {
        if let Some((_, weight)) = weighted_cues
            .iter_mut()
            .find(|(existing, _)| existing == cue)
        {
            *weight *= *multiplier;
        }
    }
    for (cue, weight) in &intent.weighted_cues {
        if let Some((_, existing_weight)) = weighted_cues
            .iter_mut()
            .find(|(existing, _)| existing == cue)
        {
            if *existing_weight < *weight {
                *existing_weight = *weight;
            }
        } else {
            weighted_cues.push((cue.clone(), *weight));
        }
    }

    weighted_cues
}

#[test]
fn extracts_general_source_and_evidence_facets() {
    let mut metadata = HashMap::new();
    metadata.insert("role".to_string(), json!("Doctor"));
    metadata.insert("channel".to_string(), json!("support inbox"));
    metadata.insert("source_session_id".to_string(), json!("Case-123"));

    let facets = extract_memory_facets(
        "Doctor: I currently take 20 mg daily for 2 weeks and paid $15 last week.",
        Some(&metadata),
        &[],
    );

    assert!(facets.contains(&"source_role:doctor".to_string()));
    assert!(facets.contains(&"source_channel:support_inbox".to_string()));
    assert!(facets.contains(&"source_session:case_123".to_string()));
    assert!(facets.contains(&"has:number".to_string()));
    assert!(facets.contains(&"has:money".to_string()));
    assert!(facets.contains(&"has:duration".to_string()));
    assert!(facets.contains(&"temporal:current".to_string()));
    assert!(facets.contains(&"temporal:last_week".to_string()));
}

#[test]
fn extracts_source_date_facets_from_metadata_timestamps() {
    let mut metadata = HashMap::new();
    metadata.insert(
        "source_date".to_string(),
        json!("2023/04/21 (Fri) 00:30"),
    );

    let facets = extract_memory_facets(
        "user: I just planted 12 new tomato saplings today.",
        Some(&metadata),
        &[],
    );

    assert!(facets.contains(&"source_time:dated".to_string()));
    assert!(facets.contains(&"source_date:2023_04_21".to_string()));
    assert!(facets.contains(&"source_week:2023_w16".to_string()));
    assert!(facets.contains(&"source_month:2023_04".to_string()));
    assert!(facets.contains(&"source_year:2023".to_string()));
}

#[test]
fn extracts_time_of_day_facets_from_words_and_clock_times() {
    let evening = extract_memory_facets(
        "User: I prefer winding down by 9:30 pm during the later part of the day.",
        None,
        &[],
    );

    assert!(evening.contains(&"has:time".to_string()));
    assert!(evening.contains(&"time_of_day:evening".to_string()));

    let morning = extract_memory_facets(
        "User: I usually exercise at 7:15 am before work.",
        None,
        &[],
    );
    assert!(morning.contains(&"time_of_day:morning".to_string()));
}

#[test]
fn extracts_content_month_facets_from_month_names_and_short_dates() {
    let named = extract_memory_facets(
        "User: I went to the opening night on 15th February.",
        None,
        &[],
    );
    let numeric = extract_memory_facets(
        "User: I took my niece to the Natural History Museum on 2/8.",
        None,
        &[],
    );

    assert!(named.contains(&"has:date".to_string()));
    assert!(named.contains(&"content_month:02".to_string()));
    assert!(numeric.contains(&"has:date".to_string()));
    assert!(numeric.contains(&"content_month:02".to_string()));
}

#[test]
fn extracts_frequency_facets_from_cadence_expressions() {
    let facets = extract_memory_facets(
        "User: I've been doing yoga twice a week, usually after work.",
        None,
        &[],
    );

    assert!(facets.contains(&"has:frequency".to_string()));
    assert!(facets.contains(&"schedule:frequency".to_string()));
    assert!(facets.contains(&"frequency_unit:week".to_string()));
    assert!(facets.contains(&"schedule:weekly".to_string()));

    let hourly = extract_memory_facets(
        "User: I usually work 40 hours per week during peak campaign seasons.",
        None,
        &[],
    );
    assert!(hourly.contains(&"has:frequency".to_string()));
    assert!(hourly.contains(&"frequency_unit:week".to_string()));
}

#[test]
fn compiles_time_of_day_query_intent_from_raw_query_text() {
    let intent = compile_query_intent("Can you suggest activities for the evening?", |cue| {
        matches!(cue, "time_of_day:evening" | "has:time")
    });

    assert!(intent.labels.contains(&"time_of_day".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "time_of_day:evening"));
}

#[test]
fn extracts_type_and_entity_facets_without_benchmark_roles() {
    let facets = extract_memory_facets(
        "Chef: I prefer Sony A7R IV photos, dislike cinnamon, and bought \"Peak Design Bag\".",
        None,
        &[],
    );

    assert!(facets.contains(&"source_role:chef".to_string()));
    assert!(facets.contains(&"type:preference".to_string()));
    assert!(facets.contains(&"type:dislike".to_string()));
    assert!(facets.contains(&"type:ownership".to_string()));
    assert!(facets.contains(&"entity:sony_a7r_iv".to_string()));
    assert!(facets.contains(&"entity:peak_design_bag".to_string()));
}

#[test]
fn extracts_purchase_consideration_from_first_person_planning_language() {
    let upgrade = extract_memory_facets(
        "User: I'm considering upgrading from a Fender Stratocaster to a Gibson Les Paul.",
        None,
        &[],
    );

    assert!(upgrade.contains(&"type:purchase_consideration".to_string()));

    let comparison_question = extract_memory_facets(
        "User: What are the differences between open D tuning and standard tuning?",
        None,
        &[],
    );
    assert!(!comparison_question.contains(&"type:purchase_consideration".to_string()));
}

#[test]
fn extracts_competition_event_facets_from_first_person_sports_events() {
    let soccer = extract_memory_facets(
        "User: I participate in the company's annual charity soccer tournament today.",
        None,
        &[],
    );
    assert!(soccer.contains(&"type:competition_event".to_string()));
    assert!(soccer.contains(&"activity_domain:sport".to_string()));
    assert!(soccer.contains(&"type:activity".to_string()));
    assert!(soccer.contains(&"type:event".to_string()));

    let future_soccer = extract_memory_facets(
        "User: I will participate in the company's annual charity soccer tournament today.",
        None,
        &[],
    );
    assert!(future_soccer.contains(&"type:competition_event".to_string()));
    assert!(future_soccer.contains(&"activity_domain:sport".to_string()));
    assert!(future_soccer.contains(&"type:activity".to_string()));
    assert!(future_soccer.contains(&"type:event".to_string()));

    let planning = extract_memory_facets(
        "User: I might watch a soccer tournament this weekend.",
        None,
        &[],
    );
    assert!(!planning.contains(&"type:competition_event".to_string()));
}

#[test]
fn extracts_activity_event_from_first_person_did_named_event() {
    let charity_walk = extract_memory_facets(
        "User: I just did the \"Walk for Hunger\" charity event today with my colleagues from work.",
        None,
        &[],
    );

    assert!(charity_walk.contains(&"type:activity".to_string()));
    assert!(charity_walk.contains(&"type:event".to_string()));
    assert!(charity_walk.contains(&"event_domain:charity".to_string()));

    let donation = extract_memory_facets(
        "User: I donated to charity online today after reading about the campaign.",
        None,
        &[],
    );
    assert!(!donation.contains(&"event_domain:charity".to_string()));
}

#[test]
fn extracts_wake_time_routine_from_timed_wake_language() {
    let timed = extract_memory_facets(
        "User: I like to wake up at 7:30 am on Saturdays and fit in coffee beforehand.",
        None,
        &[],
    );

    assert!(timed.contains(&"routine:wake_time".to_string()));
    assert!(timed.contains(&"type:routine".to_string()));

    let untimed = extract_memory_facets(
        "User: Do you have any advice for waking up earlier without feeling tired?",
        None,
        &[],
    );
    assert!(!untimed.contains(&"routine:wake_time".to_string()));
}

#[test]
fn extracts_bed_time_routine_from_timed_bed_language() {
    let timed = extract_memory_facets(
        "User: I didn't get to bed until 2 AM last Wednesday.",
        None,
        &[],
    );

    assert!(timed.contains(&"routine:bed_time".to_string()));
    assert!(timed.contains(&"type:routine".to_string()));
    assert!(timed.contains(&"has:time".to_string()));

    let untimed = extract_memory_facets(
        "User: I bought a new bed frame last weekend.",
        None,
        &[],
    );
    assert!(!untimed.contains(&"routine:bed_time".to_string()));
}

#[test]
fn extracts_iteration_facets_from_revised_or_followup_outputs() {
    let revised = extract_memory_facets(
        "Assistant: Sure, here's a more romantic and heart-felt song for you.",
        None,
        &[],
    );
    assert!(revised.contains(&"type:iteration".to_string()));

    let first = extract_memory_facets(
        "Assistant: Here's a sad song with notes for you.",
        None,
        &[],
    );
    assert!(!first.contains(&"type:iteration".to_string()));

    let genre = extract_memory_facets(
        "User: I'm really into indie and alternative rock right now.",
        None,
        &[],
    );
    assert!(!genre.contains(&"type:iteration".to_string()));
}

#[test]
fn extracts_named_entity_attribute_relations() {
    let facets = extract_memory_facets(
        "User: I need a collar that would suit a Golden Retriever like Max.",
        None,
        &[],
    );

    assert!(facets.contains(&"type:entity_attribute".to_string()));
    assert!(facets.contains(&"attribute:class_relation".to_string()));
    assert!(facets.contains(&"entity:golden_retriever".to_string()));
    assert!(facets.contains(&"entity:max".to_string()));
}

#[test]
fn extracts_preferred_attribute_relations_and_canonicalizes_favourite() {
    let facets = extract_memory_facets(
        "User: Nike has been my favourite brand so far for running shoes.",
        None,
        &[],
    );
    let cues = tokenize_to_cues("Nike has been my favourite brand so far for running shoes.");

    assert!(facets.contains(&"type:preference".to_string()));
    assert!(facets.contains(&"type:entity_attribute".to_string()));
    assert!(facets.contains(&"attribute:class_relation".to_string()));
    assert!(cues.contains(&"favorite".to_string()));
    assert!(!cues.contains(&"favourite".to_string()));
}

#[test]
fn extracts_expertise_facets_from_professional_field_language() {
    let facets = extract_memory_facets(
        "User: Can you give me an overview of recent advancements in this field? Skip the basics as I am working in the field.",
        None,
        &[],
    );

    assert!(facets.contains(&"type:expertise".to_string()));
    assert!(facets.contains(&"type:interest".to_string()));

    let service_query_facets = extract_memory_facets(
        "User: Can you suggest contractors or services that specialize in backyard landscaping?",
        None,
        &[],
    );
    assert!(!service_query_facets.contains(&"type:expertise".to_string()));
}

#[test]
fn extracts_inspiration_source_facets_from_explicit_source_language() {
    let facets = extract_memory_facets(
        "User: I have been getting inspiration from social media and recently started a 30-day painting challenge.",
        None,
        &[],
    );

    assert!(facets.contains(&"type:inspiration_source".to_string()));
    assert!(facets.contains(&"type:interest".to_string()));

    let generic_request = extract_memory_facets(
        "User: Do you have any ideas for how I can find new inspiration for my paintings?",
        None,
        &[],
    );
    assert!(!generic_request.contains(&"type:inspiration_source".to_string()));
}

#[test]
fn extracts_decision_selection_facets_from_confirmation_language() {
    let facets = extract_memory_facets(
        "User: Fissionator is a really cool one, especially as a final name for the enemy.",
        None,
        &[],
    );

    assert!(facets.contains(&"type:decision".to_string()));
    assert!(facets.contains(&"type:selection".to_string()));
    assert!(facets.contains(&"type:naming".to_string()));

    let undecided_options = extract_memory_facets(
        "Assistant: Here are some possible names: Radik, Irradon, Nucleus, and Fissionator.",
        None,
        &[],
    );
    assert!(!undecided_options.contains(&"type:decision".to_string()));
    assert!(!undecided_options.contains(&"type:selection".to_string()));
}

#[test]
fn extracts_activity_event_facets_from_first_person_completed_actions() {
    let activity = extract_memory_facets(
        "User: I just planted 12 new tomato saplings today.",
        None,
        &[],
    );
    assert!(activity.contains(&"type:activity".to_string()));
    assert!(activity.contains(&"type:event".to_string()));

    let planning = extract_memory_facets(
        "User: I am thinking about planting tomatoes next month.",
        None,
        &[],
    );
    assert!(!planning.contains(&"type:activity".to_string()));
    assert!(!planning.contains(&"type:event".to_string()));

    let intention = extract_memory_facets(
        "User: I wanted to create the trip plan along with dinner suggestions in the same summary.",
        None,
        &[],
    );
    assert!(!intention.contains(&"type:activity".to_string()));
    assert!(!intention.contains(&"type:event".to_string()));
}

#[test]
fn extracts_activity_event_facets_from_experience_phrases() {
    let got_back = extract_memory_facets(
        "User: I just got back from my friend's wedding last weekend.",
        None,
        &[],
    );
    let been_to = extract_memory_facets(
        "User: I've been to a few galleries recently.",
        None,
        &[],
    );

    assert!(got_back.contains(&"type:activity".to_string()));
    assert!(got_back.contains(&"type:event".to_string()));
    assert!(been_to.contains(&"type:activity".to_string()));
    assert!(been_to.contains(&"type:event".to_string()));
}

#[test]
fn extracts_religious_activity_facets_from_religious_service_events() {
    let service = extract_memory_facets(
        "User: I attended the Maundy Thursday service at the Episcopal Church.",
        None,
        &[],
    );
    assert!(service.contains(&"topic:religion".to_string()));
    assert!(service.contains(&"activity_domain:religion".to_string()));
    assert!(service.contains(&"type:activity".to_string()));
    assert!(service.contains(&"type:event".to_string()));

    let customer_service = extract_memory_facets(
        "User: I called customer service about my subscription renewal.",
        None,
        &[],
    );
    assert!(!customer_service.contains(&"activity_domain:religion".to_string()));

    let topic_discussion = extract_memory_facets(
        "User: How does the Tripitaka influence Theravada Buddhist worship and practice?",
        None,
        &[],
    );
    assert!(topic_discussion.contains(&"topic:religion".to_string()));
    assert!(!topic_discussion.contains(&"activity_domain:religion".to_string()));
}

#[test]
fn extracts_media_streaming_usage_facets_from_watch_history() {
    let long_term_services = extract_memory_facets(
        "User: I've been using Netflix, Hulu, and Amazon Prime for the past 6 months while looking for new shows to watch.",
        None,
        &[],
    );
    assert!(long_term_services.contains(&"media:watching".to_string()));
    assert!(long_term_services.contains(&"media:streaming".to_string()));
    assert!(long_term_services.contains(&"type:usage".to_string()));

    let free_trial = extract_memory_facets(
        "User: I saw a documentary on Disney+ during my free trial last month.",
        None,
        &[],
    );
    assert!(free_trial.contains(&"media:watching".to_string()));
    assert!(free_trial.contains(&"media:streaming".to_string()));
    assert!(free_trial.contains(&"type:usage".to_string()));

    let comedy_special = extract_memory_facets(
        "User: Can you recommend some stand-up comedy specials on Netflix with strong storytelling?",
        None,
        &[],
    );
    assert!(comedy_special.contains(&"media:watching".to_string()));

    let music = extract_memory_facets(
        "User: I've been listening to their songs a lot on Spotify lately.",
        None,
        &[],
    );
    assert!(music.contains(&"media:music".to_string()));
    assert!(music.contains(&"media:music_streaming".to_string()));
    assert!(music.contains(&"media:streaming".to_string()));
    assert!(music.contains(&"type:usage".to_string()));

    let live_show = extract_memory_facets(
        "User: Have they been playing any new songs or focusing on their older material?",
        None,
        &[],
    );
    assert!(!live_show.contains(&"media:music_streaming".to_string()));

    let writing = extract_memory_facets(
        "User: I want to practice using vivid language in my essays.",
        None,
        &[],
    );
    assert!(!writing.contains(&"media:streaming".to_string()));
    assert!(!writing.contains(&"type:usage".to_string()));
}

#[test]
fn extracts_current_book_reading_facets_from_first_person_state() {
    let current = extract_memory_facets(
        "User: I'm currently devouring \"The Seven Husbands of Evelyn Hugo\" and it's hard to put down.",
        None,
        &[],
    );
    assert!(current.contains(&"reading:current".to_string()));
    assert!(current.contains(&"media:book_reading".to_string()));
    assert!(current.contains(&"media:book".to_string()));
    assert!(current.contains(&"temporal:current".to_string()));

    let old = extract_memory_facets(
        "User: We're going to discuss \"The Last House Guest\", which I've already read and enjoyed.",
        None,
        &[],
    );
    assert!(!old.contains(&"reading:current".to_string()));
}

#[test]
fn extracts_transport_event_facets_from_real_travel_phrases() {
    let bus = extract_memory_facets(
        "User: I just got back from a bus ride to attend a friend's wedding today.",
        None,
        &[],
    );
    assert!(bus.contains(&"transport_mode:bus".to_string()));
    assert!(bus.contains(&"transport_event:bus".to_string()));
    assert!(bus.contains(&"type:activity".to_string()));

    let train = extract_memory_facets(
        "User: I took a train ride to visit my family today.",
        None,
        &[],
    );
    assert!(train.contains(&"transport_mode:train".to_string()));
    assert!(train.contains(&"transport_event:train".to_string()));

    let general = extract_memory_facets(
        "User: I've been taking more trains and buses instead of driving.",
        None,
        &[],
    );
    assert!(general.contains(&"transport_mode:train".to_string()));
    assert!(general.contains(&"transport_mode:bus".to_string()));
    assert!(!general.contains(&"transport_event:train".to_string()));
    assert!(!general.contains(&"transport_event:bus".to_string()));
}

#[test]
fn extracts_milestone_facets_from_real_milestone_language() {
    let first_client = extract_memory_facets(
        "User: I just signed a contract with my first client today.",
        None,
        &[],
    );
    assert!(first_client.contains(&"type:activity".to_string()));
    assert!(first_client.contains(&"type:event".to_string()));
    assert!(first_client.contains(&"type:milestone".to_string()));

    let contract_advice = extract_memory_facets(
        "Assistant: A well-drafted contract should include payment terms and scope of work.",
        None,
        &[],
    );
    assert!(!contract_advice.contains(&"type:milestone".to_string()));
}

#[test]
fn extracts_ownership_from_first_person_acquisition_language() {
    let acquired = extract_memory_facets(
        "User: I just got a smoker today and I'm excited to experiment with it.",
        None,
        &[],
    );
    assert!(acquired.contains(&"type:ownership".to_string()));
    assert!(acquired.contains(&"type:activity".to_string()));
    assert!(acquired.contains(&"type:event".to_string()));
    assert!(acquired.contains(&"purchase:acquired".to_string()));

    let recipient_acquired = extract_memory_facets(
        "User: For my sister's birthday, I got her a yellow dress and a pair of earrings to match.",
        None,
        &[],
    );
    assert!(recipient_acquired.contains(&"type:ownership".to_string()));
    assert!(recipient_acquired.contains(&"purchase:acquired".to_string()));

    let sourced = extract_memory_facets(
        "User: I'm happy with my new tennis racket, which I got from a sports store downtown.",
        None,
        &[],
    );
    assert!(sourced.contains(&"type:ownership".to_string()));
    assert!(sourced.contains(&"purchase:source".to_string()));

    let source_statement = extract_memory_facets(
        "User: The new bookshelf is from IKEA, and I'm really happy with it.",
        None,
        &[],
    );
    assert!(source_statement.contains(&"type:ownership".to_string()));
    assert!(source_statement.contains(&"purchase:source".to_string()));

    let got_to = extract_memory_facets(
        "User: I got to see a great concert last night.",
        None,
        &[],
    );
    assert!(!got_to.contains(&"type:ownership".to_string()));
}

#[test]
fn extracts_ownership_from_possession_use_and_sale_language() {
    let long_term = extract_memory_facets(
        "User: I've had my acoustic guitar, a Yamaha FG800, for about 8 years.",
        None,
        &[],
    );
    assert!(long_term.contains(&"type:ownership".to_string()));
    assert!(long_term.contains(&"inventory_object:acoustic".to_string()));
    assert!(long_term.contains(&"inventory_object:guitar".to_string()));

    let active_use = extract_memory_facets(
        "User: I've been playing my black Fender Stratocaster electric guitar a lot lately.",
        None,
        &[],
    );
    assert!(active_use.contains(&"type:ownership".to_string()));
    assert!(active_use.contains(&"inventory_object:fender".to_string()));
    assert!(active_use.contains(&"inventory_object:guitar".to_string()));

    let sale = extract_memory_facets(
        "User: I'm thinking of selling my old drum set, a 5-piece Pearl Export.",
        None,
        &[],
    );
    assert!(sale.contains(&"type:ownership".to_string()));
    assert!(sale.contains(&"inventory_object:drum".to_string()));
    assert!(sale.contains(&"inventory_object:set".to_string()));

    let appositive = extract_memory_facets(
        "User: I need to service my Korg B1, which I've had for about 3 years.",
        None,
        &[],
    );
    assert!(appositive.contains(&"type:ownership".to_string()));
    assert!(appositive.contains(&"inventory_object:korg".to_string()));
    assert!(appositive.contains(&"inventory_object:b1".to_string()));
}

#[test]
fn auxiliary_have_and_got_do_not_create_ownership_facets() {
    let done = extract_memory_facets(
        "User: I think I've got everything I need. Thanks for the help!",
        None,
        &[],
    );
    assert!(!done.contains(&"type:ownership".to_string()));
    assert!(!done.iter().any(|facet| facet.starts_with("inventory_object:")));

    let heard = extract_memory_facets(
        "User: I have heard that Disney has faced criticism for its changes.",
        None,
        &[],
    );
    assert!(!heard.contains(&"type:ownership".to_string()));
    assert!(!heard.iter().any(|facet| facet.starts_with("inventory_object:")));

    let concrete = extract_memory_facets(
        "User: I currently have a Korg B1 digital piano in my studio.",
        None,
        &[],
    );
    assert!(concrete.contains(&"type:ownership".to_string()));
    assert!(concrete.contains(&"inventory_object:korg".to_string()));
    assert!(concrete.contains(&"inventory_object:b1".to_string()));
}

#[test]
fn extracts_homegrown_ingredient_facets_from_real_relations() {
    let facets = extract_memory_facets(
        "User: I've been using basil and mint in my cooking lately. I've even harvested some cherry tomatoes from my garden.",
        None,
        &[],
    );

    assert!(facets.contains(&"type:ingredient".to_string()));
    assert!(facets.contains(&"type:homegrown".to_string()));

    let unrelated = extract_memory_facets(
        "User: I'm looking for inspiration for new cocktail ingredients this weekend.",
        None,
        &[],
    );

    assert!(unrelated.contains(&"type:ingredient".to_string()));
    assert!(!unrelated.contains(&"type:homegrown".to_string()));
}

#[test]
fn extracts_list_facets_from_inline_numbered_lists() {
    let facets = extract_memory_facets(
        "Assistant: 1. Virtual customer service representative 2. Remote bookkeeper 3. Transcriptionist 4. Social media manager",
        None,
        &[],
    );

    assert!(facets.contains(&"has:list".to_string()));
}

#[test]
fn extracts_navigation_facets_from_routes_transit_passes_and_apps() {
    let route = extract_memory_facets(
        "How do I get to Shinjuku Station from Narita Airport using my Suica card?",
        None,
        &[],
    );
    assert!(route.contains(&"type:navigation".to_string()));
    assert!(route.contains(&"travel:route".to_string()));
    assert!(route.contains(&"travel:station".to_string()));
    assert!(route.contains(&"travel:pass".to_string()));

    let transit = extract_memory_facets(
        "Take the train from Union Station, transfer to the metro, and check the fare before you go.",
        None,
        &[],
    );
    assert!(transit.contains(&"type:navigation".to_string()));
    assert!(transit.contains(&"travel:transit".to_string()));
    assert!(transit.contains(&"travel:station".to_string()));
    assert!(transit.contains(&"travel:fare".to_string()));

    let app = extract_memory_facets(
        "I downloaded a travel app to keep my tour meeting point and itinerary organized.",
        None,
        &[],
    );
    assert!(app.contains(&"type:navigation".to_string()));
    assert!(app.contains(&"travel:route".to_string()));
    assert!(app.contains(&"travel:app".to_string()));
}

#[test]
fn generic_location_recommendations_do_not_get_navigation_facets() {
    let facets = extract_memory_facets(
        "Can you recommend some good restaurants near the Park Hyatt Tokyo?",
        None,
        &[],
    );

    assert!(!facets.contains(&"type:navigation".to_string()));
    assert!(!facets.iter().any(|facet| facet.starts_with("travel:")));
}

#[test]
fn extracts_age_and_education_facets_from_structured_age_language() {
    let current_age = extract_memory_facets(
        "As a 32-year-old Digital Marketing Specialist, I'm considering an MBA.",
        None,
        &[],
    );
    assert!(current_age.contains(&"has:age".to_string()));
    assert!(current_age.contains(&"age:current".to_string()));
    assert!(!current_age.contains(&"age:event".to_string()));

    let graduation_age = extract_memory_facets(
        "I have a Bachelor's degree from the University of California, which I completed at the age of 25.",
        None,
        &[],
    );
    assert!(graduation_age.contains(&"has:age".to_string()));
    assert!(graduation_age.contains(&"age:event".to_string()));
    assert!(graduation_age.contains(&"education:degree".to_string()));
    assert!(graduation_age.contains(&"education:college".to_string()));
    assert!(graduation_age.contains(&"education:graduation".to_string()));
    assert!(graduation_age.contains(&"education:undergraduate".to_string()));
}

#[test]
fn extracts_undergraduate_education_facets_from_undergrad_language() {
    let facets = extract_memory_facets(
        "User: I completed my undergrad in CS from UCLA before moving to Seattle.",
        None,
        &[],
    );

    assert!(facets.contains(&"education:degree".to_string()));
    assert!(facets.contains(&"education:undergraduate".to_string()));
    assert!(facets.contains(&"education:graduation".to_string()));
    assert!(facets.contains(&"entity:cs".to_string()));
    assert!(facets.contains(&"entity:ucla".to_string()));
}

#[test]
fn bachelors_degree_query_targets_undergraduate_facets_and_initialisms() {
    let available = |cue: &str| {
        matches!(
            cue,
            "education:degree"
                | "education:undergraduate"
                | "education:graduation"
                | "source_role:user"
                | "entity:computer_science"
                | "entity:cs"
        )
    };

    let intent = compile_query_intent(
        "Where did I complete my Bachelor's degree in Computer Science?",
        available,
    );

    assert!(intent.labels.contains(&"education_query".to_string()));
    for expected in [
        "education:degree",
        "education:undergraduate",
        "education:graduation",
        "source_role:user",
        "entity:computer_science",
        "entity:cs",
    ] {
        assert!(
            intent.weighted_cues.iter().any(|(cue, _)| cue == expected),
            "missing education query cue {expected}"
        );
    }
}

#[test]
fn extracts_family_relation_facets_from_self_scoped_sibling_facts() {
    let sisters = extract_memory_facets(
        "I come from a family with 3 sisters, so I have always had a strong female presence in my life.",
        None,
        &[],
    );
    assert!(sisters.contains(&"has:number".to_string()));
    assert!(sisters.contains(&"family_relation:sibling".to_string()));
    assert!(sisters.contains(&"sibling_kind:sister".to_string()));
    assert!(sisters.contains(&"family_scope:self".to_string()));
    assert!(sisters.contains(&"family_count:sibling".to_string()));

    let brother = extract_memory_facets(
        "I should mention that I have a brother, which might influence my social circle dynamics.",
        None,
        &[],
    );
    assert!(brother.contains(&"family_relation:sibling".to_string()));
    assert!(brother.contains(&"sibling_kind:brother".to_string()));
    assert!(brother.contains(&"family_scope:self".to_string()));
    assert!(brother.contains(&"family_count:sibling".to_string()));

    let movie = extract_memory_facets(
        "The film follows twin siblings who uncover a family secret.",
        None,
        &[],
    );
    assert!(movie.contains(&"family_relation:sibling".to_string()));
    assert!(!movie.contains(&"family_scope:self".to_string()));
    assert!(!movie.contains(&"family_count:sibling".to_string()));
}

#[test]
fn extracts_co_residence_facets_from_staying_with_self_language() {
    let facets = extract_memory_facets(
        "User: My parents have been a big help; they've been staying with me for nine months now.",
        None,
        &[],
    );

    assert!(facets.contains(&"family_relation:parent".to_string()));
    assert!(facets.contains(&"co_residence:with_self".to_string()));
    assert!(facets.contains(&"has:duration".to_string()));
}

#[test]
fn extracts_update_facets_from_discourse_markers() {
    let correction =
        extract_memory_facets("I'm actually planning to stay on Oahu instead.", None, &[]);
    let switch = extract_memory_facets(
        "I have just wrapped up a model and switched to a Ford F-150 pickup truck.",
        None,
        &[],
    );

    assert!(correction.contains(&"type:update".to_string()));
    assert!(switch.contains(&"type:update".to_string()));
}

#[test]
fn does_not_extract_question_words_or_source_labels_as_entities() {
    let facets = extract_memory_facets(
        "User: What breed is Max? Any tips? Assistant: Max is a Golden Retriever.",
        None,
        &[],
    );

    assert!(facets.contains(&"source_role:user".to_string()));
    assert!(facets.contains(&"entity:max".to_string()));
    assert!(facets.contains(&"entity:golden_retriever".to_string()));
    assert!(!facets.contains(&"entity:what".to_string()));
    assert!(!facets.contains(&"entity:any".to_string()));
    assert!(!facets.contains(&"entity:user".to_string()));
    assert!(!facets.contains(&"entity:assistant".to_string()));
}

#[test]
fn query_intent_does_not_treat_stopword_sentence_openers_as_entities() {
    let available = |cue: &str| matches!(cue, "entity:any" | "entity:max");
    let intent = compile_query_intent("Any tips for Max?", available);

    assert!(!intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "entity:any"));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "entity:max"));
}

#[test]
fn extracts_person_role_facets_from_titles_and_role_phrases() {
    let facets = extract_memory_facets(
        "I saw Dr. Patel for chronic sinusitis. My primary care physician Dr. Smith prescribed antibiotics, dermatologist Dr. Lee handled the biopsy, and project manager Alice approved the plan.",
        None,
        &[],
    );

    assert!(facets.contains(&"person_title:dr".to_string()));
    assert!(facets.contains(&"person_ref:named".to_string()));
    assert!(facets.contains(&"person_role_phrase:primary_care_physician".to_string()));
    assert!(facets.contains(&"person_role_phrase:dermatologist".to_string()));
    assert!(facets.contains(&"person_role_phrase:project_manager".to_string()));
    let role_phrases = facets
        .iter()
        .filter(|facet| facet.starts_with("person_role_phrase:"))
        .cloned()
        .collect::<HashSet<_>>();
    assert_eq!(
        role_phrases,
        HashSet::from([
            "person_role_phrase:dermatologist".to_string(),
            "person_role_phrase:primary_care_physician".to_string(),
            "person_role_phrase:project_manager".to_string(),
        ])
    );
}

#[test]
fn person_role_facets_reject_clause_false_positives() {
    for content in [
        "I cut the wood with a saw.",
        "I cut the wood with a saw before Maya arrived.",
        "I cut wooden boards with Alice.",
        "I used a circular power saw beside Maya.",
        "My friend saw Dr. Patel yesterday.",
        "My neighbor visited Dr. Smith last week.",
        "Ensure we only search the remaining HTML.",
    ] {
        let facets = extract_memory_facets(content, None, &[]);
        assert!(
            !facets
                .iter()
                .any(|facet| facet.starts_with("person_role_phrase:")),
            "false role facet for {content:?}: {facets:?}"
        );
    }

    let facets = extract_memory_facets("I met project manager Alice yesterday.", None, &[]);
    assert!(
        !facets.contains(&"person_role_phrase:met_project_manager".to_string()),
        "clause verb leaked into role phrase: {facets:?}"
    );
}

#[test]
fn person_role_facets_preserve_content_words_and_complete_phrases() {
    for (content, expected) in [
        (
            "Circular saw specialist Maya arrived.",
            "person_role_phrase:circular_saw_specialist",
        ),
        (
            "Head of product Maya approved the launch.",
            "person_role_phrase:head_of_product",
        ),
        (
            "Chief information security officer Dr. Smith approved it.",
            "person_role_phrase:chief_information_security_officer",
        ),
        (
            "My director of operations Dr. Smith approved it.",
            "person_role_phrase:director_of_operations",
        ),
    ] {
        let facets = extract_memory_facets(content, None, &[]);
        assert!(
            facets.contains(&expected.to_string()),
            "missing {expected} for {content:?}: {facets:?}"
        );
    }
}

#[test]
fn extracts_numeric_object_facets_from_quantity_syntax() {
    let facets = extract_memory_facets(
        "I currently have 2 monitors, 3 laptops, and a 20-gallon freshwater community tank named Amazonia.",
        None,
        &[],
    );

    assert!(facets.contains(&"type:ownership".to_string()));
    assert!(facets.contains(&"quantity_object:monitor".to_string()));
    assert!(facets.contains(&"quantity_object:laptop".to_string()));
    assert!(facets.contains(&"quantity_object:tank".to_string()));
    assert!(facets.contains(&"quantity_count:object".to_string()));
    assert!(facets.contains(&"inventory_count:contained".to_string()));
    assert!(facets.contains(&"quantity_unit:gallon".to_string()));
    assert!(facets.contains(&"quantity_unit_object:gallon_tank".to_string()));
    assert!(facets.contains(&"inventory_object:monitor".to_string()));
    assert!(facets.contains(&"inventory_object:tank".to_string()));
    assert!(!facets.contains(&"quantity_object:week".to_string()));
}

#[test]
fn extracts_completion_count_facets_from_word_number_syntax() {
    let facets = extract_memory_facets(
        "User: I've completed three courses on Coursera and finished two workshops last month.",
        None,
        &[],
    );

    assert!(facets.contains(&"quantity_count:object".to_string()));
    assert!(facets.contains(&"completion_count:object".to_string()));
    assert!(facets.contains(&"quantity_object:course".to_string()));
    assert!(facets.contains(&"quantity_object:workshop".to_string()));
}

#[test]
fn completion_count_query_weights_completed_quantity_evidence() {
    let available = |cue: &str| {
        matches!(
            cue,
            "completion_count:object" | "quantity_count:object" | "quantity_object:course"
        )
    };
    let intent = compile_query_intent(
        "How many online courses have I completed in total?",
        available,
    );

    assert!(intent.labels.contains(&"completion_count".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "completion_count:object" && *weight > 7.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "quantity_object:course" && *weight > 4.0));
}

#[test]
fn count_query_with_possessive_collection_is_inventory_intent() {
    let available = |cue: &str| {
        matches!(
            cue,
            "quantity_count:object"
                | "inventory_object:fish"
                | "quantity_object:fish"
                | "inventory_count:contained"
                | "type:ownership"
                | "source_role:user"
        )
    };
    let intent = compile_query_intent(
        "How many fish are there in total in both of my aquariums?",
        available,
    );

    assert!(intent.labels.contains(&"count".to_string()));
    assert!(intent.labels.contains(&"inventory".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "quantity_count:object" && *weight > 3.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "inventory_count:contained" && *weight > 3.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "inventory_object:fish" && *weight > 3.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "source_role:user" && *weight >= 3.0));
}

#[test]
fn contained_singular_inventory_item_counts_as_quantity_evidence() {
    let facets = extract_memory_facets(
        "I upgraded my old 10-gallon tank, which has my betta fish, Bubbles.",
        None,
        &[],
    );

    assert!(facets.contains(&"quantity_count:object".to_string()));
    assert!(facets.contains(&"inventory_count:contained".to_string()));
    assert!(facets.contains(&"quantity_object:betta".to_string()));
    assert!(facets.contains(&"quantity_object:fish".to_string()));

    let generic = extract_memory_facets(
        "I added decorations to create more hiding places for my fish.",
        None,
        &[],
    );
    assert!(!generic.contains(&"quantity_count:object".to_string()));
}

#[test]
fn extracts_project_work_from_first_person_work_patterns() {
    let active_project = extract_memory_facets(
        "User: I've been working on a solo project for my Data Mining class.",
        None,
        &[],
    );
    assert!(active_project.contains(&"type:project_work".to_string()));
    assert!(active_project.contains(&"type:activity".to_string()));

    let research = extract_memory_facets(
        "User: I recently presented a poster on my research at an academic conference.",
        None,
        &[],
    );
    assert!(research.contains(&"type:project_work".to_string()));
    assert!(research.contains(&"type:activity".to_string()));

    let generic_advice = extract_memory_facets(
        "Assistant: A project timeline should include milestones and dependencies.",
        None,
        &[],
    );
    assert!(!generic_advice.contains(&"type:project_work".to_string()));
}

#[test]
fn inventory_count_query_prefers_quantity_object_over_temporal_current() {
    let available = |cue: &str| {
        matches!(
            cue,
            "quantity_object:tank"
                | "inventory_object:tank"
                | "type:ownership"
                | "source_role:user"
                | "type:update"
                | "has:number"
                | "temporal:current"
                | "temporal:recent"
        )
    };
    let intent = compile_query_intent(
        "How many tanks do I currently have, including the one I set up for my friend's kid?",
        available,
    );

    assert!(intent.labels.contains(&"count".to_string()));
    assert!(intent.labels.contains(&"inventory".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "inventory_object:tank" && *weight > 3.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "quantity_object:tank" && *weight > 3.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "type:ownership" && *weight > 2.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "source_role:user" && *weight >= 3.0));
    assert!(!intent.labels.contains(&"state_update".to_string()));
    assert!(!intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "type:update"));
    assert!(!intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "temporal:current" || cue == "temporal:recent"));
    assert!(intent
        .cue_weight_adjustments
        .iter()
        .any(|(cue, weight)| cue == "currently" && *weight < 1.0));
    assert!(intent
        .cue_weight_adjustments
        .iter()
        .any(|(cue, weight)| cue == "one" && *weight < 1.0));
}

#[test]
fn project_count_query_prefers_project_work_facets() {
    let available = |cue: &str| {
        matches!(
            cue,
            "project"
                | "type:project_work"
                | "type:activity"
                | "source_role:user"
                | "type:update"
                | "temporal:recent"
        )
    };
    let intent = compile_query_intent(
        "How many projects have I led or am currently leading?",
        available,
    );

    assert!(intent.labels.contains(&"count".to_string()));
    assert!(intent
        .labels
        .contains(&"project_work_count".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "type:project_work" && *weight >= 4.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "source_role:user" && *weight >= 3.0));
}

#[test]
fn event_count_query_prefers_counted_object_without_inventory_or_state_update() {
    let available = |cue: &str| {
        matches!(
            cue,
            "wedding"
                | "type:ownership"
                | "type:update"
                | "type:activity"
                | "type:event"
                | "temporal:relative"
                | "temporal:last_week"
                | "temporal:recent"
                | "source_year:2023"
        )
    };
    let intent = compile_query_intent_with_reference_time(
        "How many weddings have I attended in this year?",
        Some("2023/10/15 (Sun) 23:47"),
        available,
    );

    assert!(intent.labels.contains(&"count".to_string()));
    assert!(intent.labels.contains(&"activity_event".to_string()));
    assert!(intent.labels.contains(&"temporal_window".to_string()));
    assert!(intent
        .labels
        .contains(&"temporal_resolved_year".to_string()));
    assert!(!intent.labels.contains(&"inventory".to_string()));
    assert!(!intent.labels.contains(&"state_update".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "wedding" && *weight >= 8.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "source_year:2023" && *weight >= 1.0));
    for rejected in [
        "type:ownership",
        "type:update",
        "temporal:relative",
        "temporal:last_week",
        "temporal:recent",
    ] {
        assert!(
            !intent.weighted_cues.iter().any(|(cue, _)| cue == rejected),
            "event count query should not inject {rejected}"
        );
    }
}

#[test]
fn age_difference_query_prefers_age_and_education_over_duration_temporal_state() {
    let available = |cue: &str| {
        matches!(
            cue,
            "has:number"
                | "has:duration"
                | "has:date"
                | "has:age"
                | "age:current"
                | "age:event"
                | "education:graduation"
                | "education:degree"
                | "education:college"
                | "type:update"
                | "temporal:recent"
                | "temporal:last_week"
        )
    };
    let intent = compile_query_intent(
        "How many years older am I than when I graduated from college?",
        available,
    );

    assert!(intent.labels.contains(&"count".to_string()));
    assert!(intent.labels.contains(&"age_query".to_string()));
    assert!(intent.labels.contains(&"age_difference".to_string()));
    assert!(!intent.labels.contains(&"duration".to_string()));
    assert!(!intent.labels.contains(&"temporal_window".to_string()));
    assert!(!intent.labels.contains(&"state_update".to_string()));

    for expected in [
        "has:age",
        "age:current",
        "age:event",
        "education:graduation",
        "education:degree",
    ] {
        assert!(
            intent.weighted_cues.iter().any(|(cue, _)| cue == expected),
            "missing age-difference cue {expected}"
        );
    }
    for rejected in ["has:duration", "temporal:recent", "temporal:last_week", "type:update"] {
        assert!(
            !intent.weighted_cues.iter().any(|(cue, _)| cue == rejected),
            "age-difference query should not inject {rejected}"
        );
    }
    assert!(intent
        .cue_weight_adjustments
        .iter()
        .any(|(cue, multiplier)| cue == "year" && *multiplier < 1.0));
    assert!(intent
        .cue_weight_adjustments
        .iter()
        .any(|(cue, multiplier)| cue == "old" && *multiplier < 1.0));
}

#[test]
fn sibling_count_query_prefers_family_facets_over_inventory_ownership() {
    let available = |cue: &str| {
        matches!(
            cue,
            "has:number"
                | "type:ownership"
                | "family_count:sibling"
                | "family_scope:self"
                | "family_relation:sibling"
                | "sibling_kind:brother"
                | "sibling_kind:sister"
        )
    };
    let intent = compile_query_intent("What is the total number of siblings I have?", available);

    assert!(intent.labels.contains(&"count".to_string()));
    assert!(intent
        .labels
        .contains(&"family_relation_count".to_string()));
    assert!(!intent.labels.contains(&"inventory".to_string()));
    assert!(!intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "type:ownership"));
    for expected in [
        "family_count:sibling",
        "family_scope:self",
        "family_relation:sibling",
        "sibling_kind:brother",
        "sibling_kind:sister",
    ] {
        assert!(
            intent.weighted_cues.iter().any(|(cue, _)| cue == expected),
            "missing family cue {expected}"
        );
    }
}

#[test]
fn weekly_routine_count_queries_use_schedule_facets_not_last_week_temporal_window() {
    let facets = extract_memory_facets(
        "User: I usually take Zumba classes on Tuesdays and Thursdays at 7:00 PM.",
        None,
        &[],
    );
    assert!(facets.contains(&"has:weekday".to_string()));
    assert!(facets.contains(&"has:time".to_string()));
    assert!(facets.contains(&"schedule:weekly".to_string()));
    assert!(facets.contains(&"type:routine".to_string()));

    let available = |cue: &str| {
        matches!(
            cue,
            "fitness"
                | "class"
                | "has:frequency"
                | "schedule:frequency"
                | "frequency_unit:week"
                | "schedule:weekly"
                | "has:weekday"
                | "has:time"
                | "type:routine"
                | "type:activity"
                | "source_role:user"
                | "temporal:last_week"
                | "type:update"
        )
    };
    let intent = compile_query_intent(
        "How many fitness classes do I attend in a typical week?",
        available,
    );

    assert!(intent.labels.contains(&"count".to_string()));
    assert!(intent.labels.contains(&"weekly_routine".to_string()));
    assert!(!intent.labels.contains(&"temporal_window".to_string()));
    assert!(!intent.labels.contains(&"state_update".to_string()));
    assert!(!intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "temporal:last_week" || cue == "type:update"));
    for expected in [
        "fitness",
        "class",
        "has:frequency",
        "schedule:frequency",
        "frequency_unit:week",
        "schedule:weekly",
        "has:weekday",
        "type:routine",
    ] {
        assert!(
            intent.weighted_cues.iter().any(|(cue, _)| cue == expected),
            "missing weekly routine cue {expected}"
        );
    }
}

#[test]
fn family_duration_queries_weight_relation_and_co_residence_evidence() {
    let available = |cue: &str| {
        matches!(
            cue,
            "has:duration"
                | "family_relation:parent"
                | "co_residence:with_self"
                | "source_role:user"
        )
    };
    let intent = compile_query_intent(
        "How long have my parents been staying with me?",
        available,
    );

    assert!(intent.labels.contains(&"duration".to_string()));
    assert!(intent.labels.contains(&"family_relation".to_string()));
    assert!(intent.labels.contains(&"co_residence".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "family_relation:parent" && *weight >= 3.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "co_residence:with_self" && *weight >= 4.0));
}

#[test]
fn family_duration_query_prefers_family_stay_duration_over_unrelated_duration() {
    let engine = CueMapEngine::new();

    let add = |content: &str| {
        let mut metadata = HashMap::new();
        metadata.insert("source_role".to_string(), json!("user"));
        let cues = [
            tokenize_to_cues(content),
            extract_memory_facets(content, Some(&metadata), &[]),
        ]
        .concat();
        engine.add_memory(
            content.to_string(),
            cues,
            Some(metadata),
            MainStats::default(),
            false,
        );
    };

    let unrelated = "User: I'm a marketing specialist and have been doing it for about nine months.";
    let legal = "User: I'll ask the attorney how they can help with my parents overstaying their visa.";
    let target = "User: My parents have been a big help while living with me in the US for nine months.";

    add(unrelated);
    add(legal);
    add(target);

    let results = engine.recall_weighted(
        compile_weighted_query(
            &engine,
            "How long have my parents been staying with me in the US?",
        ),
        3,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|result| result.content.as_str()), Some(target));
}

#[test]
fn weekday_schedule_queries_use_schedule_facets_not_relative_temporal_window() {
    let available = |cue: &str| {
        matches!(
            cue,
            "has:weekday"
                | "schedule:weekly"
                | "type:routine"
                | "source_role:user"
                | "temporal:last_week"
                | "temporal:recent"
                | "type:update"
        )
    };
    let intent = compile_query_intent(
        "What day of the week do I take a cocktail-making class?",
        available,
    );

    assert!(intent.labels.contains(&"weekday_schedule".to_string()));
    assert!(!intent.labels.contains(&"temporal_window".to_string()));
    assert!(!intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "temporal:last_week" || cue == "temporal:recent"));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "has:weekday" && *weight >= 4.0));
    assert!(intent
        .cue_weight_adjustments
        .iter()
        .any(|(cue, weight)| cue == "week" && *weight < 1.0));
}

#[test]
fn weekday_schedule_query_prefers_recurring_weekday_class_over_last_week_class() {
    let engine = CueMapEngine::new();

    let add = |content: &str| {
        let mut metadata = HashMap::new();
        metadata.insert("source_role".to_string(), json!("user"));
        let cues = [
            tokenize_to_cues(content),
            extract_memory_facets(content, Some(&metadata), &[]),
        ]
        .concat();
        engine.add_memory(
            content.to_string(),
            cues,
            Some(metadata),
            MainStats::default(),
            false,
        );
    };

    let last_week = "User: We made an Indian-inspired feast in my cooking class last week.";
    let target = "User: I have a cocktail-making class on Fridays, so maybe I can experiment with tequila recipes there.";
    let recipe = "User: I love tequila cocktails and want refreshing summer drink ideas.";

    add(last_week);
    add(recipe);
    add(target);

    let results = engine.recall_weighted(
        compile_weighted_query(
            &engine,
            "What day of the week do I take a cocktail-making class?",
        ),
        3,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|result| result.content.as_str()), Some(target));
}

#[test]
fn temporal_distance_questions_do_not_default_to_recent_or_last_week() {
    let intent = compile_query_intent(
        "How many months ago did I attend the photography workshop?",
        |cue| {
            matches!(
                cue,
                "has:date"
                    | "temporal:relative"
                    | "temporal:last_week"
                    | "temporal:recent"
                    | "source_role:user"
            )
        },
    );

    assert!(intent.labels.contains(&"temporal_distance".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "has:date"));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "temporal:relative"));
    assert!(!intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "temporal:last_week"));
    assert!(!intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "temporal:recent"));
    for cue in ["month", "months", "ago"] {
        assert!(intent
            .cue_weight_adjustments
            .iter()
            .any(|(adjusted, multiplier)| adjusted == cue && *multiplier < 1.0));
    }
}

#[test]
fn professional_role_query_adds_weighted_role_without_dropping_visit() {
    let available = |cue: &str| matches!(cue, "dr" | "person_title:dr" | "person_ref:named");
    let intent = compile_query_intent("How many different doctors did I visit?", available);

    assert!(intent.labels.contains(&"count".to_string()));
    assert!(intent.weighted_cues.iter().any(|(cue, _)| cue == "dr"));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "person_title:dr"));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "person_ref:named"));
    assert!(intent
        .cue_weight_adjustments
        .iter()
        .any(|(cue, weight)| cue == "many" && *weight < 1.0));
    assert!(intent
        .cue_weight_adjustments
        .iter()
        .any(|(cue, weight)| cue == "different" && *weight < 1.0));
    assert!(!intent
        .cue_weight_adjustments
        .iter()
        .any(|(cue, _)| cue == "visit"));
}

#[test]
fn where_visit_query_keeps_visit_as_primary_cue() {
    let intent = compile_query_intent("Where did I visit?", |_| false);

    assert!(!intent.labels.contains(&"count".to_string()));
    assert!(!intent
        .cue_weight_adjustments
        .iter()
        .any(|(cue, _)| cue == "visit"));
}

#[test]
fn doctor_appointment_event_query_does_not_expand_to_doctor_title() {
    let intent = compile_query_intent(
        "What time did I go to bed on the day before I had a doctor's appointment?",
        |cue| {
            matches!(
                cue,
                "dr"
                    | "person_title:dr"
                    | "routine:bed_time"
                    | "has:time"
                    | "time_of_day:night"
                    | "source_role:user"
            )
        },
    );

    assert!(intent.labels.contains(&"bed_time".to_string()));
    assert!(!intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "dr" || cue == "person_title:dr"));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "routine:bed_time" && *weight >= 4.0));
}

#[test]
fn title_abbreviation_query_expansion_is_not_doctor_specific() {
    let available = |cue: &str| matches!(cue, "prof" | "person_title:prof" | "person_ref:named");
    let intent = compile_query_intent("How many professors did I meet?", available);

    assert!(intent.labels.contains(&"person_role".to_string()));
    assert!(intent.weighted_cues.iter().any(|(cue, _)| cue == "prof"));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "person_title:prof"));
}

#[test]
fn present_state_queries_compile_update_intent_when_available() {
    let available = |cue: &str| matches!(cue, "type:update" | "temporal:recent");
    let travel_intent = compile_query_intent(
        "Where am I planning to stay for my birthday trip to Hawaii?",
        available,
    );
    let model_intent = compile_query_intent(
        "What type of vehicle model am I currently working on?",
        available,
    );

    assert!(travel_intent.labels.contains(&"state_update".to_string()));
    assert!(travel_intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "type:update" && *weight > 2.0));
    assert!(model_intent.labels.contains(&"state_update".to_string()));
    assert!(model_intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "type:update" && *weight > 2.0));
    assert!(model_intent
        .cue_weight_adjustments
        .iter()
        .any(|(cue, multiplier)| cue == "type" && *multiplier < 1.0));
}

#[test]
fn future_scheduled_advice_queries_do_not_compile_temporal_window() {
    let available = |cue: &str| {
        matches!(
            cue,
            "has:date"
                | "temporal:relative"
                | "temporal:last_week"
                | "temporal:recent"
                | "type:homegrown"
                | "type:ingredient"
        )
    };

    for query in [
        "What should I serve for dinner this weekend with my homegrown ingredients?",
        "I was thinking about rearranging the furniture in my bedroom this weekend. Any tips?",
        "I'm getting excited about my visit to the music store this weekend. Any tips on what to look for in a new guitar?",
    ] {
        let intent = compile_query_intent(query, available);
        assert!(
            !intent.labels.contains(&"temporal_window".to_string()),
            "future advice query was treated as temporal recall: {query}"
        );
        assert!(
            !intent.weighted_cues.iter().any(|(cue, _)| matches!(
                cue.as_str(),
                "has:date" | "temporal:relative" | "temporal:last_week" | "temporal:recent"
            )),
            "future advice query injected temporal facets: {query}"
        );
    }

    let homegrown_intent = compile_query_intent(
        "What should I serve for dinner this weekend with my homegrown ingredients?",
        available,
    );
    assert!(homegrown_intent.labels.contains(&"homegrown".to_string()));
    assert!(homegrown_intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "type:homegrown" && *weight > 3.0));
    assert!(homegrown_intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "type:ingredient" && *weight > 2.0));
    assert!(homegrown_intent
        .cue_weight_adjustments
        .iter()
        .any(|(cue, multiplier)| cue == "weekend" && *multiplier < 1.0));
}

#[test]
fn past_time_window_queries_still_compile_temporal_window() {
    let available = |cue: &str| {
        matches!(
            cue,
            "has:date" | "temporal:relative" | "temporal:last_week" | "temporal:recent"
        )
    };

    for query in [
        "What gardening-related activity did I do two weeks ago?",
        "What did I buy last weekend?",
    ] {
        let intent = compile_query_intent(query, available);
        assert!(
            intent.labels.contains(&"temporal_window".to_string()),
            "past recall query lost temporal intent: {query}"
        );
        assert!(
            intent
                .weighted_cues
                .iter()
                .any(|(cue, _)| cue == "temporal:relative" || cue == "temporal:last_week"),
            "past recall query did not inject temporal facets: {query}"
        );
    }
}

#[test]
fn temporal_order_queries_do_not_become_latest_state_updates() {
    let available = |cue: &str| {
        matches!(
            cue,
            "source_time:dated" | "source_role:user" | "type:activity" | "type:event"
        )
    };
    let intent = compile_query_intent(
        "What is the order of the six museums I visited from earliest to latest?",
        available,
    );

    assert!(intent.labels.contains(&"temporal_order".to_string()));
    assert!(intent.labels.contains(&"activity_event".to_string()));
    assert!(!intent.labels.contains(&"latest_current".to_string()));
    assert!(!intent.labels.contains(&"state_update".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "source_time:dated" && *weight >= 2.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "type:activity" && *weight >= 3.0));
    assert!(intent
        .cue_weight_adjustments
        .iter()
        .any(|(cue, multiplier)| cue == "latest" && *multiplier < 1.0));
}

#[test]
fn transport_mode_comparison_queries_target_concrete_transport_events() {
    let available = |cue: &str| {
        matches!(
            cue,
            "source_time:dated"
                | "source_role:user"
                | "type:activity"
                | "type:event"
                | "transport_event:bus"
                | "transport_event:train"
                | "transport_mode:bus"
                | "transport_mode:train"
                | "bus"
                | "train"
                | "temporal:recent"
                | "type:update"
        )
    };
    let intent = compile_query_intent(
        "Which mode of transport did I use most recently, a bus or a train?",
        available,
    );

    assert!(intent
        .labels
        .contains(&"transport_mode_comparison".to_string()));
    assert!(!intent.labels.contains(&"latest_current".to_string()));
    assert!(!intent.labels.contains(&"state_update".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "transport_event:bus" && *weight >= 4.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "transport_event:train" && *weight >= 4.0));
    assert!(!intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "type:update"));
    assert!(intent
        .cue_weight_adjustments
        .iter()
        .any(|(cue, multiplier)| cue == "recently" && *multiplier < 1.0));
}

#[test]
fn relative_temporal_query_resolves_against_reference_time() {
    let available = |cue: &str| {
        matches!(
            cue,
            "has:date"
                | "has:duration"
                | "temporal:relative"
                | "temporal:last_week"
                | "temporal:recent"
                | "temporal:today"
                | "source_date:2023_04_21"
                | "source_week:2023_w16"
                | "source_month:2023_04"
        )
    };
    let intent = compile_query_intent_with_reference_time(
        "What gardening-related activity did I do two weeks ago?",
        Some("2023/05/05 (Fri) 16:42"),
        available,
    );

    assert!(intent.labels.contains(&"temporal_window".to_string()));
    assert!(intent
        .labels
        .contains(&"temporal_resolved_date".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "source_date:2023_04_21" && *weight >= 1.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "source_week:2023_w16"));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "temporal:today"));
    assert!(intent
        .cue_weight_adjustments
        .iter()
        .any(|(cue, multiplier)| cue == "ago" && *multiplier < 1.0));
    assert!(intent
        .cue_weight_adjustments
        .iter()
        .any(|(cue, multiplier)| cue == "two" && *multiplier < 1.0));
}

#[test]
fn charity_event_temporal_query_targets_exact_date_and_domain() {
    let available = |cue: &str| {
        matches!(
            cue,
            "source_date:2023_03_19"
                | "source_week:2023_w11"
                | "source_month:2023_03"
                | "temporal:today"
                | "type:activity"
                | "type:event"
                | "source_role:user"
                | "event_domain:charity"
        )
    };
    let intent = compile_query_intent_with_reference_time(
        "What charity event did I participate in a month ago?",
        Some("2023/04/18 (Tue) 18:34"),
        available,
    );

    assert!(intent.labels.contains(&"temporal_window".to_string()));
    assert!(intent
        .labels
        .contains(&"temporal_resolved_date".to_string()));
    assert!(intent.labels.contains(&"charity_event".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "source_date:2023_03_19" && *weight >= 1.8));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "event_domain:charity" && *weight >= 3.0));
}

#[test]
fn last_week_query_targets_source_week_not_single_monday() {
    let available = |cue: &str| {
        matches!(
            cue,
            "source_date:2023_04_03"
                | "source_week:2023_w14"
                | "source_month:2023_04"
                | "type:activity"
                | "type:event"
                | "source_role:user"
                | "activity_domain:religion"
                | "topic:religion"
        )
    };
    let intent = compile_query_intent_with_reference_time(
        "Where did I attend the religious activity last week?",
        Some("2023/04/10 (Mon) 12:00"),
        available,
    );

    assert!(intent.labels.contains(&"temporal_resolved_week".to_string()));
    assert!(intent.labels.contains(&"activity_event".to_string()));
    assert!(intent.labels.contains(&"religious_activity".to_string()));
    assert!(!intent
        .labels
        .contains(&"temporal_resolved_date".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "source_week:2023_w14" && *weight >= 1.8));
    assert!(!intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "source_date:2023_04_03"));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "type:activity" && *weight >= 3.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "activity_domain:religion" && *weight >= 4.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "source_role:user" && *weight >= 1.0));
}

#[test]
fn past_weekend_query_targets_weekend_source_dates() {
    let available = |cue: &str| {
        matches!(
            cue,
            "source_date:2023_03_18"
                | "source_date:2023_03_19"
                | "source_week:2023_w11"
                | "temporal:relative"
                | "temporal:last_week"
                | "temporal:recent"
        )
    };
    let intent = compile_query_intent_with_reference_time(
        "Which bike did I fixed or serviced the past weekend?",
        Some("2023/03/21 (Tue) 21:43"),
        available,
    );

    assert!(intent.labels.contains(&"temporal_window".to_string()));
    assert!(intent
        .labels
        .contains(&"temporal_resolved_weekend".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "source_date:2023_03_18" && *weight >= 1.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "source_date:2023_03_19" && *weight >= 1.0));
    assert!(!intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "temporal:last_week" || cue == "temporal:recent"));
    assert!(intent
        .cue_weight_adjustments
        .iter()
        .any(|(cue, multiplier)| cue == "past" && *multiplier < 1.0));
    assert!(intent
        .cue_weight_adjustments
        .iter()
        .any(|(cue, multiplier)| cue == "weekend" && *multiplier < 1.0));
}

#[test]
fn past_month_query_targets_previous_source_month() {
    let available = |cue: &str| {
        matches!(
            cue,
            "source_month:2023_06"
                | "source_month:2023_07"
                | "source_year:2023"
                | "type:competition_event"
                | "activity_domain:sport"
                | "type:activity"
                | "type:event"
                | "source_role:user"
        )
    };
    let intent = compile_query_intent_with_reference_time(
        "What is the order of the three sports events I participated in during the past month, from earliest to latest?",
        Some("2023/07/01 (Sat) 20:43"),
        available,
    );

    assert!(intent.labels.contains(&"temporal_order".to_string()));
    assert!(intent
        .labels
        .contains(&"temporal_resolved_month".to_string()));
    assert!(intent.labels.contains(&"competition_event".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "source_month:2023_06" && *weight >= 1.5));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "type:competition_event"));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "activity_domain:sport"));
    assert!(intent
        .cue_weight_adjustments
        .iter()
        .any(|(cue, multiplier)| cue == "participated" && *multiplier > 1.0));
}

#[test]
fn streaming_service_query_targets_usage_not_global_recency() {
    let available = |cue: &str| {
        matches!(
            cue,
            "media:streaming"
                | "media:music_streaming"
                | "media:music"
                | "media:watching"
                | "type:usage"
                | "source_role:user"
                | "type:update"
                | "temporal:recent"
        )
    };
    let intent = compile_query_intent(
        "Which streaming service did I start using most recently?",
        available,
    );

    assert!(intent
        .labels
        .contains(&"streaming_service_usage".to_string()));
    assert!(!intent.labels.contains(&"latest_current".to_string()));
    assert!(!intent.labels.contains(&"state_update".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "media:streaming" && *weight >= 4.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "type:usage" && *weight >= 3.0));
    assert!(!intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "type:update"));

    let music_intent = compile_query_intent(
        "What is the name of the music streaming service have I been using lately?",
        available,
    );
    assert!(music_intent
        .labels
        .contains(&"music_streaming_service_usage".to_string()));
    assert!(music_intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "media:music_streaming" && *weight >= 4.0));
    assert!(music_intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "media:music"));
}

#[test]
fn wake_time_routine_query_targets_time_weekday_and_user_routine() {
    let available = |cue: &str| {
        matches!(
            cue,
            "routine:wake_time"
                | "has:time"
                | "has:weekday"
                | "type:routine"
                | "source_role:user"
                | "type:update"
        )
    };
    let intent = compile_query_intent("What time do I wake up on Saturday mornings?", available);

    assert!(intent.labels.contains(&"wake_time_routine".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "routine:wake_time" && *weight >= 4.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "has:time" && *weight >= 3.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "has:weekday" && *weight >= 2.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "type:routine" && *weight >= 2.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "source_role:user" && *weight >= 2.0));
}

#[test]
fn current_reading_query_targets_book_reading_state() {
    let available = |cue: &str| {
        matches!(
            cue,
            "reading:current"
                | "media:book_reading"
                | "media:book"
                | "source_role:user"
                | "temporal:current"
                | "type:update"
        )
    };
    let intent = compile_query_intent("What book am I currently reading?", available);

    assert!(intent.labels.contains(&"current_reading".to_string()));
    assert!(!intent.labels.contains(&"state_update".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "reading:current" && *weight >= 4.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "media:book_reading" && *weight >= 4.0));
    assert!(!intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "type:update"));
}

#[test]
fn personal_attribute_queries_target_entity_attribute_facets() {
    let available = |cue: &str| {
        matches!(
            cue,
            "type:entity_attribute"
                | "attribute:class_relation"
                | "source_role:user"
                | "type:ownership"
        )
    };
    let intent = compile_query_intent("What breed is my dog?", available);

    assert!(intent.labels.contains(&"entity_attribute".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "type:entity_attribute" && *weight >= 3.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "attribute:class_relation" && *weight >= 2.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "source_role:user" && *weight >= 1.0));

    let preference_intent = compile_query_intent("What brand are my favorite running shoes?", |cue| {
        matches!(
            cue,
            "type:entity_attribute"
                | "attribute:class_relation"
                | "source_role:user"
                | "type:preference"
        )
    });
    assert!(preference_intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "source_role:user" && *weight >= 3.0));
}

#[test]
fn personal_attribute_recall_prefers_class_relation_over_generic_object_mentions() {
    let engine = CueMapEngine::new();
    let dog_walker = "User: I need help finding a good dog walker in my area.";
    let dog_toy = "User: I'm also thinking of getting Max a new toy, something interactive for dogs.";
    let breed_memory = "User: I'm thinking of getting Max a new collar with a nice name tag. Do you have any recommendations for a good collar brand or type that would suit a Golden Retriever like Max?";

    for content in [dog_walker, dog_toy, breed_memory] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let results = engine.recall_weighted(
        compile_weighted_query(&engine, "What breed is my dog?"),
        3,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(breed_memory));
}

#[test]
fn favorite_attribute_recall_prefers_user_fact_over_generic_advice() {
    let engine = CueMapEngine::new();
    let advice =
        "Assistant: Nike is a great choice for running shoes, and there are many brands to compare.";
    let generic =
        "Assistant: When buying gym shoes, compare brand, fit, cushioning, and durability.";
    let target =
        "User: Nike has been my favourite brand so far for running shoes.";

    for content in [advice, generic, target] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let results = engine.recall_weighted(
        compile_weighted_query(&engine, "What brand are my favorite running shoes?"),
        3,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(target));
}

#[test]
fn purchase_source_query_weights_acquired_object_and_source_facet() {
    let intent = compile_query_intent(
        "Where did I buy my new tennis racket from?",
        |cue| {
            matches!(
                cue,
                "tennis"
                    | "racket"
                    | "tennis_racket"
                    | "purchase:source"
                    | "type:ownership"
                    | "type:activity"
                    | "type:event"
                    | "source_role:user"
            )
        },
    );

    assert!(intent.labels.contains(&"purchase".to_string()));
    assert!(intent.labels.contains(&"purchase_source".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "tennis_racket" && *weight >= 4.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "purchase:source" && *weight >= 3.0));
}

#[test]
fn received_from_whom_query_uses_acquisition_source_intent() {
    let available = |cue: &str| {
        matches!(
            cue,
            "purchase:source" | "type:ownership" | "type:activity" | "type:event" | "source_role:user"
        )
    };
    let intent = compile_query_intent(
        "I received a piece of jewelry last Saturday from whom?",
        available,
    );

    assert!(intent.labels.contains(&"purchase".to_string()));
    assert!(intent.labels.contains(&"purchase_source".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "purchase:source" && *weight > 3.0));
}

#[test]
fn purchase_query_weights_actual_acquisition_over_same_date_activity() {
    let available = |cue: &str| {
        matches!(
            cue,
            "purchase:acquired"
                | "type:ownership"
                | "type:activity"
                | "type:event"
                | "source_role:user"
        )
    };
    let intent = compile_query_intent(
        "I mentioned an investment for a competition four weeks ago. What did I buy?",
        available,
    );

    assert!(intent.labels.contains(&"purchase".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "purchase:acquired" && *weight > 4.0));
}

#[test]
fn purchase_query_ranks_acquired_item_over_temporal_activity() {
    let engine = CueMapEngine::new();
    let distractor = "User: We've been cooking a lot of Indian dishes, and last Sunday, we made chicken biryani.";
    let target = "User: I actually got my own set of sculpting tools, including a modeling tool set, a wire cutter, and a sculpting mat today.";

    for content in [distractor, target] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let query = "I mentioned an investment for a competition four weeks ago. What did I buy?";
    let mut weighted_cues: Vec<(String, f64)> =
        tokenize_to_cues(query).into_iter().map(|cue| (cue, 1.0)).collect();
    let intent = compile_query_intent(query, |cue| engine.get_cue_frequency(cue) > 0);
    for (cue, weight) in intent.weighted_cues {
        if let Some((_, existing)) = weighted_cues.iter_mut().find(|(existing, _)| existing == &cue)
        {
            if *existing < weight {
                *existing = weight;
            }
        } else {
            weighted_cues.push((cue, weight));
        }
    }

    let results = engine.recall_weighted(
        weighted_cues,
        2,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(target));
}

#[test]
fn acquisition_source_query_ranks_source_memory_over_temporal_distractor() {
    let engine = CueMapEngine::new();
    let distractor = "User: Last Saturday I hit a new personal best of 12,345 steps while running errands.";
    let target = "User: I also got a stunning crystal chandelier from my aunt today, which used to belong to my great-grandmother.";

    for content in [distractor, target] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let query = "I received a piece last Saturday from whom?";
    let mut weighted_cues: Vec<(String, f64)> =
        tokenize_to_cues(query).into_iter().map(|cue| (cue, 1.0)).collect();
    let intent = compile_query_intent(query, |cue| engine.get_cue_frequency(cue) > 0);
    for (cue, weight) in intent.weighted_cues {
        if let Some((_, existing)) = weighted_cues.iter_mut().find(|(existing, _)| existing == &cue)
        {
            if *existing < weight {
                *existing = weight;
            }
        } else {
            weighted_cues.push((cue, weight));
        }
    }

    let results = engine.recall_weighted(
        weighted_cues,
        2,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(target));
}

#[test]
fn purchase_source_recall_prefers_matching_item_over_other_purchases() {
    let engine = CueMapEngine::new();
    let other_purchase =
        "User: I recently bought an eyeshadow palette at Sephora and earned loyalty points.";
    let tennis_followup =
        "User: I'll check the weather forecast online. Do you know warmups before playing tennis?";
    let target =
        "User: I'm really happy with my new tennis racket, which I got from a sports store downtown.";

    for content in [other_purchase, tennis_followup, target] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let results = engine.recall_weighted(
        compile_weighted_query(&engine, "Where did I buy my new tennis racket from?"),
        3,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(target));
}

#[test]
fn purchase_source_recall_handles_item_is_from_source_statement() {
    let engine = CueMapEngine::new();
    let other_purchase =
        "User: I bought old light bulbs from Home Depot and checked for spares.";
    let generic_bookshelf =
        "User: I got a new bookshelf, which helped me declutter the living room.";
    let target =
        "User: The new bookshelf is from IKEA, and I'm really happy with it.";

    for content in [other_purchase, generic_bookshelf, target] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let results = engine.recall_weighted(
        compile_weighted_query(&engine, "Where did I buy my new bookshelf from?"),
        3,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(target));
}

#[test]
fn wake_time_routine_recall_prefers_timed_user_routine_over_weather_mentions() {
    let engine = CueMapEngine::new();
    let weather = "User: I'm planning to go for a jog on Saturday morning, what's the weather forecast like?";
    let assistant_schedule = "Assistant: Your desired wake-up times include a consistent bedtime and Saturday morning routine.";
    let target = "User: I've been waking up around 8:30 am on Saturdays, which gives me enough time to fit in a 30-minute jog.";

    for content in [weather, assistant_schedule, target] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let results = engine.recall_weighted(
        compile_weighted_query(&engine, "What time do I wake up on Saturday mornings?"),
        3,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(target));
}

#[test]
fn bed_time_query_prefers_timed_bed_memory_over_doctor_title_mentions() {
    let engine = CueMapEngine::new();
    let doctor_lecture =
        "User: I attended a lecture series downtown where the speaker, Dr. Khan, discussed stress.";
    let appointment =
        "User: I had a doctor's appointment at 10 AM last Thursday, and that's when I got my blood test results.";
    let target =
        "User: I'm feeling sluggish because I didn't get to bed until 2 AM last Wednesday.";

    for content in [doctor_lecture, appointment, target] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let results = engine.recall_weighted(
        compile_weighted_query(
            &engine,
            "What time did I go to bed on the day before I had a doctor's appointment?",
        ),
        3,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(target));
}

#[test]
fn media_watch_recommendations_target_media_memories() {
    let available = |cue: &str| matches!(cue, "media:watching" | "source_role:user");
    let intent = compile_query_intent(
        "Can you recommend a show or movie for me to watch tonight?",
        available,
    );

    assert!(intent.labels.contains(&"recommendation".to_string()));
    assert!(intent
        .labels
        .contains(&"media_watch_recommendation".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "media:watching" && *weight >= 3.0));
    assert!(!intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| matches!(cue.as_str(), "show" | "movie" | "watch")));
}

#[test]
fn inspiration_recommendation_prefers_source_memory_over_same_topic_advice() {
    let engine = CueMapEngine::new();
    let pricing = "User: I'm having trouble pricing my paintings online and want advice on setting fair prices.";
    let flowers = "User: I saw flower paintings on Instagram and asked for tips to paint realistic flowers.";
    let texture = "User: I've been trying to add texture into my paintings with palette knives.";
    let target = "User: I have been getting inspiration from social media and recently started a 30-day painting challenge.";

    for content in [pricing, flowers, texture, target] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let results = engine.recall_weighted(
        compile_weighted_query(
            &engine,
            "I've been feeling a bit stuck with my paintings lately. Do you have any ideas on how I can find new inspiration?",
        ),
        4,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(target));
}

#[test]
fn compiles_query_intent_only_for_available_facets() {
    let available = |cue: &str| matches!(cue, "has:number" | "has:money" | "type:preference");
    let intent = compile_query_intent(
        "How much money did I spend on my favorite camera?",
        available,
    );

    assert!(intent.labels.contains(&"money".to_string()));
    assert!(intent.labels.contains(&"preference".to_string()));
    assert!(intent.suppress_generic);
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "has:money" && *weight > 3.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "type:preference"));
    assert!(!intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "has:duration"));
}

#[test]
fn how_much_time_is_duration_not_money_intent() {
    let available = |cue: &str| matches!(cue, "has:number" | "has:money" | "has:duration");
    let intent = compile_query_intent(
        "How much time do I dedicate to practicing guitar every day?",
        available,
    );

    assert!(intent.labels.contains(&"duration".to_string()));
    assert!(!intent.labels.contains(&"money".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "has:duration"));
    assert!(!intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "has:money"));
}

#[test]
fn activity_duration_queries_keep_duration_intent_with_temporal_windows() {
    let available = |cue: &str| {
        matches!(
            cue,
            "has:number" | "has:duration" | "jog" | "yoga" | "source_week:2023_w21"
        )
    };
    let intent = compile_query_intent_with_reference_time(
        "How many hours of jogging and yoga did I do last week?",
        Some("2023/05/30 (Tue) 21:24"),
        available,
    );

    assert!(intent.labels.contains(&"duration".to_string()));
    assert!(intent.labels.contains(&"temporal_window".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "has:duration"));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "jog" && *weight >= 6.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "yoga" && *weight >= 6.0));
}

#[test]
fn question_words_are_not_query_entity_intents() {
    let available = |cue: &str| {
        matches!(
            cue,
            "entity:what" | "entity:how" | "entity:user" | "entity:assistant"
        )
    };

    for query in [
        "What degree did I graduate with?",
        "How much time do I practice guitar?",
        "Assistant: what did you recommend?",
    ] {
        let intent = compile_query_intent(query, available);
        assert!(
            !intent.weighted_cues.iter().any(|(cue, _)| matches!(
                cue.as_str(),
                "entity:what" | "entity:how" | "entity:user" | "entity:assistant"
            )),
            "generic entity leaked for query: {query}"
        );
    }
}

#[test]
fn conversational_source_intent_uses_structured_roles_when_available() {
    let assistant_available =
        |cue: &str| matches!(cue, "source_role:assistant" | "type:recommendation");
    let assistant_intent = compile_query_intent(
        "You mentioned a store and recommended a fabric supplier. What was it?",
        assistant_available,
    );

    assert!(assistant_intent
        .labels
        .contains(&"source_assistant".to_string()));
    assert!(assistant_intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "source_role:assistant"));

    let say_intent = compile_query_intent(
        "How many eggs did you say we need for the recipe?",
        assistant_available,
    );
    assert!(say_intent.labels.contains(&"source_answer".to_string()));
    assert!(say_intent
        .labels
        .contains(&"source_assistant".to_string()));
    assert!(say_intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "source_role:assistant"));

    let remind_intent = compile_query_intent(
        "Could you remind me of the name of that restaurant in Cihampelas Walk?",
        assistant_available,
    );
    assert!(remind_intent.labels.contains(&"source_answer".to_string()));
    assert!(remind_intent
        .labels
        .contains(&"source_assistant".to_string()));
    assert!(remind_intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "source_role:assistant"));

    let created_available = |cue: &str| {
        matches!(
            cue,
            "source_role:assistant" | "type:answer" | "type:iteration"
        )
    };
    let created_intent = compile_query_intent(
        "Can you remind me what was in the second version you created?",
        created_available,
    );
    assert!(created_intent
        .labels
        .contains(&"source_answer".to_string()));
    assert!(created_intent
        .labels
        .contains(&"source_assistant".to_string()));
    assert!(created_intent
        .labels
        .contains(&"iteration_reference".to_string()));
    assert!(created_intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "type:iteration"));

    let request_intent = compile_query_intent(
        "Can you suggest some accessories that would complement my current photography setup?",
        assistant_available,
    );
    assert!(request_intent
        .labels
        .contains(&"recommendation".to_string()));
    assert!(!request_intent.labels.contains(&"source_answer".to_string()));
    assert!(!request_intent
        .labels
        .contains(&"source_assistant".to_string()));
    assert!(!request_intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "source_role:assistant"));

    let profile_available = |cue: &str| {
        matches!(
            cue,
            "source_role:user"
                | "type:recommendation"
                | "type:preference"
                | "type:ownership"
                | "type:usage"
                | "type:update"
                | "photography"
                | "complement"
        )
    };
    let personal_request_intent = compile_query_intent(
        "Can you suggest some accessories that would complement my current photography setup?",
        profile_available,
    );
    assert!(personal_request_intent
        .labels
        .contains(&"personal_recommendation_context".to_string()));
    assert!(!personal_request_intent
        .labels
        .contains(&"state_update".to_string()));
    assert!(personal_request_intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "source_role:user"));
    assert!(personal_request_intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "type:ownership"));

    let purchase_consideration_available = |cue: &str| {
        matches!(
            cue,
            "source_role:user"
                | "type:purchase_consideration"
                | "type:ownership"
                | "type:preference"
                | "guitar"
        )
    };
    let purchase_consideration_intent = compile_query_intent(
        "Any tips on what to look for in a new guitar?",
        purchase_consideration_available,
    );
    assert!(purchase_consideration_intent
        .labels
        .contains(&"purchase_consideration".to_string()));
    assert!(purchase_consideration_intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "type:purchase_consideration"));

    let research_interest_available = |cue: &str| {
        matches!(
            cue,
            "source_role:user"
                | "type:preference"
                | "type:interest"
                | "type:expertise"
                | "publication"
                | "conference"
                | "might"
                | "find"
                | "interest"
        )
    };
    let research_interest_intent = compile_query_intent(
        "Can you recommend some recent publications or conferences that I might find interesting?",
        research_interest_available,
    );
    assert!(research_interest_intent
        .labels
        .contains(&"vague_interest_recommendation".to_string()));
    assert!(research_interest_intent
        .labels
        .contains(&"research_interest_recommendation".to_string()));
    assert!(research_interest_intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "type:expertise"));
    assert!(research_interest_intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "source_role:user"));
    for cue in ["publication", "conference", "might", "find", "interest"] {
        assert!(
            !research_interest_intent
                .weighted_cues
                .iter()
                .any(|(weighted, _)| weighted == cue),
            "vague interest recommendation should not treat {cue} as a profile topic"
        );
    }

    let inspiration_available = |cue: &str| {
        matches!(
            cue,
            "source_role:user"
                | "type:inspiration_source"
                | "type:interest"
                | "type:recommendation"
                | "type:ownership"
                | "painting"
                | "inspiration"
        )
    };
    let inspiration_intent = compile_query_intent(
        "I've been feeling a bit stuck with my paintings lately. Do you have any ideas on how I can find new inspiration?",
        inspiration_available,
    );
    assert!(inspiration_intent
        .labels
        .contains(&"inspiration_recommendation".to_string()));
    assert!(inspiration_intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "type:inspiration_source" && *weight >= 4.0));
    assert!(inspiration_intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "type:interest"));

    let decision_available = |cue: &str| {
        matches!(
            cue,
            "type:decision" | "type:selection" | "type:naming"
        )
    };
    let decision_intent = compile_query_intent(
        "What did we finally decide to name it?",
        decision_available,
    );
    assert!(decision_intent
        .labels
        .contains(&"decision_selection".to_string()));
    assert!(decision_intent
        .labels
        .contains(&"naming_decision".to_string()));
    assert!(decision_intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "type:decision"));
    assert!(decision_intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "type:naming"));

    let user_available = |cue: &str| matches!(cue, "source_role:user" | "has:date");
    let user_intent = compile_query_intent("What did I mention last week?", user_available);

    assert!(user_intent.labels.contains(&"source_user".to_string()));
    assert!(user_intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "source_role:user"));

    let brought_up_intent = compile_query_intent(
        "Can you list the order in which I brought up the deployment issues?",
        user_available,
    );
    assert!(brought_up_intent
        .labels
        .contains(&"source_user".to_string()));
    assert!(brought_up_intent
        .weighted_cues
        .iter()
        .any(|(cue, _)| cue == "source_role:user"));
}

#[test]
fn recommendation_queries_downweight_prompt_scaffolding_not_topic_terms() {
    let intent = compile_query_intent(
        "Can you suggest some useful accessories for my phone?",
        |cue| matches!(cue, "type:recommendation" | "type:preference" | "phone"),
    );

    assert!(intent.labels.contains(&"recommendation".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "phone" && *weight >= 2.4));
    for cue in ["can", "suggest", "useful", "some"] {
        assert!(intent
            .cue_weight_adjustments
            .iter()
            .any(|(adjusted, multiplier)| adjusted == cue && *multiplier < 1.0));
    }
    assert!(!intent
        .cue_weight_adjustments
        .iter()
        .any(|(adjusted, _)| adjusted == "phone"));
    assert!(!intent
        .cue_weight_adjustments
        .iter()
        .any(|(adjusted, _)| adjusted == "accessory"));

    let recipe_intent = compile_query_intent(
        "I was thinking of trying a new coffee creamer recipe. Any recommendations?",
        |cue| matches!(cue, "type:recipe" | "coffee" | "creamer" | "recipe" | "think" | "try"),
    );

    assert!(recipe_intent.labels.contains(&"recommendation".to_string()));
    for cue in ["think", "try", "new", "recipe"] {
        assert!(recipe_intent
            .cue_weight_adjustments
            .iter()
            .any(|(adjusted, multiplier)| adjusted == cue && *multiplier < 1.0));
        assert!(!recipe_intent
            .weighted_cues
            .iter()
            .any(|(weighted, _)| weighted == cue));
    }
    assert!(recipe_intent
        .weighted_cues
        .iter()
        .any(|(weighted, _)| weighted == "type:recipe"));
    assert!(recipe_intent
        .weighted_cues
        .iter()
        .any(|(weighted, weight)| weighted == "creamer" && *weight >= 4.0));
    assert!(recipe_intent
        .weighted_cues
        .iter()
        .any(|(weighted, weight)| weighted == "coffee" && *weight < 2.4));

    let advice_intent = compile_query_intent(
        "I'm a bit anxious about getting around Tokyo. Do you have any helpful tips?",
        |cue| {
            matches!(
                cue,
                "type:recommendation"
                    | "type:preference"
                    | "type:navigation"
                    | "travel:route"
                    | "tokyo"
                    | "helpful"
            )
        },
    );

    assert!(advice_intent.labels.contains(&"recommendation".to_string()));
    assert!(advice_intent
        .labels
        .contains(&"personal_recommendation_context".to_string()));
    assert!(advice_intent
        .labels
        .contains(&"navigation".to_string()));
    assert!(!advice_intent
        .weighted_cues
        .iter()
        .any(|(weighted, _)| weighted == "helpful"));
    for cue in ["bit", "got", "around", "helpful", "tip"] {
        assert!(advice_intent
            .cue_weight_adjustments
            .iter()
            .any(|(adjusted, multiplier)| adjusted == cue && *multiplier < 1.0));
    }

    let hotel_intent = compile_query_intent(
        "Can you suggest a hotel for my upcoming trip to Miami?",
        |cue| {
            matches!(
                cue,
                "type:recommendation"
                    | "type:preference"
                    | "hotel"
                    | "upcoming"
                    | "trip"
                    | "miami"
            )
        },
    );

    assert!(hotel_intent.labels.contains(&"recommendation".to_string()));
    assert!(hotel_intent
        .weighted_cues
        .iter()
        .any(|(weighted, weight)| weighted == "hotel" && *weight >= 2.4));
    assert!(hotel_intent
        .weighted_cues
        .iter()
        .any(|(weighted, weight)| weighted == "miami" && *weight >= 2.4));
    for cue in ["upcoming", "trip"] {
        assert!(!hotel_intent
            .weighted_cues
            .iter()
            .any(|(weighted, _)| weighted == cue));
        assert!(hotel_intent
            .cue_weight_adjustments
            .iter()
            .any(|(adjusted, multiplier)| adjusted == cue && *multiplier < 1.0));
    }

    let cocktail_intent = compile_query_intent(
        "I've been thinking about making a cocktail for an upcoming get-together, but I'm not sure which one to choose. Any suggestions?",
        |cue| {
            matches!(
                cue,
                "type:recommendation"
                    | "type:preference"
                    | "cocktail"
                    | "make"
                    | "choose"
                    | "sure"
                    | "one"
            )
        },
    );

    assert!(cocktail_intent.labels.contains(&"recommendation".to_string()));
    assert!(cocktail_intent
        .weighted_cues
        .iter()
        .any(|(weighted, weight)| weighted == "cocktail" && *weight >= 4.0));
    for cue in ["make", "choose", "sure", "one"] {
        assert!(
            !cocktail_intent
                .weighted_cues
                .iter()
                .any(|(weighted, _)| weighted == cue),
            "recommendation scaffold cue should not be a topic: {cue}"
        );
    }

    let bake_intent = compile_query_intent(
        "I'm thinking of inviting my colleagues over for a small gathering. Any tips on what to bake?",
        |cue| matches!(cue, "gather" | "bake" | "colleague" | "type:recipe"),
    );

    assert!(bake_intent.labels.contains(&"recommendation".to_string()));
    assert!(bake_intent
        .weighted_cues
        .iter()
        .any(|(weighted, weight)| weighted == "bake" && *weight >= 4.0));
    assert!(bake_intent
        .weighted_cues
        .iter()
        .any(|(weighted, weight)| weighted == "gather" && *weight < 2.4));
}

#[test]
fn weighted_facet_query_reranks_structured_evidence() {
    let engine = CueMapEngine::new();
    engine.add_memory(
        "Camera maintenance notes and lens cleaning checklist.".to_string(),
        vec!["camera".to_string()],
        None,
        MainStats::default(),
        false,
    );
    engine.add_memory(
        "I prefer Fuji cameras for street photography.".to_string(),
        vec!["camera".to_string()],
        None,
        MainStats::default(),
        false,
    );

    let results = engine.recall_weighted(
        vec![
            ("camera".to_string(), 1.0),
            ("type:preference".to_string(), 3.0),
        ],
        2,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(
        results.first().map(|r| r.content.as_str()),
        Some("I prefer Fuji cameras for street photography.")
    );
}

#[test]
fn update_facet_reranks_current_state_over_older_topic_match() {
    let engine = CueMapEngine::new();
    let older = "I'm planning a birthday trip to Hawaii and I was wondering if you could recommend some good hiking trails on Kauai?";
    let updated =
        "I'm actually planning to stay on Oahu, so Hanauma Bay and Shark's Cove sound perfect.";
    let generic_stay = "I'm planning a trip to Seoul and looking for the best areas to stay.";

    for content in [older, updated, generic_stay] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let results = engine.recall_weighted(
        compile_weighted_query(
            &engine,
            "Where am I planning to stay for my birthday trip to Hawaii?",
        ),
        3,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(updated));
}

#[test]
fn update_facet_reranks_switched_current_model_over_previous_project() {
    let engine = CueMapEngine::new();
    let noise = "Thanks for the SSD advice. I recently upgraded my PC and will check out those drive models.";
    let previous =
        "I'm looking for tips on weathering effects for my current project, a Ford Mustang Shelby GT350R model.";
    let updated = "I have just wrapped up a model and switched to a Ford F-150 pickup truck.";

    for content in [noise, previous, updated] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let results = engine.recall_weighted(
        compile_weighted_query(
            &engine,
            "What type of vehicle model am I currently working on?",
        ),
        3,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(updated));
}

#[test]
fn person_role_facets_help_count_query_rank_structured_role_evidence() {
    let engine = CueMapEngine::new();
    let travel = "And any scenic drives or lookouts that I should visit?";
    let generic_doctor = "What questions should I ask my doctor before the colonoscopy?";
    let dr_patel = "I had an appointment with Dr. Patel, the ENT specialist, who diagnosed chronic sinusitis and prescribed nasal spray.";
    let dr_smith = "My primary care physician Dr. Smith prescribed antibiotics for a UTI.";
    let dr_lee = "Dermatologist Dr. Lee handled the biopsy and said it was benign.";

    for content in [travel, generic_doctor, dr_patel, dr_smith, dr_lee] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let query = "How many different doctors did I visit?";
    let mut weighted_cues: Vec<(String, f64)> = tokenize_to_cues(query)
        .into_iter()
        .map(|cue| (cue, 1.0))
        .collect();
    let total_memories = engine.total_memories().max(1);
    let intent = compile_query_intent(query, |cue| {
        let df = engine.get_cue_frequency(cue);
        df > 0 && (df <= 16 || df * 5 <= total_memories)
    });

    for (cue, multiplier) in &intent.cue_weight_adjustments {
        if let Some((_, weight)) = weighted_cues
            .iter_mut()
            .find(|(existing, _)| existing == cue)
        {
            *weight *= *multiplier;
        }
    }
    for (cue, weight) in &intent.weighted_cues {
        if let Some((_, existing_weight)) = weighted_cues
            .iter_mut()
            .find(|(existing, _)| existing == cue)
        {
            if *existing_weight < *weight {
                *existing_weight = *weight;
            }
        } else {
            weighted_cues.push((cue.clone(), *weight));
        }
    }
    let results = engine.recall_weighted(
        weighted_cues,
        5,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    let top = results.first().map(|result| result.content.as_str());
    assert!(
        matches!(top, Some(content) if content == dr_patel || content == dr_smith || content == dr_lee),
        "unexpected top result: {top:?}"
    );
    assert!(results
        .iter()
        .take(3)
        .all(|result| result.content != travel && result.content != generic_doctor));
}

#[test]
fn event_count_query_ranks_counted_object_over_generic_recent_activity() {
    let engine = CueMapEngine::new();
    let workshop_today = "User: I'm attending another theater workshop today, focusing on improvisation techniques.";
    let lecture_recent = "User: I attended a lecture series at the National Gallery recently, which was enlightening.";
    let expected_barn = "User: I'm planning my own wedding and I just got back from a friend's wedding last weekend at a rustic barn.";
    let expected_vineyard = "User: I'm getting married soon and I've been to a few weddings recently, including my cousin's wedding at a vineyard.";

    let dated = |date: &str| {
        let mut metadata = HashMap::new();
        metadata.insert("source_date".to_string(), json!(date));
        Some(metadata)
    };

    for (content, date) in [
        (workshop_today, "2023/10/15 (Sun) 07:23"),
        (lecture_recent, "2023/10/15 (Sun) 14:36"),
        (expected_barn, "2023/10/15 (Sun) 19:23"),
        (expected_vineyard, "2023/10/15 (Sun) 05:48"),
    ] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            dated(date),
            MainStats::default(),
            false,
        );
    }

    let query = "How many weddings have I attended in this year?";
    let weighted_cues = compile_weighted_query_at(&engine, query, Some("2023/10/15 (Sun) 23:47"));
    let results = engine.recall_weighted(
        weighted_cues,
        4,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    let top_three = results
        .iter()
        .take(3)
        .map(|result| result.content.as_str())
        .collect::<Vec<_>>();
    assert!(top_three.contains(&expected_barn), "top results: {top_three:?}");
    assert!(top_three.contains(&expected_vineyard), "top results: {top_three:?}");
    assert!(!top_three.contains(&workshop_today), "top results: {top_three:?}");
}

#[test]
fn explicit_month_query_weights_content_month_facet() {
    let intent = compile_query_intent(
        "How many different museums or galleries did I visit in the month of February?",
        |cue| matches!(cue, "content_month:02" | "museum" | "gallery" | "has:date"),
    );

    assert!(intent.labels.contains(&"temporal_window".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "content_month:02" && *weight >= 3.0));
}

#[test]
fn explicit_month_visit_count_ranks_numeric_month_visits() {
    let engine = CueMapEngine::new();
    let january =
        "User: I attended a guided workshop at the Modern Art Museum in January.";
    let expected_museum =
        "User: I took my niece to the Natural History Museum on 2/8 and she loved it.";
    let expected_gallery =
        "User: I recently saw work when I visited The Art Cube on 2/15.";

    for content in [january, expected_museum, expected_gallery] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let results = engine.recall_weighted(
        compile_weighted_query(
            &engine,
            "How many different museums or galleries did I visit in the month of February?",
        ),
        3,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    let top_two = results
        .iter()
        .take(2)
        .map(|result| result.content.as_str())
        .collect::<Vec<_>>();
    assert!(top_two.contains(&expected_museum), "top results: {top_two:?}");
    assert!(top_two.contains(&expected_gallery), "top results: {top_two:?}");
}

#[test]
fn count_query_boosts_structural_scope_terms() {
    let intent = compile_query_intent(
        "How many different types of citrus fruits have I used in my cocktail recipes?",
        |cue| {
            matches!(
                cue,
                "citrus"
                    | "fruit"
                    | "cocktail"
                    | "recipe"
                    | "quantity_object:citrus"
                    | "quantity_object:fruit"
                    | "type:recipe"
                    | "type:ingredient"
                    | "source_role:user"
            )
        },
    );

    assert!(intent.labels.contains(&"count".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "cocktail" && *weight >= 3.0));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "recipe" && *weight >= 3.0));
}

#[test]
fn scoped_count_query_ranks_scope_matches_over_object_only_matches() {
    let engine = CueMapEngine::new();
    let citrus_syrup = "User: I made a citrus honey syrup with orange and lemon for iced tea.";
    let sangria = "User: I served sangria with slices of citrus fruit at the gathering.";
    let expected =
        "User: I used orange bitters in my cocktail recipe, and lime in another cocktail recipe.";

    for content in [citrus_syrup, sangria, expected] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let weighted_cues = compile_weighted_query(
        &engine,
        "How many different types of citrus fruits have I used in my cocktail recipes?",
    );
    let results = engine.recall_weighted(
        weighted_cues,
        3,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    let top = results.first().map(|result| result.content.as_str());
    assert_eq!(top, Some(expected));
}

#[test]
fn duration_count_query_boosts_activity_scope_terms() {
    let intent = compile_query_intent(
        "How many days did I spend on camping trips in the United States this year?",
        |cue| {
            matches!(
                cue,
                "camp"
                    | "trip"
                    | "camp_trip"
                    | "has:duration"
                    | "source_year:2023"
                    | "entity:united_states"
            )
        },
    );

    assert!(intent.labels.contains(&"duration".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "camp_trip" && *weight >= 3.0));
}

#[test]
fn duration_count_query_ranks_scoped_duration_memories_over_entity_only_matches() {
    let engine = CueMapEngine::new();
    let entity_noise = "User: How does the United States Senate hold a filibuster?";
    let duration_noise =
        "User: I've been taking my antibiotics for 10 days now and feel better.";
    let target =
        "User: I just got back from a 5-day camping trip to Yellowstone National Park.";

    for content in [entity_noise, duration_noise, target] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let results = engine.recall_weighted(
        compile_weighted_query(
            &engine,
            "How many days did I spend on camping trips in the United States this year?",
        ),
        3,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(target));
}

#[test]
fn numeric_object_facets_help_count_inventory_query_rank_owned_objects() {
    let engine = CueMapEngine::new();
    let current_noise = "I'm currently on Season 4 Episode 10 and loving this show.";
    let kid_noise = "What are some popular kid-friendly flooring options for heavy traffic?";
    let friend_noise = "I met a friend at a board game cafe and bought the starter set.";
    let one_gallon =
        "I've also been taking care of a small 1-gallon tank that I set up for a friend's kid.";
    let five_gallon = "I have a 5-gallon tank with a solitary betta fish named Finley.";
    let twenty_gallon =
        "I've finally set up my 20-gallon freshwater community tank named Amazonia.";
    let plant_tank = "I've got an anacharis and a java moss in my community tank.";

    for content in [
        current_noise,
        kid_noise,
        friend_noise,
        one_gallon,
        five_gallon,
        twenty_gallon,
        plant_tank,
    ] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let query =
        "How many tanks do I currently have, including the one I set up for my friend's kid?";
    let mut weighted_cues: Vec<(String, f64)> = tokenize_to_cues(query)
        .into_iter()
        .map(|cue| (cue, 1.0))
        .collect();
    let total_memories = engine.total_memories().max(1);
    let intent = compile_query_intent(query, |cue| {
        let df = engine.get_cue_frequency(cue);
        df > 0 && (df <= 16 || df * 5 <= total_memories)
    });

    for (cue, multiplier) in &intent.cue_weight_adjustments {
        if let Some((_, weight)) = weighted_cues
            .iter_mut()
            .find(|(existing, _)| existing == cue)
        {
            *weight *= *multiplier;
        }
    }
    for (cue, weight) in &intent.weighted_cues {
        if let Some((_, existing_weight)) = weighted_cues
            .iter_mut()
            .find(|(existing, _)| existing == cue)
        {
            if *existing_weight < *weight {
                *existing_weight = *weight;
            }
        } else {
            weighted_cues.push((cue.clone(), *weight));
        }
    }

    let results = engine.recall_weighted(
        weighted_cues,
        5,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    let top_contents = results
        .iter()
        .take(3)
        .map(|result| result.content.as_str())
        .collect::<Vec<_>>();
    assert!(top_contents.contains(&one_gallon));
    assert!(top_contents.contains(&five_gallon));
    assert!(top_contents.contains(&twenty_gallon));
    assert!(!top_contents.contains(&current_noise));
    assert!(!top_contents.contains(&kid_noise));
    assert!(!top_contents.contains(&friend_noise));
}

#[test]
fn first_person_inventory_queries_penalize_non_user_source_matches() {
    let engine = CueMapEngine::new();

    let user_owned = "User: I'm looking to find a piano technician to service my Korg B1, which I've had for about 3 years.";
    let assistant_advice = "Assistant: The Korg B1 is a digital piano, not an acoustic piano, so you will want a technician for musical instruments.";

    for content in [user_owned, assistant_advice] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let weighted_cues =
        compile_weighted_query(&engine, "How many musical instruments do I currently own?");
    assert!(weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "source_role:user" && *weight >= 3.0));

    let results = engine.recall_weighted(
        weighted_cues,
        2,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(user_owned));
}

#[test]
fn navigation_facets_help_getting_around_query_rank_transit_resources() {
    let engine = CueMapEngine::new();
    let restaurant =
        "I'm heading to Tokyo soon and was wondering if you could recommend restaurants near my hotel.";
    let shopping =
        "I'm planning to do some shopping while I'm in Tokyo. Can you recommend popular malls?";
    let tofu = "Do you have any tips for making sure tofu gets crispy in the stir-fry?";
    let suica =
        "I'm visiting Tsukiji. What's the best way to get there from Shinjuku Station using my Suica card?";
    let trip_app = "I'm taking a guided tour tomorrow. How can I get to the meeting point using my transit app and rail pass?";

    for content in [restaurant, shopping, tofu, suica, trip_app] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let results = engine.recall_weighted(
        compile_weighted_query(
            &engine,
            "I'm anxious about getting around Tokyo. Do you have any helpful tips?",
        ),
        5,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    let top_two = results
        .iter()
        .take(2)
        .map(|result| result.content.as_str())
        .collect::<Vec<_>>();
    assert!(top_two.contains(&suica));
    assert!(top_two.contains(&trip_app));
    assert!(!top_two.contains(&restaurant));
    assert!(!top_two.contains(&shopping));
    assert!(!top_two.contains(&tofu));
}

#[test]
fn source_session_answer_projection_cues_rank_inline_list_answer_over_user_prompt() {
    let engine = CueMapEngine::new();
    let prompt = "User: Brainstorm ideas for work from home jobs for seniors";
    let answer = "Assistant: 1. Virtual customer service representative 2. Telehealth professional 3. Remote bookkeeper 4. Virtual tutor or teacher 5. Freelance writer or editor 6. Online survey taker 7. Transcriptionist 8. Social media manager";
    let distractor = "Assistant: Giving a presentation to students who want to work in logistics can be valuable. Here are a few tips: 1. Define logistics 2. Explain supply chains 3. Share examples";

    let mut prompt_metadata = HashMap::new();
    prompt_metadata.insert("source_role".to_string(), json!("user"));
    prompt_metadata.insert("source_session_id".to_string(), json!("answer_sharegpt_hA7AkP3_0"));
    engine.add_memory(
        prompt.to_string(),
        tokenize_to_cues(prompt),
        Some(prompt_metadata),
        MainStats::default(),
        false,
    );

    let mut answer_metadata = HashMap::new();
    answer_metadata.insert("source_role".to_string(), json!("assistant"));
    answer_metadata.insert("source_session_id".to_string(), json!("answer_sharegpt_hA7AkP3_0"));
    engine.add_memory(
        answer.to_string(),
        tokenize_to_cues(answer),
        Some(answer_metadata),
        MainStats::default(),
        false,
    );

    let mut distractor_metadata = HashMap::new();
    distractor_metadata.insert("source_role".to_string(), json!("assistant"));
    distractor_metadata.insert("source_session_id".to_string(), json!("answer_sharegpt_other_0"));
    engine.add_memory(
        distractor.to_string(),
        tokenize_to_cues(distractor),
        Some(distractor_metadata),
        MainStats::default(),
        false,
    );

    let results = engine.recall_weighted(
        vec![
            ("source_session:answer_sharegpt_ha7akp3_0".to_string(), 4.0),
            ("source_role:assistant".to_string(), 3.0),
            ("has:list".to_string(), 2.8),
            ("has:number".to_string(), 1.0),
        ],
        5,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(answer));
}

#[test]
fn age_difference_query_ranks_current_and_event_age_evidence() {
    let engine = CueMapEngine::new();
    let old_table = "User: I'm restoring an old oak coffee table and need antique furniture care tips.";
    let workout_duration = "User: I completed a 5K run two Sundays ago in 27 minutes and 12 seconds.";
    let mba_answer = "Assistant: An MBA can be useful for long-term career goals and leadership roles.";
    let current_age = "User: I'm considering pursuing the CDMP certification. As a 32-year-old Digital Marketing Specialist at TechSavvy Inc., I believe it will prepare me for an MBA.";
    let graduation_age = "User: I have a Bachelor's degree in Business Administration from the University of California, Berkeley, which I completed at the age of 25.";

    for content in [old_table, workout_duration, mba_answer, current_age, graduation_age] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let results = engine.recall_weighted(
        compile_weighted_query(
            &engine,
            "How many years older am I than when I graduated from college?",
        ),
        5,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    let top_two = results
        .iter()
        .take(2)
        .map(|result| result.content.as_str())
        .collect::<Vec<_>>();
    assert!(top_two.contains(&current_age));
    assert!(top_two.contains(&graduation_age));
    assert!(!top_two.contains(&old_table));
    assert!(!top_two.contains(&workout_duration));
}

#[test]
fn sibling_count_query_ranks_brother_and_sister_facts() {
    let engine = CueMapEngine::new();
    let board_game = "User: Actually, I have been playing a lot of board games recently.";
    let sweet_tooth = "User: I have a bit of a sweet tooth and want dessert spots nearby.";
    let sister_gift = "User: Can you remind me about the necklace I got for my sister's birthday?";
    let twin_movie = "Assistant: The film follows twin siblings who uncover a family secret.";
    let sisters = "User: I come from a family with 3 sisters, so I've always had a strong female presence in my life.";
    let brother = "User: I should mention that I have a brother, which might be influencing my social circle dynamics.";

    for content in [board_game, sweet_tooth, sister_gift, twin_movie, sisters, brother] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let results = engine.recall_weighted(
        compile_weighted_query(&engine, "What is the total number of siblings I have?"),
        5,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    let top_two = results
        .iter()
        .take(2)
        .map(|result| result.content.as_str())
        .collect::<Vec<_>>();
    assert!(top_two.contains(&sisters));
    assert!(top_two.contains(&brother));
    assert!(!top_two.contains(&board_game));
    assert!(!top_two.contains(&sweet_tooth));
    assert!(!top_two.contains(&sister_gift));
    assert!(!top_two.contains(&twin_movie));
}

#[test]
fn relative_temporal_query_ranks_source_dated_memory_without_exact_domain_word() {
    let engine = CueMapEngine::new();
    let museum = "User: I attended an art museum exhibition three weeks ago and enjoyed the guided tour.";
    let garden_app = "User: I've been using a gardening app to track weather and soil moisture levels.";
    let fertilizer = "User: I attended a gardening workshop recently and learned about companion planting.";
    let expected = "User: I'm looking for advice on keeping my tomato plants healthy and pest-free. I just planted 12 new tomato saplings today.";

    let dated = |date: &str| {
        let mut metadata = HashMap::new();
        metadata.insert("source_date".to_string(), json!(date));
        Some(metadata)
    };

    engine.add_memory(
        museum.to_string(),
        tokenize_to_cues(museum),
        dated("2023/04/14 (Fri) 12:00"),
        MainStats::default(),
        false,
    );
    engine.add_memory(
        garden_app.to_string(),
        tokenize_to_cues(garden_app),
        dated("2023/05/01 (Mon) 12:00"),
        MainStats::default(),
        false,
    );
    engine.add_memory(
        fertilizer.to_string(),
        tokenize_to_cues(fertilizer),
        dated("2023/04/28 (Fri) 12:00"),
        MainStats::default(),
        false,
    );
    engine.add_memory(
        "User: hello".to_string(),
        tokenize_to_cues("User: hello"),
        dated("2023/04/21 (Fri) 00:29"),
        MainStats::default(),
        false,
    );
    engine.add_memory(
        "User: I will provide context data in the next several queries.".to_string(),
        tokenize_to_cues("User: I will provide context data in the next several queries."),
        dated("2023/04/21 (Fri) 00:29"),
        MainStats::default(),
        false,
    );
    engine.add_memory(
        expected.to_string(),
        tokenize_to_cues(expected),
        dated("2023/04/21 (Fri) 00:30"),
        MainStats::default(),
        false,
    );
    engine.add_memory(
        "User: I recently attended a documentary filmmaking panel and completed a workshop."
            .to_string(),
        tokenize_to_cues(
            "User: I recently attended a documentary filmmaking panel and completed a workshop.",
        ),
        None,
        MainStats::default(),
        false,
    );

    let results = engine.recall_weighted(
        compile_weighted_query_at(
            &engine,
            "What gardening-related activity did I do two weeks ago?",
            Some("2023/05/05 (Fri) 16:42"),
        ),
        5,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(expected));
}

#[test]
fn religious_activity_query_ranks_religious_service_over_generic_attendance() {
    let engine = CueMapEngine::new();

    let add = |content: &str, date: &str| {
        let mut metadata = HashMap::new();
        metadata.insert("source_role".to_string(), json!("user"));
        metadata.insert("source_date".to_string(), json!(date));
        let cues = [
            tokenize_to_cues(content),
            extract_memory_facets(content, Some(&metadata), &[]),
        ]
        .concat();
        engine.add_memory(
            content.to_string(),
            cues,
            Some(metadata),
            MainStats::default(),
            false,
        );
    };

    let museum = "User: I attended a ceramics workshop at the Museum of Craft and Design.";
    let volunteer = "User: I helped out at an Easter Egg Hunt event last week.";
    let expected =
        "User: I attended the Maundy Thursday service at the Episcopal Church last week.";

    add(museum, "2023/03/26 (Sun) 21:45");
    add(volunteer, "2023/04/06 (Thu) 12:00");
    add(expected, "2023/04/06 (Thu) 05:36");

    let results = engine.recall_weighted(
        compile_weighted_query_at(
            &engine,
            "Where did I attend the religious activity last week?",
            Some("2023/04/10 (Mon) 12:00"),
        ),
        3,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(expected));
}

#[test]
fn streaming_service_query_ranks_media_usage_over_recent_noise() {
    let engine = CueMapEngine::new();

    let add = |content: &str, date: &str| {
        let mut metadata = HashMap::new();
        metadata.insert("source_role".to_string(), json!("user"));
        metadata.insert("source_date".to_string(), json!(date));
        let cues = [
            tokenize_to_cues(content),
            extract_memory_facets(content, Some(&metadata), &[]),
        ]
        .concat();
        engine.add_memory(
            content.to_string(),
            cues,
            Some(metadata),
            MainStats::default(),
            false,
        );
    };

    let recent_noise =
        "User: I recently attended a writing workshop and want more prompts for memoir writing.";
    let netflix =
        "User: I've been using Netflix, Hulu, and Amazon Prime for the past 6 months while looking for new shows to watch.";
    let disney = "User: I saw a documentary on Disney+ during my free trial last month.";
    let apple =
        "User: I've also been using Apple TV+ for a few months now and finished watching For All Mankind.";

    add(recent_noise, "2023/05/26 (Fri) 23:59");
    add(netflix, "2023/05/26 (Fri) 08:25");
    add(disney, "2023/05/26 (Fri) 01:08");
    add(apple, "2023/05/26 (Fri) 23:40");

    let results = engine.recall_weighted(
        compile_weighted_query(
            &engine,
            "Which streaming service did I start using most recently?",
        ),
        4,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    let top_three = results
        .iter()
        .take(3)
        .map(|result| result.content.as_str())
        .collect::<Vec<_>>();
    assert!(top_three.contains(&netflix));
    assert!(top_three.contains(&disney));
    assert!(top_three.contains(&apple));
    assert!(!top_three.contains(&recent_noise));
}

#[test]
fn music_streaming_service_query_prefers_music_usage_over_video_streaming() {
    let engine = CueMapEngine::new();

    let netflix =
        "User: I've been using Netflix for a while now and watching more original content lately.";
    let documentaries = "User: I've been meaning to try out some documentaries on Netflix.";
    let target =
        "User: I've been listening to their songs a lot on Spotify lately.";

    for content in [netflix, documentaries, target] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let results = engine.recall_weighted(
        compile_weighted_query(
            &engine,
            "What is the name of the music streaming service have I been using lately?",
        ),
        3,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(target));
}

#[test]
fn counted_item_facets_rank_inventory_counts_over_generic_collection_mentions() {
    let engine = CueMapEngine::new();
    let generic = "I'm thinking of adding decorations to the 20-gallon tank to create more hiding places for the fish.";
    let counted = "My new 20-gallon tank currently has 10 neon tetras, 5 golden honey gouramis, and a small pleco catfish.";

    for content in [generic, counted] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let mut weighted_cues: Vec<(String, f64)> =
        tokenize_to_cues("How many fish are in my tank?").into_iter().map(|cue| (cue, 1.0)).collect();
    let intent = compile_query_intent("How many fish are in my tank?", |cue| {
        engine.get_cue_frequency(cue) > 0
    });
    for (cue, weight) in intent.weighted_cues {
        if let Some((_, existing)) = weighted_cues.iter_mut().find(|(existing, _)| existing == &cue)
        {
            if *existing < weight {
                *existing = weight;
            }
        } else {
            weighted_cues.push((cue, weight));
        }
    }

    let results = engine.recall_weighted(
        weighted_cues,
        2,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(counted));
}

#[test]
fn current_book_query_prefers_current_reading_over_finished_book_mentions() {
    let engine = CueMapEngine::new();
    let finished = "User: We're going to discuss \"The Last House Guest\" by Megan Miranda, which I've already read and really enjoyed.";
    let reading_habit = "User: I love making reading a habit loop and read before bed every night.";
    let target = "User: I'm currently devouring \"The Seven Husbands of Evelyn Hugo\" and it's hard to put down.";

    for content in [finished, reading_habit, target] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let results = engine.recall_weighted(
        compile_weighted_query(&engine, "What book am I currently reading?"),
        3,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(target));
}

#[test]
fn temporal_order_query_ranks_visited_events_over_latest_update_noise() {
    let engine = CueMapEngine::new();

    let add = |content: &str, date: &str| {
        let mut metadata = HashMap::new();
        metadata.insert("source_role".to_string(), json!("user"));
        metadata.insert("source_date".to_string(), json!(date));
        let cues = [
            tokenize_to_cues(content),
            extract_memory_facets(content, Some(&metadata), &[]),
        ]
        .concat();
        engine.add_memory(
            content.to_string(),
            cues,
            Some(metadata),
            MainStats::default(),
            false,
        );
    };

    let latest_noise = "User: I'm planning to attend another art-related event soon and need the latest trends and exhibition updates.";
    let science = "User: I visited the Science Museum's Space Exploration exhibition today.";
    let history = "User: I participated in a behind-the-scenes tour of the Museum of History's conservation lab today.";

    add(latest_noise, "2023/03/10 (Fri) 10:00");
    add(science, "2023/01/15 (Sun) 16:31");
    add(history, "2023/02/15 (Wed) 12:20");

    let results = engine.recall_weighted(
        compile_weighted_query(
            &engine,
            "What is the order of the six museums I visited from earliest to latest?",
        ),
        3,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert!(results[0].content.contains("Science Museum"));
    assert!(results
        .iter()
        .take(2)
        .any(|result| result.content.contains("Museum of History")));
    assert!(!results[0].content.contains("latest trends"));
}

#[test]
fn sports_event_order_query_ranks_competitions_in_past_month() {
    let engine = CueMapEngine::new();

    let add = |content: &str, date: &str| {
        let mut metadata = HashMap::new();
        metadata.insert("source_role".to_string(), json!("user"));
        metadata.insert("source_date".to_string(), json!(date));
        let cues = [
            tokenize_to_cues(content),
            extract_memory_facets(content, Some(&metadata), &[]),
        ]
        .concat();
        engine.add_memory(
            content.to_string(),
            cues,
            Some(metadata),
            MainStats::default(),
            false,
        );
    };

    let old_progress = "User: I've been tracking body fat percentage and made progress over the past month.";
    let old_workout = "User: I've been thinking about trying a quick 20-minute workout before work.";
    let soccer = "User: I participate in the company's annual charity soccer tournament today.";
    let triathlon = "User: I just completed the Spring Sprint Triathlon today, which included a 20K bike ride.";
    let run = "User: I just finished a 5K run with a personal best time at the Midsummer 5K Run.";

    add(old_progress, "2023/05/12 (Fri) 15:34");
    add(old_workout, "2023/05/21 (Sun) 12:34");
    add(triathlon, "2023/06/02 (Fri) 15:29");
    add(run, "2023/06/10 (Sat) 15:00");
    add(soccer, "2023/06/17 (Sat) 11:09");

    let results = engine.recall_weighted(
        compile_weighted_query_at(
            &engine,
            "What is the order of the three sports events I participated in during the past month, from earliest to latest?",
            Some("2023/07/01 (Sat) 20:43"),
        ),
        5,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    let top_three = results
        .iter()
        .take(3)
        .map(|result| result.content.as_str())
        .collect::<Vec<_>>();
    assert!(top_three.contains(&triathlon));
    assert!(top_three.contains(&run));
    assert!(top_three.contains(&soccer));
    assert!(!top_three.contains(&old_progress));
    assert!(!top_three.contains(&old_workout));
}

#[test]
fn relative_charity_event_query_prefers_matching_date_and_domain() {
    let engine = CueMapEngine::new();

    let add = |content: &str, date: &str| {
        let mut metadata = HashMap::new();
        metadata.insert("source_role".to_string(), json!("user"));
        metadata.insert("source_date".to_string(), json!(date));
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            Some(metadata),
            MainStats::default(),
            false,
        );
    };

    let wrong_date = "User: I just got back from the \"24-Hour Bike Ride\" charity event, where I cycled for 4 hours.";
    let same_date_noise = "User: I've played Ticket to Ride and Settlers of Catan before, and they're great games.";
    let target = "User: I just did the \"Walk for Hunger\" charity event today with my colleagues from work.";

    add(wrong_date, "2023/02/14 (Tue) 06:22");
    add(same_date_noise, "2023/03/19 (Sun) 04:24");
    add(target, "2023/03/19 (Sun) 15:44");

    let results = engine.recall_weighted(
        compile_weighted_query_at(
            &engine,
            "What charity event did I participate in a month ago?",
            Some("2023/04/18 (Tue) 18:34"),
        ),
        3,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(target));
}

#[test]
fn extracts_companion_facets_from_first_person_event_language() {
    let facets = extract_memory_facets(
        "User: I just saw Queen live with Adam Lambert at the Prudential Center with my parents.",
        None,
        &[],
    );

    assert!(facets.contains(&"type:activity".to_string()));
    assert!(facets.contains(&"companion:with".to_string()));
}

#[test]
fn companion_query_weights_with_companion_evidence() {
    let available = |cue: &str| matches!(cue, "companion:with" | "source_role:user" | "type:activity" | "type:event");
    let intent = compile_query_intent(
        "Who did I go with to the music event last Saturday?",
        available,
    );

    assert!(intent.labels.contains(&"companion".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "companion:with" && *weight >= 5.0));
}

#[test]
fn companion_query_prefers_events_with_with_companion_evidence() {
    let engine = CueMapEngine::new();

    let add = |content: &str, date: &str| {
        let mut metadata = HashMap::new();
        metadata.insert("source_role".to_string(), json!("user"));
        metadata.insert("source_date".to_string(), json!(date));
        let cues = [
            tokenize_to_cues(content),
            extract_memory_facets(content, Some(&metadata), &[]),
        ]
        .concat();
        engine.add_memory(
            content.to_string(),
            cues,
            Some(metadata),
            MainStats::default(),
            false,
        );
    };

    let generic_music =
        "User: I've been listening to a lot of music and planning to check out jazz clubs.";
    let target = "User: I just saw Queen live with Adam Lambert at the Prudential Center with my parents, and I want more classic rock playlists.";
    let friend_festival =
        "User: I went to a music festival in Brooklyn with a group of friends recently.";

    add(generic_music, "2023/05/20 (Sat) 12:00");
    add(friend_festival, "2023/05/22 (Mon) 12:00");
    add(target, "2023/05/27 (Sat) 21:00");

    let results = engine.recall_weighted(
        compile_weighted_query_at(
            &engine,
            "Who did I go with to the music event last Saturday?",
            Some("2023/06/03 (Sat) 12:00"),
        ),
        3,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert!(results.iter().take(2).any(|result| result.content == target));
    assert!(!results[0].content.contains("jazz clubs"));
}

#[test]
fn extracts_completed_clean_facets_from_first_person_action_language() {
    let facets = extract_memory_facets(
        "User: I'm glad I finally got around to cleaning my white Adidas sneakers last month.",
        None,
        &[],
    );

    assert!(facets.contains(&"type:activity".to_string()));
    assert!(facets.contains(&"completed_action:clean".to_string()));
}

#[test]
fn completed_clean_query_weights_completed_action_evidence() {
    let available = |cue: &str| {
        matches!(
            cue,
            "completed_action:clean" | "source_role:user" | "type:activity" | "type:event"
        )
    };
    let intent = compile_query_intent("Which pair of shoes did I clean last month?", available);

    assert!(intent.labels.contains(&"completed_action".to_string()));
    assert!(intent
        .weighted_cues
        .iter()
        .any(|(cue, weight)| cue == "completed_action:clean" && *weight >= 5.0));
}

#[test]
fn completed_clean_query_prefers_completed_cleaning_over_cleaning_advice() {
    let engine = CueMapEngine::new();

    let add = |content: &str, date: &str| {
        let mut metadata = HashMap::new();
        metadata.insert("source_role".to_string(), json!("user"));
        metadata.insert("source_date".to_string(), json!(date));
        let cues = [
            tokenize_to_cues(content),
            extract_memory_facets(content, Some(&metadata), &[]),
        ]
        .concat();
        engine.add_memory(
            content.to_string(),
            cues,
            Some(metadata),
            MainStats::default(),
            false,
        );
    };

    let advice = "User: What's the best way to clean and maintain my new hiking boots?";
    let lent = "User: I lent my spare pair of running shoes to my sister a few weeks ago.";
    let target = "User: I'm glad I finally got around to cleaning my white Adidas sneakers last month.";

    add(lent, "2023/04/11 (Tue) 12:00");
    add(advice, "2023/05/05 (Fri) 12:00");
    add(target, "2023/05/21 (Sun) 12:00");

    let results = engine.recall_weighted(
        compile_weighted_query_at(
            &engine,
            "Which pair of shoes did I clean last month?",
            Some("2023/06/15 (Thu) 12:00"),
        ),
        3,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|result| result.content.as_str()), Some(target));
}

#[test]
fn transport_mode_comparison_ranks_ride_events_over_habitual_transport_mentions() {
    let engine = CueMapEngine::new();

    let add = |content: &str, date: &str| {
        let mut metadata = HashMap::new();
        metadata.insert("source_role".to_string(), json!("user"));
        metadata.insert("source_date".to_string(), json!(date));
        let cues = [
            tokenize_to_cues(content),
            extract_memory_facets(content, Some(&metadata), &[]),
        ]
        .concat();
        engine.add_memory(
            content.to_string(),
            cues,
            Some(metadata),
            MainStats::default(),
            false,
        );
    };

    let generic = "User: I have been tracking my modes of transport and have been taking more trains and buses instead of driving.";
    let bus = "User: I just got back from a bus ride to attend a friend's wedding today.";
    let train = "User: I took a train ride to visit my family today, and it was a nice 2-hour journey.";

    add(generic, "2023/03/03 (Fri) 19:17");
    add(bus, "2023/02/27 (Mon) 06:17");
    add(train, "2023/03/03 (Fri) 19:17");

    let results = engine.recall_weighted(
        compile_weighted_query(
            &engine,
            "Which mode of transport did I use most recently, a bus or a train?",
        ),
        3,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    let top_two = results
        .iter()
        .take(2)
        .map(|result| result.content.as_str())
        .collect::<Vec<_>>();
    assert!(top_two.contains(&bus));
    assert!(top_two.contains(&train));
    assert!(!top_two.contains(&generic));
}

#[test]
fn milestone_temporal_query_ranks_first_client_contract() {
    let engine = CueMapEngine::new();
    let instagram = "User: I recently collaborated with an influencer who promoted my product to 10,000 followers.";
    let plant = "User: I recently repotted my spider plant 3 weeks ago and it is doing better.";
    let expected = "User: I'm looking for advice on creating a solid contract for my freelance clients. I just signed a contract with my first client today.";

    let dated = |date: &str| {
        let mut metadata = HashMap::new();
        metadata.insert("source_date".to_string(), json!(date));
        Some(metadata)
    };

    engine.add_memory(
        instagram.to_string(),
        tokenize_to_cues(instagram),
        dated("2023/02/28 (Tue) 12:00"),
        MainStats::default(),
        false,
    );
    engine.add_memory(
        plant.to_string(),
        tokenize_to_cues(plant),
        dated("2023/03/01 (Wed) 12:00"),
        MainStats::default(),
        false,
    );
    engine.add_memory(
        expected.to_string(),
        tokenize_to_cues(expected),
        dated("2023/03/01 (Wed) 02:43"),
        MainStats::default(),
        false,
    );

    let results = engine.recall_weighted(
        compile_weighted_query_at(
            &engine,
            "What was the significant buisiness milestone I mentioned four weeks ago?",
            Some("2023/03/28 (Tue) 20:35"),
        ),
        5,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(expected));
}

#[test]
fn purchase_temporal_query_ranks_acquired_object_memory() {
    let engine = CueMapEngine::new();
    let wood = "User: I'm thinking of trying out a mix of hickory and apple wood today.";
    let jacket = "User: I also have a blue denim jacket from Zara that I've been loving lately.";
    let expected = "User: I'm looking for BBQ sauce recipes. By the way, I just got a smoker today and I'm excited to experiment with different woods.";

    let dated = |date: &str| {
        let mut metadata = HashMap::new();
        metadata.insert("source_date".to_string(), json!(date));
        Some(metadata)
    };

    engine.add_memory(
        wood.to_string(),
        tokenize_to_cues(wood),
        dated("2023/03/15 (Wed) 09:00"),
        MainStats::default(),
        false,
    );
    engine.add_memory(
        jacket.to_string(),
        tokenize_to_cues(jacket),
        dated("2023/03/14 (Tue) 09:00"),
        MainStats::default(),
        false,
    );
    engine.add_memory(
        expected.to_string(),
        tokenize_to_cues(expected),
        dated("2023/03/15 (Wed) 10:00"),
        MainStats::default(),
        false,
    );

    let results = engine.recall_weighted(
        compile_weighted_query_at(
            &engine,
            "What kitchen appliance did I buy 10 days ago?",
            Some("2023/03/25 (Sat) 20:00"),
        ),
        5,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(expected));
}

#[test]
fn homegrown_ingredient_query_ranks_harvested_garden_memory() {
    let engine = CueMapEngine::new();
    let cocktail =
        "User: I'm looking for inspiration for new cocktail recipes and ingredients this weekend.";
    let family_dinner =
        "User: I'm trying to plan a family dinner and need healthy meal ideas.";
    let expected = "User: I've been using basil and mint in my cooking lately. I've even harvested some cherry tomatoes from my garden. Do you have suggestions for companion plants?";
    let garden_noise =
        "User: I've been thinking about introducing beneficial insects to my garden.";

    for content in [cocktail, family_dinner, expected, garden_noise] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let results = engine.recall_weighted(
        compile_weighted_query(
            &engine,
            "What should I serve for dinner this weekend with my homegrown ingredients?",
        ),
        4,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(expected));
    assert_ne!(results.first().map(|r| r.content.as_str()), Some(cocktail));
}

#[test]
fn generated_facets_do_not_create_bare_value_aliases() {
    let engine = CueMapEngine::new();
    let mut metadata = HashMap::new();
    metadata.insert("source_date".to_string(), json!("2023/04/21 (Fri) 00:30"));
    engine.add_memory(
        "Doctor: As a 32-year-old commuter, I currently practice 20 minutes daily, paid $15 yesterday, prefer Fuji X100V, use a transit app, and I have a brother."
            .to_string(),
        vec![],
        Some(metadata),
        MainStats::default(),
        false,
    );

    assert_eq!(engine.get_cue_frequency("source_role:doctor"), 1);
    assert_eq!(engine.get_cue_frequency("source_date:2023_04_21"), 1);
    assert_eq!(engine.get_cue_frequency("source_week:2023_w16"), 1);
    assert_eq!(engine.get_cue_frequency("has:number"), 1);
    assert_eq!(engine.get_cue_frequency("has:money"), 1);
    assert_eq!(engine.get_cue_frequency("has:duration"), 1);
    assert_eq!(engine.get_cue_frequency("has:age"), 1);
    assert_eq!(engine.get_cue_frequency("age:current"), 1);
    assert_eq!(engine.get_cue_frequency("type:preference"), 1);
    assert_eq!(engine.get_cue_frequency("type:navigation"), 1);
    assert_eq!(engine.get_cue_frequency("travel:app"), 1);
    assert_eq!(engine.get_cue_frequency("family_relation:sibling"), 1);
    assert_eq!(engine.get_cue_frequency("family_count:sibling"), 1);
    assert_eq!(engine.get_cue_frequency("sibling_kind:brother"), 1);
    assert_eq!(engine.get_cue_frequency("entity:fuji_x100v"), 1);

    for leaked_alias in [
        "doctor",
        "2023_04_21",
        "2023_w16",
        "number",
        "money",
        "duration",
        "age",
        "current",
        "preference",
        "navigation",
        "app",
        "sibling",
        "brother",
        "fuji_x100v",
    ] {
        assert_eq!(
            engine.get_cue_frequency(leaked_alias),
            0,
            "generated facet leaked bare alias: {leaked_alias}"
        );
    }
}

#[test]
fn structured_facet_only_match_does_not_dominate_lexical_match() {
    let engine = CueMapEngine::new();
    let relevant = "User: I practice guitar daily.";
    let distractor = "User: I paid $15 at the market yesterday.";

    engine.add_memory(
        distractor.to_string(),
        tokenize_to_cues(distractor),
        None,
        MainStats::default(),
        false,
    );
    engine.add_memory(
        relevant.to_string(),
        tokenize_to_cues(relevant),
        None,
        MainStats::default(),
        false,
    );

    let results = engine.recall_weighted(
        vec![("guitar".to_string(), 1.0), ("has:money".to_string(), 3.5)],
        2,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(relevant));
}

#[test]
fn weak_recommendation_scaffold_does_not_outrank_topic_match() {
    let engine = CueMapEngine::new();
    let relevant = "User: Besides great views, I also like hotels with unique features, such as a rooftop pool or hot tub.";
    let distractor =
        "User: Can you suggest how to design a gamified conference challenge with hints?";

    engine.add_memory(
        distractor.to_string(),
        tokenize_to_cues(distractor),
        None,
        MainStats::default(),
        false,
    );
    engine.add_memory(
        relevant.to_string(),
        tokenize_to_cues(relevant),
        None,
        MainStats::default(),
        false,
    );

    let results = engine.recall_weighted(
        compile_weighted_query(
            &engine,
            "Can you suggest a hotel for my upcoming trip to Miami?",
        ),
        2,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|result| result.content.as_str()), Some(relevant));
}

#[test]
fn generic_question_distractors_do_not_beat_specific_lexical_match() {
    let engine = CueMapEngine::new();
    let relevant = "User: I graduated with a degree in Business Administration.";
    let distractor = "User: What math problem do they solve?";

    engine.add_memory(
        distractor.to_string(),
        tokenize_to_cues(distractor),
        None,
        MainStats::default(),
        false,
    );
    engine.add_memory(
        relevant.to_string(),
        tokenize_to_cues(relevant),
        None,
        MainStats::default(),
        false,
    );

    let results = engine.recall(
        tokenize_to_cues("What degree did I graduate with?"),
        2,
        false,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(relevant));
}

#[test]
fn undergraduate_degree_query_prefers_matching_level_and_field_initialism() {
    let engine = CueMapEngine::new();
    let target =
        "User: I completed my undergrad in CS from UCLA before starting my first developer job.";
    let masters_distractor =
        "User: I am pursuing a Master's degree in Data Science while taking evening classes.";
    let computer_science_distractor =
        "Assistant: Computer Science programs often cover algorithms, systems, and theory.";

    for content in [masters_distractor, computer_science_distractor, target] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let results = engine.recall_weighted(
        compile_weighted_query(
            &engine,
            "Where did I complete my Bachelor's degree in Computer Science?",
        ),
        3,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(target));
}

#[test]
fn shopping_advice_query_prefers_first_person_purchase_consideration() {
    let engine = CueMapEngine::new();
    let tuning = "User: What are the differences between open D tuning and standard tuning?";
    let genre = "User: What are the most common types of music that people play on a Les Paul?";
    let target = "User: I'm considering upgrading from a Fender Stratocaster to a Gibson Les Paul. Can you tell me the main differences between these two guitars?";

    for content in [tuning, genre, target] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let results = engine.recall_weighted(
        compile_weighted_query(
            &engine,
            "I'm getting excited about my visit to the music store this weekend. Any tips on what to look for in a new guitar?",
        ),
        3,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(target));
}

#[test]
fn assistant_created_second_version_query_prefers_iteration_answer() {
    let engine = CueMapEngine::new();
    let user_prompt = "User: Create a sad song with notes";
    let first_song = "Assistant: Here's a sad song with notes for you:\n\nVerse 1:\nC D E E E D C C\nThe rain falls down on me\n\nChorus:\nG G G G A G F\nWhy did you leave?";
    let theory = "Assistant: Understanding music theory will help you create chord progressions and songs.";
    let target = "Assistant: Sure, here's a more romantic and heart-felt song for you:\n\nVerse 1:\nG A B C D E D C B A G\nWhen I first saw you\n\nChorus:\nC D E F G A B A G F E D C\nYou're the one I want";

    for content in [user_prompt, first_song, theory, target] {
        engine.add_memory(
            content.to_string(),
            tokenize_to_cues(content),
            None,
            MainStats::default(),
            false,
        );
    }

    let results = engine.recall_weighted(
        compile_weighted_query(
            &engine,
            "I'm looking back at our previous conversation where you created two sad songs for me. Can you remind me what was the chord progression for the chorus in the second song?",
        ),
        4,
        false,
        None,
        1,
        true,
        true,
        None,
        None,
    );

    assert_eq!(results.first().map(|r| r.content.as_str()), Some(target));
}
