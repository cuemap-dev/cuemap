    use super::*;

    #[test]
    fn test_ingestion_counter() {
        let metrics = MetricsCollector::new();
        assert_eq!(metrics.ingestion_count.load(Ordering::Relaxed), 0);

        metrics.record_ingestion();
        metrics.record_ingestion();

        assert_eq!(metrics.ingestion_count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_recall_counter_and_latency() {
        let metrics = MetricsCollector::new();

        metrics.record_recall(1.0);
        metrics.record_recall(2.0);
        metrics.record_recall(10.0);

        assert_eq!(metrics.recall_count.load(Ordering::Relaxed), 3);

        // With only 3 samples, P99 should be the max
        let p99 = metrics.get_p99_latency();
        assert!((p99 - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_avg_latency() {
        let metrics = MetricsCollector::new();

        metrics.record_recall(1.0);
        metrics.record_recall(2.0);
        metrics.record_recall(3.0);

        let avg = metrics.get_avg_latency();
        assert!((avg - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_empty_latencies() {
        let metrics = MetricsCollector::new();

        assert_eq!(metrics.get_p99_latency(), 0.0);
        assert_eq!(metrics.get_avg_latency(), 0.0);
        assert_eq!(metrics.get_sample_count(), 0);
    }
