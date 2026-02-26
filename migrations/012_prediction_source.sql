-- Migration 012: Add source column to prediction_results
-- Tracks origin (interactive vs sleep_loop) to filter PCI contamination
-- Proposal #42: Prediction-Consolidation Interference fix

ALTER TABLE prediction_results ADD COLUMN source TEXT DEFAULT 'interactive';
